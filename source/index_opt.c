/*
 * This file is part of vitaGL
 *
 * Index reordering for the SGX vertex index CAM (HAVE_INDEX_REORDER).
 *
 * Each SGX core keeps the last 12 indices of an indexed draw and skips vertex
 * shading for an index that matches one of them (PS Vita GPU User's Guide,
 * "Vertex Indices and Vertex Input Attributes"). Index lists ordered for a
 * larger LRU cache (desktop or older iOS tools) reuse much less in a window
 * that short, so vertex-heavy scenes shade most vertices several times.
 *
 * For a triangle-list draw from a GL_STATIC_DRAW element buffer, this module
 * builds, once, a copy of the draw's index range with the same triangles
 * (each keeping its own vertex order, so winding is unchanged) in an order
 * that hits the 12-entry window more often, and the draw uses that copy.
 * Only draws whose result cannot depend on the order of their own triangles
 * use it: blending off, stencil test off, depth test on with LESS or LEQUAL
 * (only fragments at exactly equal depth inside one draw could resolve
 * differently).
 *
 * A copy is keyed by the buffer, its current store, the byte offset and the
 * count, and is dropped when the buffer's data changes (glBufferData,
 * glBufferSubData, map, delete). The work is limited per frame; draws wait
 * with their original indices until their turn.
 */
#include "shared.h"

#ifdef HAVE_INDEX_REORDER

#ifdef INDEX_OPT_DUMP_ON
#define INDEX_OPT_DUMP "host0:/bioshock-idx.bin"
#endif
#define IO_SETS 2048u
#define IO_WAYS 4u
#define IO_MIN_INDICES 96
#define IO_FRAME_BUDGET 45000u
#define IO_BYTES_CAP (8u << 20)
#define IO_WINDOW 12u

typedef struct {
	const vbo *buf;
	const void *store;
	uint32_t offset, count, last_frame;
	uint32_t orig_misses, misses;
	uint16_t *opt; // NULL: kept the original order
} io_entry;

static io_entry io_table[IO_SETS][IO_WAYS];
static uint32_t io_bloom[128]; // buffers that may have entries
static uint32_t io_bytes, io_budget_frame, io_budget_used, io_model = 0xffffffffu;
static uint32_t io_stats_frame, io_stats[9]; // draws, indices, orig misses, misses, built, kept, no-colour idx, no-depth-write idx, repeat idx

extern unsigned ml_experiment_knobs2 __attribute__((weak));

static inline uint32_t io_hash_buf(const vbo *buf) {
	return ((uint32_t)buf >> 3) * 2654435761u;
}

static void io_drop(io_entry *e) {
	if (e->opt) {
		mark_as_dirty(e->opt);
		io_bytes -= e->count * 2u;
	}
	e->buf = NULL;
	e->opt = NULL;
}

void vgl_index_opt_invalidate(const vbo *buf) {
	const uint32_t h = io_hash_buf(buf) >> 20; // 12 bits
	if (!(io_bloom[h >> 5] & (1u << (h & 31))))
		return;
	for (uint32_t s = 0; s < IO_SETS; s++)
		for (uint32_t w = 0; w < IO_WAYS; w++)
			if (io_table[s][w].buf == buf)
				io_drop(&io_table[s][w]);
}

static void io_flush(void) {
	for (uint32_t s = 0; s < IO_SETS; s++)
		for (uint32_t w = 0; w < IO_WAYS; w++)
			if (io_table[s][w].buf)
				io_drop(&io_table[s][w]);
	sceClibMemset(io_bloom, 0, sizeof(io_bloom));
}

// Scratch, grown on demand (only the GL thread gets here)
static uint32_t *s_off, *s_stamp, *s_tris;
static uint16_t *s_live, *s_copy;
static uint8_t *s_done;
static uint32_t s_vcap, s_icap;

static int io_reserve(uint32_t span, uint32_t count) {
	if (span > s_vcap) {
		free(s_off);
		free(s_stamp);
		free(s_live);
		s_off = malloc((span + 1) * 4);
		s_stamp = malloc(span * 4);
		s_live = malloc(span * 2);
		s_vcap = (s_off && s_stamp && s_live) ? span : 0;
		if (!s_vcap)
			return 0;
	}
	if (count > s_icap) {
		free(s_tris);
		free(s_copy);
		free(s_done);
		s_tris = malloc(count * 4);
		s_copy = malloc(count * 2);
		s_done = malloc(count / 3 + 1);
		s_icap = (s_tris && s_copy && s_done) ? count : 0;
		if (!s_icap)
			return 0;
	}
	return 1;
}

// Shading misses of an order: fifo 0 = the last IO_WINDOW indices, 1 = a FIFO of the
// last IO_WINDOW shaded vertices. Uses s_stamp (span entries, indices rebased by base).
static uint32_t io_misses(const uint16_t *idx, uint32_t count, uint32_t base, uint32_t span, int fifo) {
	sceClibMemset(s_stamp, 0, span * 4);
	uint32_t now = 0, misses = 0;
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t v = idx[i] - base;
		const int hit = s_stamp[v] && now - s_stamp[v] < IO_WINDOW;
		if (!hit)
			misses++;
		if (!fifo || !hit)
			s_stamp[v] = ++now;
	}
	return misses;
}

// Greedy reorder of whole triangles: next is the not yet emitted triangle around the
// last three triangles' vertices with the fewest misses (ties: the one that finishes
// more vertices), else the first not yet emitted one in the original order.
static uint32_t io_reorder(const uint16_t *in, uint32_t count, uint16_t *out, uint32_t base, uint32_t span, int fifo) {
	const uint32_t n = count / 3;
	sceClibMemset(s_off, 0, (span + 1) * 4);
	for (uint32_t i = 0; i < n * 3; i++)
		s_off[in[i] - base + 1]++;
	for (uint32_t v = 0; v < span; v++) {
		s_live[v] = (uint16_t)(s_off[v + 1] > 0xffff ? 0xffff : s_off[v + 1]);
		s_off[v + 1] += s_off[v];
	}
	// s_stamp as fill cursors
	sceClibMemcpy(s_stamp, s_off, span * 4);
	for (uint32_t t = 0; t < n; t++)
		for (uint32_t k = 0; k < 3; k++)
			s_tris[s_stamp[in[t * 3 + k] - base]++] = t;
	sceClibMemset(s_stamp, 0, span * 4);
	sceClibMemset(s_done, 0, n);

	uint32_t recent[9], nrecent = 0, scan = 0, now = 0, misses = 0;
	for (uint32_t e = 0; e < n; e++) {
		int32_t best = -1, best_score = 0x7fffffff;
		for (uint32_t r = 0; r < nrecent; r++) {
			const uint32_t v = recent[r];
			for (uint32_t j = s_off[v]; j < s_off[v + 1]; j++) {
				const uint32_t t = s_tris[j];
				if (s_done[t])
					continue;
				int32_t score = 0;
				for (uint32_t k = 0; k < 3; k++) {
					const uint32_t u = in[t * 3 + k] - base;
					if (!(s_stamp[u] && now - s_stamp[u] < IO_WINDOW))
						score += 4;
					if (s_live[u] == 1)
						score -= 1;
				}
				if (score < best_score) {
					best_score = score;
					best = (int32_t)t;
				}
			}
		}
		if (best < 0) {
			while (s_done[scan])
				scan++;
			best = (int32_t)scan;
		}
		s_done[best] = 1;
		if (nrecent == 9) {
			for (uint32_t r = 0; r < 6; r++)
				recent[r] = recent[r + 3];
			nrecent = 6;
		}
		for (uint32_t k = 0; k < 3; k++) {
			const uint16_t iv = in[best * 3 + k];
			const uint32_t u = iv - base;
			const int hit = s_stamp[u] && now - s_stamp[u] < IO_WINDOW;
			if (!hit)
				misses++;
			if (!fifo || !hit)
				s_stamp[u] = ++now;
			if (s_live[u])
				s_live[u]--;
			out[e * 3 + k] = iv;
			recent[nrecent++] = u;
		}
	}
	for (uint32_t i = n * 3; i < count; i++)
		out[i] = in[i];
	return misses + (count - n * 3);
}

static void io_build(io_entry *e, const uint16_t *src, uint32_t count, int fifo) {
	e->orig_misses = e->misses = count;
	e->opt = NULL;
	if (!io_reserve(0, count))
		return;
	vgl_fast_memcpy(s_copy, src, count * 2);
	uint32_t lo = 0xffff, hi = 0;
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t v = s_copy[i];
		lo = v < lo ? v : lo;
		hi = v > hi ? v : hi;
	}
	const uint32_t span = hi - lo + 1;
	if (!io_reserve(span, count))
		return;
	e->orig_misses = e->misses = io_misses(s_copy, count, lo, span, fifo);
#ifdef INDEX_OPT_DUMP
	{ // diag: every range once, to host0 (count, then the indices)
		static SceUID fd = -1;
		if (fd < 0)
			fd = sceIoOpen(INDEX_OPT_DUMP, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0666);
		if (fd >= 0) {
			sceIoWrite(fd, &count, 4);
			sceIoWrite(fd, s_copy, count * 2);
		}
	}
#endif
	if (io_bytes + count * 2u > IO_BYTES_CAP)
		return;
	uint16_t *out = gpu_alloc_mapped_for_gpu(count * 2);
	if (!out)
		return;
	// Built in RAM, then copied in one pass to the GPU-mapped store
	uint16_t *built = malloc(count * 2);
	if (!built) {
		vgl_free(out);
		return;
	}
	const uint32_t misses = io_reorder(s_copy, count, built, lo, span, fifo);
	if (misses >= e->orig_misses) {
		free(built);
		vgl_free(out);
		io_stats[5]++;
		return;
	}
	vgl_fast_memcpy(out, built, count * 2);
	free(built);
	e->misses = misses;
	e->opt = out;
	io_bytes += count * 2u;
	io_stats[4]++;
}

const uint16_t *vgl_index_opt_lookup(const vbo *buf, uint32_t offset, int32_t count, const uint16_t *src) {
#ifdef INDEX_OPT_DUMP_ON
	{ // diag: one frame in 300, every static triangle draw
		extern uint32_t vgl_index_opt_program_info(uint32_t *vhash, uint32_t *fhash);
		if (vgl_framecount % 300 == 150) {
			uint32_t vh, fh, pr = vgl_index_opt_program_info(&vh, &fh);
			const uint32_t rk = (((uint32_t)buf >> 3) * 2654435761u) ^ (offset * 0x9e3779b1u) ^ ((uint32_t)count * 0x85ebca6bu);
			sceClibPrintf("[index-draw] f %u range %08x count %d prog %u vhash %08x fhash %08x blend %u zfunc %x zw %u cull %u front %x stencil %u tex %u %u %u\n", vgl_framecount, rk, count, pr, vh, fh, blend_state, depth_func, depth_mask_state, cull_face_state, gl_front_face, stencil_test_state, texture_units[0].tex_id[0], texture_units[1].tex_id[0], texture_units[2].tex_id[0]);
		}
	}
#endif
	if (count < IO_MIN_INDICES || buf->usage != GL_STATIC_DRAW || buf->mapped || blend_state || stencil_test_state || !depth_test_state ||
	    (depth_func != SCE_GXM_DEPTH_FUNC_LESS && depth_func != SCE_GXM_DEPTH_FUNC_LESS_EQUAL))
		return src;
	// knobs2 0x10000000: original order; 0x20000000: build for the FIFO model
	const uint32_t knobs = &ml_experiment_knobs2 ? ml_experiment_knobs2 : 0;
	const uint32_t model = (knobs >> 28) & 3u;
	if (model != io_model) {
		io_flush();
		io_model = model;
	}
	if (vgl_framecount - io_stats_frame >= 300) {
		if (io_stats[0])
			sceClibPrintf("[index-opt] model %u frames %u draws/f %u idx/f %u misses/f orig %u now %u | built %u kept %u bytes %u | idx/f no-colour %u no-zwrite %u repeat %u\n", model, vgl_framecount - io_stats_frame,
				io_stats[0] / 300, io_stats[1] / 300, io_stats[2] / 300, io_stats[3] / 300, io_stats[4], io_stats[5], io_bytes, io_stats[6] / 300, io_stats[7] / 300, io_stats[8] / 300);
		sceClibMemset(io_stats, 0, sizeof(io_stats));
		io_stats_frame = vgl_framecount;
	}
	const uint32_t h = (io_hash_buf(buf) ^ (offset * 0x9e3779b1u) ^ ((uint32_t)count * 0x85ebca6bu));
	io_entry *set = io_table[(h >> 7) & (IO_SETS - 1)];
	io_entry *e = NULL;
	for (uint32_t w = 0; w < IO_WAYS; w++) {
		io_entry *c = &set[w];
		if (c->buf == buf && c->store == buf->ptr && c->offset == offset && c->count == (uint32_t)count) {
			e = c;
			break;
		}
	}
	if (!e) {
		if (model & 1u)
			return src;
		if (io_budget_frame != vgl_framecount) {
			io_budget_frame = vgl_framecount;
			io_budget_used = 0;
		}
		if (io_budget_used && io_budget_used + (uint32_t)count > IO_FRAME_BUDGET)
			return src;
		io_budget_used += (uint32_t)count;
		e = &set[0];
		for (uint32_t w = 0; w < IO_WAYS; w++) {
			if (!set[w].buf) {
				e = &set[w];
				break;
			}
			if (set[w].last_frame < e->last_frame)
				e = &set[w];
		}
		if (e->buf)
			io_drop(e);
		e->buf = buf;
		e->store = buf->ptr;
		e->offset = offset;
		e->count = (uint32_t)count;
		const uint32_t hb = io_hash_buf(buf) >> 20;
		io_bloom[hb >> 5] |= 1u << (hb & 31);
		io_build(e, src, (uint32_t)count, (model >> 1) & 1u);
	}
	if (e->last_frame == vgl_framecount)
		io_stats[8] += (uint32_t)count;
	if (blend_color_mask == SCE_GXM_COLOR_MASK_NONE)
		io_stats[6] += (uint32_t)count;
	if (!depth_mask_state)
		io_stats[7] += (uint32_t)count;
	e->last_frame = vgl_framecount;
	io_stats[0]++;
	io_stats[1] += (uint32_t)count;
	io_stats[2] += e->orig_misses;
	io_stats[3] += (model & 1u) ? e->orig_misses : e->misses;
	return (model & 1u) || !e->opt ? src : e->opt;
}

#endif
