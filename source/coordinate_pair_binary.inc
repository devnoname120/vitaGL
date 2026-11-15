/* Exact paired GXP substitution at link time (HAVE_BIOSHOCK_RENDERER; included
 * by custom_shaders.c after release_shader). The loader recognizes a vertex and
 * fragment program pair and supplies the pair to run instead. The attached
 * shader objects stay owned by their callers and keep answering every query
 * (program->public_vshader/public_fshader); the substituted executables are
 * private shader objects of one linked program, outside the shader table. */
static shader *coordinate_public_vertex(program *p) {
	return p->public_vshader ? p->public_vshader : p->vshader;
}

static shader *coordinate_public_fragment(program *p) {
	return p->public_fshader ? p->public_fshader : p->fshader;
}

static shader *coordinate_clone(shader *original, const void *binary, unsigned bytes) {
	// Zeroed: slot 0 is not this object's, release_shader checks the table entry
	shader *s = (shader *)vglCalloc(1, sizeof(shader));
	if (!s)
		return NULL;
	unsigned storage = (bytes + 3u) & ~3u;
	void *copy = vglMalloc(storage);
	if (!copy) {
		vgl_free(s);
		return NULL;
	}
	memcpy(copy, binary, bytes);
	memset((unsigned char *)copy + bytes, 0, storage - bytes);
	if (sceGxmProgramGetDefaultUniformBufferSize(copy) != original->unif_buf_size || sceGxmShaderPatcherRegisterProgram(gxm_shader_patcher, copy, &s->id) < 0) {
		vgl_free(copy);
		vgl_free(s);
		return NULL;
	}
	s->type = original->type;
	s->dirty = GL_TRUE;
	s->is_glsl = GL_FALSE;
	s->ref_counter = 1;
	s->prog = copy;
	s->size = storage;
	s->unif_buf_size = original->unif_buf_size;
	// The same matrix uniforms, at the same parameter indices
	unsigned count = 0;
	for (matrix_uniform *m = original->mat; m; m = (matrix_uniform *)m->chain) {
		if (++count > 32 || !m->ptr)
			goto failure;
		const char *name = sceGxmProgramParameterGetName(m->ptr);
		const SceGxmProgramParameter *ptr = sceGxmProgramFindParameterByName(s->prog, name);
		if (!ptr || sceGxmProgramParameterGetIndex(s->prog, ptr) != sceGxmProgramParameterGetIndex(original->prog, m->ptr))
			goto failure;
		matrix_uniform *n = (matrix_uniform *)vglMalloc(sizeof(matrix_uniform));
		if (!n)
			goto failure;
		n->ptr = ptr;
		n->chain = s->mat;
		s->mat = n;
	}
	return s;
failure:
	release_shader(s);
	return NULL;
}

// Called by glLinkProgram once both stages are compiled. Returns the loader's family number, 0 without a substitution.
static unsigned coordinate_pair_prepare(program *p) {
	shader *v = p->vshader, *f = p->fshader;
	if (p->public_vshader || p->status != PROG_UNLINKED || !v || !f || !v->prog || !f->prog || v->type != GL_VERTEX_SHADER || f->type != GL_FRAGMENT_SHADER || v->unif_blk || f->unif_blk)
		return 0;
#ifdef HAVE_GLSL_TEXTURE_SIZE
	if (v->sized_samplers_num || f->sized_samplers_num)
		return 0;
#endif
	unsigned vbytes = sceGxmProgramGetSize(v->prog), fbytes = sceGxmProgramGetSize(f->prog);
	if (vbytes < 148 || fbytes < 148 || vbytes > 65536 || fbytes > 65536 || v->size < vbytes || v->size > ((vbytes + 3u) & ~3u) || f->size < fbytes || f->size > ((fbytes + 3u) & ~3u))
		return 0;
	const void *vb = NULL, *fb = NULL;
	unsigned vn = 0, fn = 0;
	unsigned family = ml_water_coordinate_pair(v->prog, vbytes, f->prog, fbytes, &vb, &vn, &fb, &fn);
	if (!family)
		return 0;
	shader *cv = coordinate_clone(v, vb, vn);
	if (!cv)
		return 0;
	shader *cf = coordinate_clone(f, fb, fn);
	if (!cf) {
		release_shader(cv);
		return 0;
	}
	p->public_vshader = v;
	p->public_fshader = f;
	p->vshader = cv;
	p->fshader = cf;
	return family;
}

/* Releases the private pair and puts the attached shaders back. Callers: a
 * link that fails after the substitution (nothing was drawn with the pair) and
 * glDeleteProgram, after its sceGxmFinish and before it drops the program's
 * references to the attached shaders. */
static void coordinate_pair_release(program *p) {
	if (!p->public_vshader)
		return;
	shader *v = p->vshader, *f = p->fshader;
	p->vshader = p->public_vshader;
	p->fshader = p->public_fshader;
	p->public_vshader = p->public_fshader = NULL;
	release_shader(v);
	release_shader(f);
}
