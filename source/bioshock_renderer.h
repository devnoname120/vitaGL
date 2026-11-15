#ifndef VGL_BIOSHOCK_RENDERER_H
#define VGL_BIOSHOCK_RENDERER_H
#ifdef HAVE_BIOSHOCK_RENDERER

/* Contract between vitaGL and the BioShock loader (HAVE_BIOSHOCK_RENDERER).
 * fresh_upload_core.h is the loader's file, byte for byte: MLFreshStore is
 * passed by value across the two objects. */
#include "fresh_upload_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Provided by the loader. */
extern const volatile unsigned char g_aggregate_variant;
#define g_bioshock_fresh_upload_variant g_aggregate_variant
/* Consumes the loader's grant for the store being replaced; non-zero keeps it. */
int ml_fresh_upload_take(const MLFreshStore *actual, uint32_t size, const void *data, uint32_t usage);
/* Rewrites a shader source in place (never longer) and returns its new length. */
unsigned ml_aggregate_shader_source(char *source, unsigned size, unsigned stage);
/* Returns a family number and the replacement GXP pair for a known vertex and
 * fragment program pair, or 0. */
unsigned ml_water_coordinate_pair(const void *vertex, unsigned vertex_size, const void *fragment, unsigned fragment_size,
	const void **out_vertex, unsigned *out_vertex_size, const void **out_fragment, unsigned *out_fragment_size);

/* Provided by vitaGL. */
MLFreshStore vglFreshUploadRead(void);
int vglFreshUploadSourceCompatible(const void *data, uint32_t size, uint32_t storage);
unsigned vgl_bioshock_bone_diag(uint32_t *out, unsigned capacity);

#ifdef __cplusplus
}
#endif

#endif
#endif
