#ifndef VGL_IOS_DEPTH_TEXTURE_H
#define VGL_IOS_DEPTH_TEXTURE_H

/* Internal interface; include after shared.h. This is the observed BioShock
 * allocation-only, level-zero, single-sample depth-texture subset. */
GLboolean vgl_is_depth_texture(const texture *tex);
GLenum vgl_depth_texture_storage(texture *tex, GLenum target, GLint level,
    GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type,
    const void *data);
GLenum vgl_depth_framebuffer_attach(framebuffer *fb, texture *tex, GLint level);
GLenum vgl_depth_framebuffer_color(framebuffer *fb, texture *tex);
void vgl_depth_framebuffer_detach(framebuffer *fb);
void vgl_depth_texture_delete_bindings(texture *tex, GLuint name);
/* Called by scene_reset before it resyncs the color attachment and begins the
 * scene. Drops a depth texture the color attachment no longer matches. */
void vgl_depth_framebuffer_used(framebuffer *fb);
GLboolean vgl_depth_texture_has_attachments(const texture *tex);
/* True while fb's depth surface is a depth texture's store, which the texture
 * owns: vitaGL must not free it as the framebuffer's own surface. */
GLboolean vgl_depth_framebuffer_has_texture(const framebuffer *fb);
/* Ends and waits for an existing scene without starting one on old storage. */
void vgl_depth_finish_framebuffer(framebuffer *fb);

#endif
