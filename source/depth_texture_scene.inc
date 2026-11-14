/* Storage/attachment edits must not begin a fresh scene on the old store.
 * Included after scene_end() so this uses the renderer's actual scene state. */
void vgl_depth_finish_framebuffer(framebuffer *fb) {
    if (fb == in_use_framebuffer && needs_end_scene) {
        scene_end();
        needs_end_scene = GL_FALSE;
        needs_scene_reset = GL_TRUE;
        dirty_framebuffer = GL_TRUE;
        sceGxmFinish(gxm_context);
    }
}
