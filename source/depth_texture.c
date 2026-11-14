#ifdef HAVE_DEPTH_TEXTURES
#include "shared.h"
#include "utils/gxm_utils.h"
#include "utils/gpu_utils.h"
#include "depth_texture.h"

/* Keep the framebuffer/texture ABI unchanged. One entry per live depth
 * attachment; color attachments retain vitaGL's existing ownership scheme.
 *
 * vitaGL frees whatever fb->depthbuffer_ptr names as the framebuffer's own
 * depth-stencil surface. While an entry exists that surface is laid over the
 * texture's store (depthbuffer_state is DEPTHBUFFER_READY so scene_reset does
 * not replace it) and the store belongs to the texture: every vitaGL path that
 * discards the framebuffer's surface must drop the entry first or skip it. */
typedef struct {
    framebuffer *fb;
    texture *tex;
} depth_attachment;
static depth_attachment depth_attachments[BUFFERS_NUM];

static depth_attachment *find_attachment(framebuffer *fb) {
    for (unsigned i=0;i<BUFFERS_NUM;i++)
        if (depth_attachments[i].fb==fb) return depth_attachments+i;
    return NULL;
}

GLboolean vgl_is_depth_texture(const texture *tex) {
    return tex && tex->status==TEX_VALID &&
        tex->format==SCE_GXM_TEXTURE_FORMAT_F32M_1RRR;
}

GLboolean vgl_depth_texture_has_attachments(const texture *tex) {
    for (unsigned i=0;i<BUFFERS_NUM;i++)
        if (depth_attachments[i].fb && depth_attachments[i].tex==tex) return GL_TRUE;
    return GL_FALSE;
}

GLboolean vgl_depth_framebuffer_has_texture(const framebuffer *fb) {
    if (!fb) return GL_FALSE;
    for (unsigned i=0;i<BUFFERS_NUM;i++)
        if (depth_attachments[i].fb==fb) return GL_TRUE;
    return GL_FALSE;
}

static void initialize_surface(SceGxmDepthStencilSurface *surface,
                               void *data, unsigned width) {
    vglDepthStencilSurfaceInit(surface,SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M,
        SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR,width,data,NULL);
    /* Unlike a transient hidden depth buffer, this storage is sampled by
     * later passes and must survive scene transitions. */
    sceGxmDepthStencilSurfaceSetForceLoadMode(surface,SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
    sceGxmDepthStencilSurfaceSetForceStoreMode(surface,SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED);
}

GLenum vgl_depth_texture_storage(texture *tex, GLenum target, GLint level,
    GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type,
    const void *data) {
    if (!tex || target!=GL_TEXTURE_2D || format!=GL_DEPTH_COMPONENT ||
        type!=GL_UNSIGNED_INT) return GL_INVALID_ENUM;
    if (msaa_mode!=SCE_GXM_MULTISAMPLE_NONE || data || tex->use_mips) return GL_INVALID_OPERATION;
    /* All observed surfaces are 512/1024-square. Reject unsupported row
     * layouts instead of creating a texture/depth-stride mismatch. */
    if (level || border || width<=0 || height<=0 || width>GXM_TEX_MAX_SIZE ||
        height>GXM_TEX_MAX_SIZE || (width&(SCE_GXM_TILE_SIZEX-1))) return GL_INVALID_VALUE;
    for (unsigned i=0;i<BUFFERS_NUM;i++) {
        depth_attachment *entry=depth_attachments+i;
        if (entry->fb && entry->tex==tex &&
            (entry->fb->width!=width || entry->fb->height!=height)) return GL_INVALID_VALUE;
    }

    size_t size=(size_t)width*VGL_ALIGN((unsigned)height,SCE_GXM_TILE_SIZEY)*4u;
    void *replacement=gpu_alloc_mapped_aligned_for_gpu(SCE_GXM_DEPTHSTENCIL_SURFACE_ALIGNMENT,size);
    if (!replacement) return GL_OUT_OF_MEMORY;
    SceGxmTexture image;
    if (sceGxmTextureInitLinear(&image,replacement,SCE_GXM_TEXTURE_FORMAT_F32M_1RRR,width,height,1)<0) {
        vgl_free(replacement);
        return GL_INVALID_VALUE;
    }
    /* Follow vitaGL's STORE_DEPTH_STENCIL initialization for the mask bit.
     * Initial depth values remain undefined, as in allocation-only GL storage. */
    vgl_memset(replacement,0x80,size);
    vglSetTexMinFilter(&image,tex->min_filter);
    vglSetTexMagFilter(&image,tex->mag_filter);
    vglSetTexMipFilter(&image,tex->mip_filter);
    vglSetTexUMode(&image,tex->u_mode);
    vglSetTexVMode(&image,tex->v_mode);
    vglSetTexLodBias(&image,tex->lod_bias);
    vglSetTexMipmapCount(&image,1);

    /* Commit only after successful allocation/descriptor creation. In-flight
     * readers retain the old store through the normal deferred-free path. */
    for (unsigned i=0;i<BUFFERS_NUM;i++)
        if (depth_attachments[i].fb && depth_attachments[i].tex==tex)
            vgl_depth_finish_framebuffer(depth_attachments[i].fb);
    gpu_free_texture_data(tex);
    tex->gxm_tex=image;
    tex->format=SCE_GXM_TEXTURE_FORMAT_F32M_1RRR;
    tex->data=replacement;
    tex->palette_data=NULL;
    tex->write_cb=(void *)GL_TRUE;
    tex->mip_count=1;
    tex->status=TEX_VALID;
    tex->overridden=GL_FALSE;
    tex->faces_counter=0;
#ifndef TEXTURES_SPEEDHACK
    tex->last_frame=OBJ_NOT_USED;
#endif
    for (unsigned i=0;i<BUFFERS_NUM;i++) {
        depth_attachment *entry=depth_attachments+i;
        if (entry->fb && entry->tex==tex) {
            initialize_surface(&entry->fb->depthbuffer,replacement,width);
            entry->fb->depthbuffer_ptr=&entry->fb->depthbuffer;
            entry->fb->depthbuffer_state=DEPTHBUFFER_READY;
            if (entry->fb==active_write_fb) dirty_framebuffer=GL_TRUE;
        }
    }
    return GL_NO_ERROR;
}

/* Returns the store to the texture alone: the framebuffer is left without a
 * depth surface (scene_reset gives it one of its own) and the texture is freed
 * only here, once deleted and unreferenced. */
static void drop_attachment(framebuffer *fb, depth_attachment *entry) {
    texture *tex=entry->tex;
    entry->fb=NULL;
    entry->tex=NULL;
    fb->depthbuffer_ptr=NULL;
    fb->depthbuffer_state=DEPTHBUFFER_MISSING;
    tex->ref_counter--;
    if (!tex->ref_counter && tex->dirty) gpu_free_texture(tex);
}

void vgl_depth_framebuffer_detach(framebuffer *fb) {
    if (!fb) return;
    depth_attachment *entry=find_attachment(fb);
    if (!entry) return;
    vgl_depth_finish_framebuffer(fb);
    drop_attachment(fb,entry);
    if (fb==active_write_fb) dirty_framebuffer=GL_TRUE;
}

GLenum vgl_depth_framebuffer_attach(framebuffer *fb, texture *tex, GLint level) {
    if (!fb || !fb->active) return GL_INVALID_OPERATION;
    if (level) return GL_INVALID_VALUE;
    depth_attachment *entry=find_attachment(fb);
    if (!tex) {
        vgl_depth_framebuffer_detach(fb);
        /* Also detach a renderbuffer: what is left in depthbuffer_ptr is the
         * framebuffer's own surface, and its stencil request goes with it. */
        if (fb->depthbuffer_ptr) {
            vgl_depth_finish_framebuffer(fb);
#ifndef DEPTH_STENCIL_HACK
            mark_depth_stencil_as_dirty(fb->depthbuffer_ptr);
#endif
            fb->depthbuffer_ptr=NULL;
            if (fb==active_write_fb) dirty_framebuffer=GL_TRUE;
        }
        fb->depthbuffer_state=DEPTHBUFFER_MISSING;
        return GL_NO_ERROR;
    }
    if (VGL_FRAMEBUFFER_MSAA(fb)!=SCE_GXM_MULTISAMPLE_NONE || !vgl_is_depth_texture(tex)) return GL_INVALID_OPERATION;
    uint32_t width,height;
    vglGetTexSizes(&tex->gxm_tex,&width,&height);
    /* The observed path attaches color first, then matching depth. Depth-only
     * FBOs and incomplete layouts are not claimed by this subset. */
    if (!fb->tex || !fb->data || fb->tex->status!=TEX_VALID ||
        fb->width!=(int)width || fb->height!=(int)height) return GL_INVALID_VALUE;
    if (entry && entry->tex==tex) return GL_NO_ERROR;
    if (tex->ref_counter==UINT8_MAX) return GL_OUT_OF_MEMORY;
    if (!entry) {
        for (unsigned i=0;i<BUFFERS_NUM;i++)
            if (!depth_attachments[i].fb) {entry=depth_attachments+i;break;}
        if (!entry) return GL_OUT_OF_MEMORY;
    }
    vgl_depth_finish_framebuffer(fb);
    /* A surface found without an entry is the framebuffer's own and is released
     * here; with one it is the previous texture's store, which only loses a
     * reference below. */
#ifndef DEPTH_STENCIL_HACK
    if (fb->depthbuffer_ptr && !entry->fb) mark_depth_stencil_as_dirty(fb->depthbuffer_ptr);
#endif
    texture *previous=entry->fb?entry->tex:NULL;
    tex->ref_counter++;
    entry->fb=fb;
    entry->tex=tex;
    initialize_surface(&fb->depthbuffer,tex->data,width);
    fb->depthbuffer_ptr=&fb->depthbuffer;
    fb->depthbuffer_state=DEPTHBUFFER_READY;
    if (fb==active_write_fb) dirty_framebuffer=GL_TRUE;
    if (previous) {
        previous->ref_counter--;
        if (!previous->ref_counter && previous->dirty) gpu_free_texture(previous);
    }
    return GL_NO_ERROR;
}

GLenum vgl_depth_framebuffer_color(framebuffer *fb, texture *tex) {
    depth_attachment *entry=fb?find_attachment(fb):NULL;
    if (!entry || !tex) return GL_NO_ERROR;
    uint32_t width,height,depth_width,depth_height;
    vglGetTexSizes(&tex->gxm_tex,&width,&height);
    vglGetTexSizes(&entry->tex->gxm_tex,&depth_width,&depth_height);
    if (width!=depth_width || height!=depth_height) return GL_INVALID_VALUE;
    vgl_depth_finish_framebuffer(fb);
    /* vitaGL's color attach discards the framebuffer's depth surface when the
     * framebuffer size changes. The new color matches the depth texture, so
     * only a size left stale by a color detach can differ: refresh it so that
     * path leaves the texture's store alone. */
    fb->width=(int)width;
    fb->height=(int)height;
    return GL_NO_ERROR;
}

void vgl_depth_texture_delete_bindings(texture *tex, GLuint name) {
    for (unsigned unit=0;unit<COMBINED_TEXTURE_IMAGE_UNITS_NUM;unit++)
        for (unsigned target=0;target<3;target++)
            if (texture_units[unit].tex_id[target]==name) texture_units[unit].tex_id[target]=0;
    if (active_read_fb) {
        depth_attachment *entry=find_attachment(active_read_fb);
        if (entry && entry->tex==tex) vgl_depth_framebuffer_detach(active_read_fb);
    }
    if (active_write_fb) {
        depth_attachment *entry=find_attachment(active_write_fb);
        if (entry && entry->tex==tex) vgl_depth_framebuffer_detach(active_write_fb);
    }
}

void vgl_depth_framebuffer_used(framebuffer *fb) {
    depth_attachment *entry=fb?find_attachment(fb):NULL;
    if (!entry) return;
    /* scene_reset re-attaches a color texture whose storage was replaced and
     * discards the depth surface if its size changed. A depth texture cannot
     * follow that: drop it before vitaGL frees its store. The previous scene
     * has ended already, so this must not end one again. */
    if (fb->tex) {
        uint32_t width,height,depth_width,depth_height;
        vglGetTexSizes(&fb->tex->gxm_tex,&width,&height);
        vglGetTexSizes(&entry->tex->gxm_tex,&depth_width,&depth_height);
        if (width!=depth_width || height!=depth_height) {
            drop_attachment(fb,entry);
            return;
        }
    }
#ifndef TEXTURES_SPEEDHACK
    entry->tex->last_frame=vgl_framecount;
#endif
}
#endif
