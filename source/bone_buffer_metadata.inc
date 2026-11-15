/* HAVE_BIOSHOCK_RENDERER; included by custom_shaders.c before glLinkProgram.
 * Native Cg BUFFER semantics can describe an unnamed GXP uniform buffer while
 * the compiler reflection has only an ARRAY, not an UNIFORMBLOCK. Reconstruct
 * metadata only for the exact BioShock wide-palette interface. Doing this at
 * link also covers the shader cache, which does not store blocks.
 * Returns GL_FALSE when the vertex program has a BoneMatrices buffer of any
 * other shape: glLinkProgram then leaves the program unlinked. */
static GLboolean ml_bioshock_bone_buffer_metadata(shader *s) {
    if (!s || !s->prog || s->type != GL_VERTEX_SHADER) return GL_TRUE;
    const SceGxmProgramParameter *bones = sceGxmProgramFindParameterByName(s->prog, "BoneMatrices");
    if (!bones || sceGxmProgramParameterGetCategory(bones) != SCE_GXM_PARAMETER_CATEGORY_UNIFORM ||
        sceGxmProgramParameterGetContainerIndex(bones) == UBOS_NUM) return GL_TRUE;
    if (sceGxmProgramParameterGetContainerIndex(bones) != 0 ||
        sceGxmProgramParameterGetType(bones) != SCE_GXM_PARAMETER_TYPE_F32 ||
        sceGxmProgramParameterGetComponentCount(bones) != 4 ||
        sceGxmProgramParameterGetArraySize(bones) != 768 ||
        sceGxmProgramParameterGetResourceIndex(bones) != 0) return GL_FALSE;
    unsigned buffers = 0, members = 0;
    unsigned count = sceGxmProgramGetParameterCount(s->prog);
    for (unsigned i = 0; i < count; ++i) {
        const SceGxmProgramParameter *p = sceGxmProgramGetParameter(s->prog, i);
        if (sceGxmProgramParameterGetCategory(p) == SCE_GXM_PARAMETER_CATEGORY_UNIFORM_BUFFER &&
            sceGxmProgramParameterGetResourceIndex(p) == 0) {
            if (sceGxmProgramParameterGetArraySize(p) != 12288) return GL_FALSE;
            ++buffers;
        }
        if (sceGxmProgramParameterGetCategory(p) == SCE_GXM_PARAMETER_CATEGORY_UNIFORM &&
            sceGxmProgramParameterGetContainerIndex(p) == 0) ++members;
    }
    if (buffers != 1 || members != 1) return GL_FALSE;
    block_uniform *b = get_block_details(s->unif_blk, 0);
    if (b) return GL_TRUE;
    b = (block_uniform *)vglMalloc(sizeof(*b));
    if (!b) return GL_FALSE;
    memcpy(b->name, "pmBones", sizeof("pmBones"));
    b->idx = 0;
    b->chain = s->unif_blk;
    s->unif_blk = b;
    return GL_TRUE;
}
