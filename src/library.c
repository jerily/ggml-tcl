/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */
#include "library.h"
#include <stdio.h>
#include <string.h>
#include <ggml.h>
#include "common.h"
#include "tensor.h"

#define XSTR(s) STR(s)
#define STR(s) #s

static int ml_ModuleInitialized;

static ml_context_t *ml_CreateContext(size_t mem_size) {

    ml_context_t *ctx = (ml_context_t *) Tcl_Alloc(sizeof(ml_context_t));
    ctx->mem_size = mem_size;
    ctx->mem_buffer = Tcl_Alloc(mem_size);

    struct ggml_init_params params = {
            .mem_size   = mem_size,                      // bytes
            .mem_buffer = ctx->mem_buffer,               // if NULL, memory will be allocated internally
            .no_alloc   = 0,                             // don't allocate memory for the tensor data
    };

    // memory allocation happens here
    struct ggml_context *ggml_ctx = ggml_init(params);
    ctx->ggml_ctx = ggml_ctx;
    ctx->gf = NULL;
    ctx->gb = NULL;
    ctx->first_tensor_ptr = NULL;
    ctx->last_tensor_ptr = NULL;

    CMD_CONTEXT_NAME(ctx->handle, ctx);
    ml_RegisterContext(ctx->handle, ctx);

    return ctx;
}

static int ml_CreateContextCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "CreateContextCmd\n"));
    CheckArgs(2, 2, 1, "mem_size");

    size_t mem_size;
    if (Tcl_GetLongFromObj(interp, objv[1], &mem_size) != TCL_OK || mem_size <= 0) {
        SetResult("mem_size is not an integer > 0");
        return TCL_ERROR;
    }

    ml_context_t *ctx = ml_CreateContext(mem_size);

    SetResult(ctx->handle);
    return TCL_OK;

}

static int ml_DestroyContext(Tcl_Interp *interp, ml_context_t *ctx) {
    if (!ml_UnregisterContext(ctx->handle)) {
        SetResult("unregister server name failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = ctx->first_tensor_ptr;
    while(tensor_ptr != NULL) {
        ml_tensor_t *next_tensor_ptr = tensor_ptr->next;
        if (!ml_UnregisterTensor(tensor_ptr->handle)) {
            SetResult("unregister tensor name failed");
            return TCL_ERROR;
        }
        Tcl_Free((char *) tensor_ptr);
        tensor_ptr = next_tensor_ptr;
    }
    ctx->first_tensor_ptr = NULL;
    ctx->last_tensor_ptr = NULL;

    if (ctx->gf) {
        if (!ml_UnregisterCGraph(ctx->gf->handle)) {
            SetResult("unregister cgraph name failed");
            return TCL_ERROR;
        }
        Tcl_Free((char *) ctx->gf);
        ctx->gf = NULL;
    }
    if (ctx->gb) {
        if (!ml_UnregisterCGraph(ctx->gb->handle)) {
            SetResult("unregister cgraph name failed");
            return TCL_ERROR;
        }
        Tcl_Free((char *) ctx->gb);
        ctx->gb = NULL;
    }
    Tcl_Free(ctx->mem_buffer);
    ggml_free(ctx->ggml_ctx);
    Tcl_Free((char *) ctx);

    return TCL_OK;
}
static int ml_DestroyContextCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DestroyContextCmd\n"));
    CheckArgs(2, 2, 1, "context_handle");
    const char *handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    return ml_DestroyContext(interp, ctx);
}

static int ml_LoadContextFromFileCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "LoadContextFromFileCmd\n"));
    CheckArgs(2, 2, 1, "filename");

    ml_context_t *ctx = (ml_context_t *) Tcl_Alloc(sizeof(ml_context_t));

    struct gguf_init_params params = {
            .no_alloc = 0,
            .ctx = &ctx->ggml_ctx,
    };

    const char *filename = Tcl_GetString(objv[1]);
    fprintf(stderr, "filename: %s\n", filename);
    struct gguf_context *gguf_ctx = gguf_init_from_file(filename, params);
    if (!gguf_ctx) {
        Tcl_Free((char *) ctx);
        SetResult("failed to load context from file");
        return TCL_ERROR;
    }
//    gguf_free(gguf_ctx);

    ctx->mem_size = ggml_get_mem_size(ctx->ggml_ctx);
    ctx->mem_buffer = ggml_get_mem_buffer(ctx->ggml_ctx);
    ctx->gf = NULL;
    ctx->gb = NULL;
    ctx->first_tensor_ptr = NULL;
    ctx->last_tensor_ptr = NULL;

    CMD_CONTEXT_NAME(ctx->handle, ctx);
    ml_RegisterContext(ctx->handle, ctx);

    SetResult(ctx->handle);
    return TCL_OK;
}


static int ml_BuildForwardCtxCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "BuildForwardCtxCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");

    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }

    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }

    ml_cgraph_t *cgraph_ptr = (ml_cgraph_t *) Tcl_Alloc(sizeof(ml_cgraph_t));
    cgraph_ptr->ggml_cgraph = ggml_build_forward_ctx(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
    cgraph_ptr->ctx = ctx;
    CMD_CGRAPH_NAME(cgraph_ptr->handle, cgraph_ptr);
    ml_RegisterCGraph(cgraph_ptr->handle, cgraph_ptr);
    ctx->gf = cgraph_ptr;

    SetResult(cgraph_ptr->handle);
    return TCL_OK;

}

static int ml_GraphComputeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GraphComputeCmd\n"));
    CheckArgs(3, 3, 1, "cgraph_handle nthreads");

    const char *cgraph_handle = Tcl_GetString(objv[1]);
    ml_cgraph_t *cgraph_ptr = ml_GetInternalFromCGraph(cgraph_handle);
    if (!cgraph_ptr) {
        SetResult("cgraph handle not found");
        return TCL_ERROR;
    }

    int nthreads;
    if (Tcl_GetIntFromObj(interp, objv[2], &nthreads) != TCL_OK || nthreads <= 0) {
        SetResult("nthreads is not a positive integer");
        return TCL_ERROR;
    }

    ggml_graph_compute_with_ctx(cgraph_ptr->ctx->ggml_ctx, cgraph_ptr->ggml_cgraph, nthreads);
    return TCL_OK;
}

static int ml_BuildBackwardCtxCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "BuildBackwardCtxCmd\n"));
    CheckArgs(4, 4, 1, "context_handle forward_cgraph_handle keep_gradient_graph");

    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }

    const char *forward_cgraph_handle = Tcl_GetString(objv[2]);
    ml_cgraph_t *forward_cgraph_ptr = ml_GetInternalFromCGraph(forward_cgraph_handle);
    if (!forward_cgraph_ptr) {
        SetResult("forward_cgraph_handle not found");
        return TCL_ERROR;
    }

    int keep_gradient_graph;
    if (Tcl_GetBooleanFromObj(interp, objv[3], &keep_gradient_graph) != TCL_OK) {
        SetResult("keep_gradient_graph is not a boolean");
        return TCL_ERROR;
    }

    ml_cgraph_t *backward_cgraph_ptr = (ml_cgraph_t *) Tcl_Alloc(sizeof(ml_cgraph_t));
    backward_cgraph_ptr->ggml_cgraph = ggml_new_graph(ctx->ggml_ctx);
    backward_cgraph_ptr->ctx = ctx;
    CMD_CGRAPH_NAME(backward_cgraph_ptr->handle, backward_cgraph_ptr);
    ml_RegisterCGraph(backward_cgraph_ptr->handle, backward_cgraph_ptr);
    ctx->gb = backward_cgraph_ptr;

    *backward_cgraph_ptr->ggml_cgraph = *forward_cgraph_ptr->ggml_cgraph;
    ggml_build_backward_expand(
            ctx->ggml_ctx,
            forward_cgraph_ptr->ggml_cgraph,
            backward_cgraph_ptr->ggml_cgraph,
            keep_gradient_graph);

    SetResult(backward_cgraph_ptr->handle);
    return TCL_OK;
}

static int ml_GraphResetCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GraphResetCmd\n"));
    CheckArgs(2, 2, 1, "cgraph_handle");

    const char *cgraph_handle = Tcl_GetString(objv[1]);
    ml_cgraph_t *cgraph_ptr = ml_GetInternalFromCGraph(cgraph_handle);
    if (!cgraph_ptr) {
        SetResult("cgraph handle not found");
        return TCL_ERROR;
    }

    ggml_graph_reset(cgraph_ptr->ggml_cgraph);
    return TCL_OK;
}

static int ml_GraphDumpDotCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GraphDumpDotCmd\n"));
    CheckArgs(4, 4, 1, "gb_handle fg_handle filename");

    const char *gb_handle = Tcl_GetString(objv[1]);
    ml_cgraph_t *gb_ptr = ml_GetInternalFromCGraph(gb_handle);
    if (!gb_ptr) {
        SetResult("cgraph handle not found");
        return TCL_ERROR;
    }
    struct ggml_cgraph *gb = gb_ptr->ggml_cgraph;
    struct ggml_cgraph *gf = NULL;

    int fg_handle_len;
    const char *fg_handle = Tcl_GetStringFromObj(objv[2], &fg_handle_len);
    if (fg_handle_len > 0) {
        ml_cgraph_t *gf_ptr = ml_GetInternalFromCGraph(fg_handle);
        if (!gf_ptr) {
            SetResult("cgraph handle not found");
            return TCL_ERROR;
        }
        gf = gf_ptr->ggml_cgraph;
    }

    const char *filename = Tcl_GetString(objv[3]);
    ggml_graph_dump_dot(gb, gf, filename);
    return TCL_OK;
}


static void ml_ExitHandler(ClientData unused) {
    ml_DeleteContextHT();
    ml_DeleteCGraphHT();
    ml_DeleteTensorHT();
}


void ml_InitModule() {
    if (!ml_ModuleInitialized) {

        ml_InitContextHT();
        ml_InitCGraphHT();
        ml_InitTensorHT();

        ml_ModuleInitialized = 1;
        DBG(fprintf(stderr, "ggml-tcl module initialized\n"));
    }
}

int Ggml_Init(Tcl_Interp *interp) {
    if (Tcl_InitStubs(interp, "8.6", 0) == NULL) {
        return TCL_ERROR;
    }

    ml_InitModule();

    Tcl_CreateNamespace(interp, "::ggml", NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::create_context", ml_CreateContextCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::destroy_context", ml_DestroyContextCmd, NULL, NULL);
//    Tcl_CreateObjCommand(interp, "::ggml::write_context_to_file", ml_WriteContextToFileCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::load_context_from_file", ml_LoadContextFromFileCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::build_forward_ctx", ml_BuildForwardCtxCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::build_backward_ctx", ml_BuildBackwardCtxCmd, NULL, NULL);

    Tcl_CreateObjCommand(interp, "::ggml::graph_compute", ml_GraphComputeCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::graph_reset", ml_GraphResetCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::graph_dump_dot", ml_GraphDumpDotCmd, NULL, NULL);

    Tcl_CreateObjCommand(interp, "::ggml::set_param", ml_SetParamCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_grad", ml_GetGradCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_f32", ml_SetF32Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_f32_1d", ml_SetF321DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_f32_1d", ml_GetF321DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::nelements", ml_NumElementsCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor", ml_NewTensorCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_1d", ml_NewTensor1DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_2d", ml_NewTensor2DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_3d", ml_NewTensor3DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_4d", ml_NewTensor4DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::add", ml_AddCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::mul", ml_MulCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sum", ml_SumCmd, NULL, NULL);

    return Tcl_PkgProvide(interp, "ggml", XSTR(PROJECT_VERSION));
}

#ifdef USE_NAVISERVER
int Ns_ModuleInit(const char *server, const char *module) {
    Ns_TclRegisterTrace(server, (Ns_TclTraceProc *) Ggml_Init, server, NS_TCL_TRACE_CREATE);
    return NS_OK;
}
#endif
