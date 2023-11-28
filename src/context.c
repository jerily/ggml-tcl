/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */
#include <tcl.h>
#include <ggml.h>
#include "common.h"
#include "context.h"

static ml_context_t *ml_CreateContext(size_t mem_size) {

    ml_context_t *ctx = (ml_context_t *) Tcl_Alloc(sizeof(ml_context_t));
    ctx->mem_buffer = Tcl_Alloc(mem_size);

    struct ggml_init_params params = {
            .mem_size   = mem_size,                      // bytes
            .mem_buffer = ctx->mem_buffer,               // if NULL, memory will be allocated internally
            .no_alloc   = 0,                             // don't allocate memory for the tensor data
    };

    // memory allocation happens here
    struct ggml_context *ggml_ctx = ggml_init(params);
    ctx->ggml_ctx = ggml_ctx;
    ctx->gguf_ctx = NULL;
    ctx->first_graph_ptr = NULL;
    ctx->last_graph_ptr = NULL;
    ctx->first_tensor_ptr = NULL;
    ctx->last_tensor_ptr = NULL;

    CMD_CONTEXT_NAME(ctx->handle, ctx);
    ml_RegisterContext(ctx->handle, ctx);

    return ctx;
}

int ml_CreateContextCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

    ml_cgraph_t *graph_ptr = ctx->first_graph_ptr;
    while (graph_ptr) {
        ml_cgraph_t *next_graph_ptr = graph_ptr->next;
        if (!ml_UnregisterCGraph(graph_ptr->handle)) {
            SetResult("unregister cgraph name failed");
            return TCL_ERROR;
        }
        Tcl_Free((char *) graph_ptr);
        graph_ptr = next_graph_ptr;
    }
    ctx->first_graph_ptr = NULL;
    ctx->last_graph_ptr = NULL;

    if (ctx->mem_buffer != NULL) {
        Tcl_Free(ctx->mem_buffer);
    }
    if (ctx->gguf_ctx != NULL) {
        gguf_free(ctx->gguf_ctx);
    } else {
        ggml_free(ctx->ggml_ctx);
    }
    Tcl_Free((char *) ctx);

    return TCL_OK;
}
int ml_DestroyContextCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_LoadContextFromFileCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

    ctx->gguf_ctx = gguf_ctx;
    ctx->mem_buffer = NULL;
    ctx->first_graph_ptr = NULL;
    ctx->last_graph_ptr = NULL;
    ctx->first_tensor_ptr = NULL;
    ctx->last_tensor_ptr = NULL;

    CMD_CONTEXT_NAME(ctx->handle, ctx);
    ml_RegisterContext(ctx->handle, ctx);

    SetResult(ctx->handle);
    return TCL_OK;
}

int ml_UsedMemCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "UsedMemCmd\n"));
    CheckArgs(2, 2, 1, "context_handle");
    const char *handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }

    size_t used_mem = ggml_used_mem(ctx->ggml_ctx);

    Tcl_SetObjResult(interp, Tcl_NewLongObj(used_mem));
    return TCL_OK;
}

int ml_GetMaxTensorSizeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "UsedMemCmd\n"));
    CheckArgs(2, 2, 1, "context_handle");
    const char *handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }

    size_t max_tensor_size = ggml_get_max_tensor_size(ctx->ggml_ctx);

    Tcl_SetObjResult(interp, Tcl_NewLongObj(max_tensor_size));
    return TCL_OK;
}

int ml_GetMemSizeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "UsedMemCmd\n"));
    CheckArgs(2, 2, 1, "context_handle");
    const char *handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }

    size_t mem_size = ggml_get_mem_size(ctx->ggml_ctx);

    Tcl_SetObjResult(interp, Tcl_NewLongObj(mem_size));
    return TCL_OK;
}