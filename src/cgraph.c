/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */

#include <tcl.h>
#include <ggml.h>
#include "cgraph.h"

int ml_InsertGraphToList(ml_context_t *ctx, ml_cgraph_t *internal) {
    if (ctx->first_graph_ptr == NULL) {
        ctx->first_graph_ptr = internal;
        ctx->last_graph_ptr = internal;
    } else {
        ctx->last_graph_ptr->next = internal;
        internal->prev = ctx->last_graph_ptr;
        ctx->last_graph_ptr = internal;
    }
    return TCL_OK;
}

int ml_NewGraphCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NewGraphCmd\n"));
    CheckArgs(2, 2, 1, "context_handle");

    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }

    ml_cgraph_t *cgraph_ptr = (ml_cgraph_t *) Tcl_Alloc(sizeof(ml_cgraph_t));
    cgraph_ptr->ggml_cgraph = ggml_new_graph(ctx->ggml_ctx);
    cgraph_ptr->ctx = ctx;
    cgraph_ptr->prev = NULL;
    cgraph_ptr->next = NULL;
    CMD_CGRAPH_NAME(cgraph_ptr->handle, cgraph_ptr);
    ml_RegisterCGraph(cgraph_ptr->handle, cgraph_ptr);
    ml_InsertGraphToList(ctx, cgraph_ptr);

    SetResult(cgraph_ptr->handle);
    return TCL_OK;
}

int ml_NewGraphCustomCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NewGraphCustomCmd\n"));
    CheckArgs(3, 4, 1, "context_handle grads ?size?");

    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }

    int grads;
    if (Tcl_GetBooleanFromObj(interp, objv[2], &grads) != TCL_OK) {
        SetResult("gradients is not a boolean");
        return TCL_ERROR;
    }

    size_t size = GGML_DEFAULT_GRAPH_SIZE;
    if (objc == 4) {
        if (Tcl_GetLongFromObj(interp, objv[3], &size) != TCL_OK || size <= 0) {
            SetResult("size is not a positive long integer");
            return TCL_ERROR;
        }
    }

    ml_cgraph_t *cgraph_ptr = (ml_cgraph_t *) Tcl_Alloc(sizeof(ml_cgraph_t));
    cgraph_ptr->ggml_cgraph = ggml_new_graph_custom(ctx->ggml_ctx, size, grads);
    cgraph_ptr->ctx = ctx;
    cgraph_ptr->prev = NULL;
    cgraph_ptr->next = NULL;
    CMD_CGRAPH_NAME(cgraph_ptr->handle, cgraph_ptr);
    ml_RegisterCGraph(cgraph_ptr->handle, cgraph_ptr);
    ml_InsertGraphToList(ctx, cgraph_ptr);

    SetResult(cgraph_ptr->handle);
    return TCL_OK;
}

int ml_GraphComputeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_GraphResetCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_GraphDumpDotCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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


int ml_BuildForwardExpandCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "BuildForwardExpandCmd\n"));
    CheckArgs(3, 3, 1, "cgraph_handle tensor_handle");

    const char *cgraph_handle = Tcl_GetString(objv[1]);
    ml_cgraph_t *cgraph_ptr = ml_GetInternalFromCGraph(cgraph_handle);
    if (!cgraph_ptr) {
        SetResult("cgraph handle not found");
        return TCL_ERROR;
    }

    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }

    ggml_build_forward_expand(cgraph_ptr->ggml_cgraph, tensor_ptr->ggml_tensor);
    return TCL_OK;

}

int ml_BuildBackwardExpandCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "BuildBackwardExpandCmd\n"));
    CheckArgs(5, 5, 1, "context_handle forward_cgraph_handle backward_cgraph_handle keep_gradient_graph");

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

    const char *backward_cgraph_handle = Tcl_GetString(objv[3]);
    ml_cgraph_t *backward_cgraph_ptr = ml_GetInternalFromCGraph(backward_cgraph_handle);
    if (!backward_cgraph_ptr) {
        SetResult("backward_cgraph_handle not found");
        return TCL_ERROR;
    }

    int keep_gradient_graph;
    if (Tcl_GetBooleanFromObj(interp, objv[4], &keep_gradient_graph) != TCL_OK) {
        SetResult("keep_gradient_graph is not a boolean");
        return TCL_ERROR;
    }

    ggml_build_backward_expand(
            ctx->ggml_ctx,
            forward_cgraph_ptr->ggml_cgraph,
            backward_cgraph_ptr->ggml_cgraph,
            keep_gradient_graph);

    return TCL_OK;
}

int ml_GraphCpyCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GraphCpyCmd\n"));
    CheckArgs(3, 3, 1, "src_graph_handle dst_graph_handle");

    const char *src_graph_handle = Tcl_GetString(objv[1]);
    ml_cgraph_t *src_graph_ptr = ml_GetInternalFromCGraph(src_graph_handle);
    if (!src_graph_ptr) {
        SetResult("src_graph_handle not found");
        return TCL_ERROR;
    }

    const char *dst_graph_handle = Tcl_GetString(objv[2]);
    ml_cgraph_t *dst_graph_ptr = ml_GetInternalFromCGraph(dst_graph_handle);
    if (!dst_graph_ptr) {
        SetResult("dst_graph_handle not found");
        return TCL_ERROR;
    }

    ggml_graph_cpy(src_graph_ptr->ggml_cgraph, dst_graph_ptr->ggml_cgraph);
    return TCL_OK;
}
