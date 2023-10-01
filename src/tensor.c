/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */

#include <tcl.h>
#include <ggml.h>
#include "common.h"
#include "tensor.h"


static const char *types[] = {
        "F32",
        "F16",
        "Q4_0",
        "Q4_1",
        "Q4_2", // support has been removed
        "Q4_3", // support has been removed
        "Q5_0",
        "Q5_1",
        "Q8_0",
        "Q8_1",
        // k-quantizations
        "Q2_K",
        "Q3_K",
        "Q4_K",
        "Q5_K",
        "Q6_K",
        "Q8_K",
        "I8",
        "I16",
        "I32",
        "COUNT",
        NULL
};

enum ggml_type ml_GetType(Tcl_Interp *interp, Tcl_Obj *objPtr) {
    int typeIndex;
    if (TCL_OK == Tcl_GetIndexFromObj(interp, objPtr, types, "ggml_type", 0, &typeIndex)) {
        return (enum ggml_type) typeIndex;
    }
    return GGML_TYPE_F32;
}

/*static*/ int ml_InsertTensorToList(ml_context_t *ctx, ml_tensor_t *internal) {
    if (ctx->first_tensor_ptr == NULL) {
        ctx->first_tensor_ptr = internal;
        ctx->last_tensor_ptr = internal;
    } else {
        ctx->last_tensor_ptr->next = internal;
        internal->prev = ctx->last_tensor_ptr;
        ctx->last_tensor_ptr = internal;
    }
    return TCL_OK;
}

/*static*/ int ml_GetGradCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GetGradCmd\n"));
    CheckArgs(2, 2, 1, "tensor_handle");

    const char *tensor_handle = Tcl_GetString(objv[1]);
    ml_tensor_t *tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *grad = tensor_ptr->ggml_tensor->grad;
    if (!grad) {
        SetResult("tensor has no gradient");
        return TCL_ERROR;
    }

    ml_tensor_t *grad_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    grad_ptr->ggml_tensor = grad;
    grad_ptr->next = NULL;
    grad_ptr->prev = NULL;
    ml_InsertTensorToList(tensor_ptr->ctx, grad_ptr);
    ml_RegisterTensor(grad_ptr->handle, grad_ptr);

    SetResult(grad_ptr->handle);
    return TCL_OK;
}

/*static*/ int ml_SetParamCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SetParamCmd\n"));
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

    ggml_set_param(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
    return TCL_OK;
}

/*static*/ int ml_SetF32Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SetF32Cmd\n"));
    CheckArgs(3, 3, 1, "tensor_handle float_value");

    const char *tensor_handle = Tcl_GetString(objv[1]);
    ml_tensor_t *tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    double value;
    if (Tcl_GetDoubleFromObj(interp, objv[2], &value) != TCL_OK) {
        return TCL_ERROR;
    }

    ggml_set_f32(tensor_ptr->ggml_tensor, value);
    return TCL_OK;
}

/*static*/ int ml_SetF321DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SetF321DCmd\n"));
    CheckArgs(4, 4, 1, "tensor_handle i float_value");

    const char *tensor_handle = Tcl_GetString(objv[1]);
    ml_tensor_t *tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    int i;
    if (Tcl_GetIntFromObj(interp, objv[2], &i) != TCL_OK || i < 0) {
        SetResult("i is not an integer >= 0");
        return TCL_ERROR;
    }
    double value;
    if (Tcl_GetDoubleFromObj(interp, objv[3], &value) != TCL_OK) {
        return TCL_ERROR;
    }

    ggml_set_f32_1d(tensor_ptr->ggml_tensor, i, value);
    return TCL_OK;
}

/*static*/ int ml_GetF321DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GetF321DCmd\n"));
    CheckArgs(3, 3, 1, "tensor_handle i");
    const char *tensor_handle = Tcl_GetString(objv[1]);
    ml_tensor_t *tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    int i;
    if (Tcl_GetIntFromObj(interp, objv[2], &i) != TCL_OK || i < 0) {
        SetResult("i is not an integer >= 0");
        return TCL_ERROR;
    }

    float value = ggml_get_f32_1d(tensor_ptr->ggml_tensor, i);

    Tcl_SetObjResult(interp, Tcl_NewDoubleObj(value));
    return TCL_OK;
}

/*static*/ int ml_NumElementsCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NumElementsCmd\n"));
    CheckArgs(2, 2, 1, "tensor_handle");

    const char *tensor_handle = Tcl_GetString(objv[1]);
    ml_tensor_t *tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }

    long ne = ggml_nelements(tensor_ptr->ggml_tensor);
    Tcl_SetObjResult(interp, Tcl_NewLongObj(ne));
    return TCL_OK;
}

/*static*/ int ml_NewTensorCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NewTensorCmd\n"));
    CheckArgs(5, 5, 1, "context_handle type ndims ne_list");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    int ndims;
    if (Tcl_GetIntFromObj(interp, objv[3], &ndims) != TCL_OK || ndims < 1) {
        SetResult("ndims is not an integer >= 1");
        return TCL_ERROR;
    }
    Tcl_Obj **ne_list;
    int ne_list_len;
    if (Tcl_ListObjGetElements(interp, objv[4], &ne_list_len, &ne_list) != TCL_OK) {
        SetResult("ne_list is not a list");
        return TCL_ERROR;
    }
//    if (ne_list_len != ndims) {
//        SetResult("ne_list length does not match ndims");
//        return TCL_ERROR;
//    }
    int64_t ne[ndims];
    for (int i = 0; i < ndims; i++) {
        if (Tcl_GetLongFromObj(interp, ne_list[i], &ne[i]) != TCL_OK || ne[i] < 0) {
            SetResult("ne_list element is not an integer >= 0");
            return TCL_ERROR;
        }
    }

    enum ggml_type type = ml_GetType(interp, objv[2]);
    struct ggml_tensor *tensor = ggml_new_tensor(ctx->ggml_ctx, type, ndims, ne);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
    tensor_ptr->ctx = ctx;
    tensor_ptr->next = NULL;
    tensor_ptr->prev = NULL;
    CMD_TENSOR_NAME(tensor_ptr->handle, tensor_ptr);
    ml_RegisterTensor(tensor_ptr->handle, tensor_ptr);
    ml_InsertTensorToList(ctx, tensor_ptr);

    SetResult(tensor_ptr->handle);
    return TCL_OK;
}

/*static*/ int ml_NewTensor1DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NewTensor1DCmd\n"));
    CheckArgs(4, 4, 1, "context_handle type ne0");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    int ne0;
    if (Tcl_GetIntFromObj(interp, objv[3], &ne0) != TCL_OK || ne0 < 0) {
        SetResult("ne0 is not an integer >= 0");
        return TCL_ERROR;
    }

    enum ggml_type type = ml_GetType(interp, objv[2]);
    struct ggml_tensor *tensor = ggml_new_tensor_1d(ctx->ggml_ctx, type, ne0);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
    tensor_ptr->ctx = ctx;
    tensor_ptr->next = NULL;
    tensor_ptr->prev = NULL;
    CMD_TENSOR_NAME(tensor_ptr->handle, tensor_ptr);
    ml_RegisterTensor(tensor_ptr->handle, tensor_ptr);
    ml_InsertTensorToList(ctx, tensor_ptr);

    SetResult(tensor_ptr->handle);
    return TCL_OK;
}

/*static*/ int ml_NewTensor2DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NewTensor1DCmd\n"));
    CheckArgs(5, 5, 1, "context_handle type ne0 ne1");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    int ne0;
    if (Tcl_GetIntFromObj(interp, objv[3], &ne0) != TCL_OK || ne0 < 0) {
        SetResult("ne0 is not an integer >= 0");
        return TCL_ERROR;
    }
    int ne1;
    if (Tcl_GetIntFromObj(interp, objv[4], &ne1) != TCL_OK || ne1 < 0) {
        SetResult("ne1 is not an integer >= 0");
        return TCL_ERROR;
    }

    enum ggml_type type = ml_GetType(interp, objv[2]);
    struct ggml_tensor *tensor = ggml_new_tensor_2d(ctx->ggml_ctx, type, ne0, ne1);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
    tensor_ptr->ctx = ctx;
    tensor_ptr->next = NULL;
    tensor_ptr->prev = NULL;
    ml_InsertTensorToList(ctx, tensor_ptr);

    CMD_TENSOR_NAME(tensor_ptr->handle, tensor_ptr);
    ml_RegisterTensor(tensor_ptr->handle, tensor_ptr);

    SetResult(tensor_ptr->handle);
    return TCL_OK;
}

/*static*/ int ml_NewTensor3DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NewTensor1DCmd\n"));
    CheckArgs(6, 6, 1, "context_handle type ne0 ne1 ne2");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    int ne0;
    if (Tcl_GetIntFromObj(interp, objv[3], &ne0) != TCL_OK || ne0 < 0) {
        SetResult("ne0 is not an integer >= 0");
        return TCL_ERROR;
    }
    int ne1;
    if (Tcl_GetIntFromObj(interp, objv[4], &ne1) != TCL_OK || ne1 < 0) {
        SetResult("ne1 is not an integer >= 0");
        return TCL_ERROR;
    }
    int ne2;
    if (Tcl_GetIntFromObj(interp, objv[5], &ne2) != TCL_OK || ne2 < 0) {
        SetResult("ne2 is not an integer >= 0");
        return TCL_ERROR;
    }

    enum ggml_type type = ml_GetType(interp, objv[2]);
    struct ggml_tensor *tensor = ggml_new_tensor_3d(ctx->ggml_ctx, type, ne0, ne1, ne2);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
    tensor_ptr->ctx = ctx;
    tensor_ptr->next = NULL;
    tensor_ptr->prev = NULL;
    ml_InsertTensorToList(ctx, tensor_ptr);

    CMD_TENSOR_NAME(tensor_ptr->handle, tensor_ptr);
    ml_RegisterTensor(tensor_ptr->handle, tensor_ptr);

    SetResult(tensor_ptr->handle);
    return TCL_OK;
}

/*static*/ int ml_NewTensor4DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NewTensor1DCmd\n"));
    CheckArgs(7, 7, 1, "context_handle type ne0 ne1 ne2 ne3");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    int ne0;
    if (Tcl_GetIntFromObj(interp, objv[3], &ne0) != TCL_OK || ne0 < 0) {
        SetResult("ne0 is not an integer >= 0");
        return TCL_ERROR;
    }
    int ne1;
    if (Tcl_GetIntFromObj(interp, objv[4], &ne1) != TCL_OK || ne1 < 0) {
        SetResult("ne1 is not an integer >= 0");
        return TCL_ERROR;
    }
    int ne2;
    if (Tcl_GetIntFromObj(interp, objv[5], &ne2) != TCL_OK || ne2 < 0) {
        SetResult("ne2 is not an integer >= 0");
        return TCL_ERROR;
    }
    int ne3;
    if (Tcl_GetIntFromObj(interp, objv[6], &ne3) != TCL_OK || ne3 < 0) {
        SetResult("ne3 is not an integer >= 0");
        return TCL_ERROR;
    }

    enum ggml_type type = ml_GetType(interp, objv[2]);
    struct ggml_tensor *tensor = ggml_new_tensor_4d(ctx->ggml_ctx, type, ne0, ne1, ne2, ne3);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
    tensor_ptr->ctx = ctx;
    tensor_ptr->next = NULL;
    tensor_ptr->prev = NULL;
    ml_InsertTensorToList(ctx, tensor_ptr);

    CMD_TENSOR_NAME(tensor_ptr->handle, tensor_ptr);
    ml_RegisterTensor(tensor_ptr->handle, tensor_ptr);

    SetResult(tensor_ptr->handle);
    return TCL_OK;
}

/*static*/ int ml_AddCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "AddCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_a tensor_b");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_a_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *a = ml_GetInternalFromTensor(tensor_a_handle);
    if (!a) {
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }
    const char *tensor_b_handle = Tcl_GetString(objv[3]);
    ml_tensor_t *b = ml_GetInternalFromTensor(tensor_b_handle);
    if (!b) {
        SetResult("tensor b handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_add(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
    tensor_ptr->ctx = ctx;
    tensor_ptr->next = NULL;
    tensor_ptr->prev = NULL;
    ml_InsertTensorToList(ctx, tensor_ptr);

    CMD_TENSOR_NAME(tensor_ptr->handle, tensor_ptr);
    ml_RegisterTensor(tensor_ptr->handle, tensor_ptr);

    SetResult(tensor_ptr->handle);
    return TCL_OK;
}

/*static*/ int ml_MulCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "MulCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_a tensor_b");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_a_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *a = ml_GetInternalFromTensor(tensor_a_handle);
    if (!a) {
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }
    const char *tensor_b_handle = Tcl_GetString(objv[3]);
    ml_tensor_t *b = ml_GetInternalFromTensor(tensor_b_handle);
    if (!b) {
        SetResult("tensor b handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_mul(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
    tensor_ptr->ctx = ctx;
    tensor_ptr->next = NULL;
    tensor_ptr->prev = NULL;
    ml_InsertTensorToList(ctx, tensor_ptr);

    CMD_TENSOR_NAME(tensor_ptr->handle, tensor_ptr);
    ml_RegisterTensor(tensor_ptr->handle, tensor_ptr);

    SetResult(tensor_ptr->handle);
    return TCL_OK;
}

/*static*/ int ml_SumCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SumCmd\n"));
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
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_sum(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
    if (!output_tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *output_tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    output_tensor_ptr->ggml_tensor = output_tensor;
    output_tensor_ptr->ctx = ctx;
    output_tensor_ptr->next = NULL;
    output_tensor_ptr->prev = NULL;
    ml_InsertTensorToList(ctx, output_tensor_ptr);

    CMD_TENSOR_NAME(output_tensor_ptr->handle, output_tensor_ptr);
    ml_RegisterTensor(output_tensor_ptr->handle, output_tensor_ptr);

    SetResult(output_tensor_ptr->handle);
    return TCL_OK;
}
