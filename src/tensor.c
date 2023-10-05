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

int ml_InsertTensorToList(ml_context_t *ctx, ml_tensor_t *internal) {
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

int ml_GetGradCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_SetParamCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_NumElementsCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_NewTensorCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_NewTensor1DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_NewTensor2DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_NewTensor3DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_NewTensor4DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_NewI32Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NewI32Cmd\n"));
    CheckArgs(3, 3, 1, "context_handle int32_value");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    int value;
    if (Tcl_GetIntFromObj(interp, objv[2], &value) != TCL_OK) {
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_new_i32(ctx->ggml_ctx, value);
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

int ml_NewF32Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NewF32Cmd\n"));
    CheckArgs(3, 3, 1, "context_handle float_value");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    double value;
    if (Tcl_GetDoubleFromObj(interp, objv[2], &value) != TCL_OK) {
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_new_f32(ctx->ggml_ctx, value);
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

int ml_DupTensorCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DupTensorCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_dup(ctx->ggml_ctx, input_tensor_ptr->ggml_tensor);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *output_tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    output_tensor_ptr->ggml_tensor = tensor;
    output_tensor_ptr->ctx = ctx;
    output_tensor_ptr->next = NULL;
    output_tensor_ptr->prev = NULL;
    ml_InsertTensorToList(ctx, output_tensor_ptr);

    CMD_TENSOR_NAME(output_tensor_ptr->handle, output_tensor_ptr);
    ml_RegisterTensor(output_tensor_ptr->handle, output_tensor_ptr);

    SetResult(output_tensor_ptr->handle);
    return TCL_OK;
}

int ml_ViewTensorCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "ViewTensorCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_view_tensor(ctx->ggml_ctx, input_tensor_ptr->ggml_tensor);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *output_tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    output_tensor_ptr->ggml_tensor = tensor;
    output_tensor_ptr->ctx = ctx;
    output_tensor_ptr->next = NULL;
    output_tensor_ptr->prev = NULL;
    ml_InsertTensorToList(ctx, output_tensor_ptr);

    CMD_TENSOR_NAME(output_tensor_ptr->handle, output_tensor_ptr);
    ml_RegisterTensor(output_tensor_ptr->handle, output_tensor_ptr);

    SetResult(output_tensor_ptr->handle);
    return TCL_OK;
}

int ml_SetZeroCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SetZeroCmd\n"));
    CheckArgs(2, 2, 1, "tensor_handle");
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_set_zero(input_tensor_ptr->ggml_tensor);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }
    return TCL_OK;
}

int ml_SetI32Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SetI32Cmd\n"));
    CheckArgs(3, 3, 1, "tensor_handle int32_value");

    const char *tensor_handle = Tcl_GetString(objv[1]);
    ml_tensor_t *tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    int value;
    if (Tcl_GetIntFromObj(interp, objv[2], &value) != TCL_OK) {
        return TCL_ERROR;
    }

    ggml_set_i32(tensor_ptr->ggml_tensor, value);
    return TCL_OK;
}

int ml_SetF32Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_GetI321DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GetI321DCmd\n"));
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

    int32_t value = ggml_get_i32_1d(tensor_ptr->ggml_tensor, i);

    Tcl_SetObjResult(interp, Tcl_NewIntObj(value));
    return TCL_OK;
}

int ml_SetI321DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SetI321DCmd\n"));
    CheckArgs(4, 4, 1, "tensor_handle i int32_value");

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
    int32_t value;
    if (Tcl_GetIntFromObj(interp, objv[3], &value) != TCL_OK) {
        return TCL_ERROR;
    }

    ggml_set_i32_1d(tensor_ptr->ggml_tensor, i, value);
    return TCL_OK;
}

int ml_GetF321DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_SetF321DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_DupCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DupCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_dup(ctx->ggml_ctx, input_tensor_ptr->ggml_tensor);
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

int ml_DupInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DupInplaceCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_dup_inplace(ctx->ggml_ctx, input_tensor_ptr->ggml_tensor);
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

int ml_AddCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_AddInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "AddInplaceCmd\n"));
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

    struct ggml_tensor *tensor = ggml_add_inplace(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_Add1Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "Add1Cmd\n"));
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

    struct ggml_tensor *tensor = ggml_add1(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_Add1InplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "Add1InplaceCmd\n"));
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

    struct ggml_tensor *tensor = ggml_add1_inplace(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_SubCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SubCmd\n"));
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

    struct ggml_tensor *tensor = ggml_sub(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_SubInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SubInplaceCmd\n"));
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

    struct ggml_tensor *tensor = ggml_sub_inplace(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_MulCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

int ml_MulInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "MulInplaceCmd\n"));
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

    struct ggml_tensor *tensor = ggml_mul_inplace(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_DivCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DivCmd\n"));
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

    struct ggml_tensor *tensor = ggml_div(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_DivInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DivInplaceCmd\n"));
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

    struct ggml_tensor *tensor = ggml_div_inplace(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_SqrCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SqrCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_sqr(ctx->ggml_ctx, input_tensor_ptr->ggml_tensor);
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

int ml_SqrInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SqrInplaceCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_sqr_inplace(ctx->ggml_ctx, input_tensor_ptr->ggml_tensor);
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

int ml_SqrtCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SqrtCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_sqrt(ctx->ggml_ctx, input_tensor_ptr->ggml_tensor);
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

int ml_SqrtInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SqrtInplaceCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_sqrt_inplace(ctx->ggml_ctx, input_tensor_ptr->ggml_tensor);
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

int ml_LogCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "LogCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_log(ctx->ggml_ctx, input_tensor_ptr->ggml_tensor);
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

int ml_LogInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "LogInplaceCmd\n"));
    CheckArgs(3, 3, 1, "context_handle tensor_handle");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *input_tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!input_tensor_ptr) {
        SetResult("tensor a handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_log_inplace(ctx->ggml_ctx, input_tensor_ptr->ggml_tensor);
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

int ml_SumCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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
        SetResult("tensor handle not found");
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

int ml_SumRowsCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SumRowsCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_sum_rows(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_MeanCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "MeanCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_mean(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_ArgmaxCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "ArgmaxCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_argmax(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_RepeatCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RepeatCmd\n"));
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

    struct ggml_tensor *tensor = ggml_repeat(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_RepeatBackCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RepeatBackCmd\n"));
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

    struct ggml_tensor *tensor = ggml_repeat_back(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_ConcatCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "ConcatCmd\n"));
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

    struct ggml_tensor *tensor = ggml_concat(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_AbsCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "AbsCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_abs(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_AbsInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "AbsInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_abs_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_SgnCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SgnCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_sgn(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_SgnInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SgnInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_sgn_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_NegCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NegCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_neg(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_NegInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NegInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_neg_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_StepCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "StepCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_step(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_StepInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "StepInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_step_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_TanhCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "TanhCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_tanh(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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


int ml_TanhInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "TanhInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_tanh_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_EluCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "EluCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_elu(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_EluInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "EluInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_elu_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_ReluCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "ReluCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_relu(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_ReluInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "ReluInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_relu_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_GeluCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GeluCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_gelu(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_GeluInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GeluInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_gelu_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_GeluQuickCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GeluQuickCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_gelu_quick(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_GeluQuickInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GeluQuickInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_gelu_quick_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_SiluCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SiluCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_silu(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_SiluInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SiluInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_silu_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_SiluBackCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SiluBackCmd\n"));
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

    struct ggml_tensor *tensor = ggml_silu_back(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_NormCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NormCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle eps");
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

    float eps;
    if (Tcl_GetDoubleFromObj(interp, objv[3], &eps) != TCL_OK) {
        SetResult("eps must be a float");
        return TCL_ERROR;
    }
    struct ggml_tensor *output_tensor = ggml_norm(ctx->ggml_ctx, tensor_ptr->ggml_tensor, eps);
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
int ml_NormInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NormInplaceCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle eps");
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

    float eps;
    if (Tcl_GetDoubleFromObj(interp, objv[3], &eps) != TCL_OK) {
        SetResult("eps must be a float");
        return TCL_ERROR;
    }
    struct ggml_tensor *output_tensor = ggml_norm_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor, eps);
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

int ml_RmsNormCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RmsNormCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle eps");
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

    float eps;
    if (Tcl_GetDoubleFromObj(interp, objv[3], &eps) != TCL_OK) {
        SetResult("eps must be a float");
        return TCL_ERROR;
    }
    struct ggml_tensor *output_tensor = ggml_rms_norm(ctx->ggml_ctx, tensor_ptr->ggml_tensor, eps);
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

int ml_RmsNormInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RmsNormInplaceCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle eps");
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

    float eps;
    if (Tcl_GetDoubleFromObj(interp, objv[3], &eps) != TCL_OK) {
        SetResult("eps must be a float");
        return TCL_ERROR;
    }
    struct ggml_tensor *output_tensor = ggml_rms_norm_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor, eps);
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

int ml_GroupNormCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GroupNormCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle eps");
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

    int n_groups;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_groups) != TCL_OK) {
        SetResult("n_groups must be an integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *output_tensor = ggml_group_norm(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_groups);
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
int ml_GroupNormInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GroupNormInplaceCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle eps");
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

    int n_groups;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_groups) != TCL_OK) {
        SetResult("n_groups must be an integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *output_tensor = ggml_group_norm_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_groups);
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

int ml_RmsNormBackCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RmsNormBackCmd\n"));
    CheckArgs(5, 5, 1, "context_handle tensor_a tensor_b eps");
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

    float eps;
    if (Tcl_GetDoubleFromObj(interp, objv[4], &eps) != TCL_OK) {
        SetResult("eps must be a float");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_rms_norm_back(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor, eps);
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

int ml_MulMatCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "MulMatCmd\n"));
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

    struct ggml_tensor *tensor = ggml_mul_mat(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_OutProdCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "OutProdCmd\n"));
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

    struct ggml_tensor *tensor = ggml_out_prod(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_ScaleCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "ScaleCmd\n"));
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

    struct ggml_tensor *tensor = ggml_scale(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_ScaleInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "ScaleInplaceCmd\n"));
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

    struct ggml_tensor *tensor = ggml_scale_inplace(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_SetCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SetCmd\n"));
    CheckArgs(8, 8, 1, "context_handle tensor_a tensor_b nb1 nb2 nb3 offset");
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

    size_t nb1;
    if (Tcl_GetLongFromObj(interp, objv[4], &nb1) != TCL_OK) {
        SetResult("nb1 must be a long integer");
        return TCL_ERROR;
    }
    size_t nb2;
    if (Tcl_GetLongFromObj(interp, objv[5], &nb2) != TCL_OK) {
        SetResult("nb2 must be a long integer");
        return TCL_ERROR;
    }
    size_t nb3;
    if (Tcl_GetLongFromObj(interp, objv[6], &nb3) != TCL_OK) {
        SetResult("nb3 must be a long integer");
        return TCL_ERROR;
    }
    size_t offset;
    if (Tcl_GetLongFromObj(interp, objv[7], &offset) != TCL_OK) {
        SetResult("offset must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_set(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor, nb1, nb2, nb3, offset);
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

int ml_SetInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SetInplaceCmd\n"));
    CheckArgs(8, 8, 1, "context_handle tensor_a tensor_b nb1 nb2 nb3 offset");
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

    size_t nb1;
    if (Tcl_GetLongFromObj(interp, objv[4], &nb1) != TCL_OK) {
        SetResult("nb1 must be a long integer");
        return TCL_ERROR;
    }
    size_t nb2;
    if (Tcl_GetLongFromObj(interp, objv[5], &nb2) != TCL_OK) {
        SetResult("nb2 must be a long integer");
        return TCL_ERROR;
    }
    size_t nb3;
    if (Tcl_GetLongFromObj(interp, objv[6], &nb3) != TCL_OK) {
        SetResult("nb3 must be a long integer");
        return TCL_ERROR;
    }
    size_t offset;
    if (Tcl_GetLongFromObj(interp, objv[7], &offset) != TCL_OK) {
        SetResult("offset must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_set_inplace(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor, nb1, nb2, nb3, offset);
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

int ml_Set1DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "Set1DCmd\n"));
    CheckArgs(5, 5, 1, "context_handle tensor_a tensor_b offset");
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
    size_t offset;
    if (Tcl_GetLongFromObj(interp, objv[4], &offset) != TCL_OK) {
        SetResult("offset must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_set_1d(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor, offset);
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

int ml_Set1DInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "Set1DInplaceCmd\n"));
    CheckArgs(5, 5, 1, "context_handle tensor_a tensor_b offset");
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
    size_t offset;
    if (Tcl_GetLongFromObj(interp, objv[4], &offset) != TCL_OK) {
        SetResult("offset must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_set_1d_inplace(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor, offset);
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

int ml_Set2DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "Set2DCmd\n"));
    CheckArgs(6, 6, 1, "context_handle tensor_a tensor_b nb1 offset");
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
    size_t nb1;
    if (Tcl_GetLongFromObj(interp, objv[4], &nb1) != TCL_OK) {
        SetResult("nb1 must be a long integer");
        return TCL_ERROR;
    }
    size_t offset;
    if (Tcl_GetLongFromObj(interp, objv[5], &offset) != TCL_OK) {
        SetResult("offset must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_set_2d(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor, nb1, offset);
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

int ml_Set2DInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "Set2DInplaceCmd\n"));
    CheckArgs(6, 6, 1, "context_handle tensor_a tensor_b nb1 offset");
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
    size_t nb1;
    if (Tcl_GetLongFromObj(interp, objv[4], &nb1) != TCL_OK) {
        SetResult("nb1 must be a long integer");
        return TCL_ERROR;
    }
    size_t offset;
    if (Tcl_GetLongFromObj(interp, objv[5], &offset) != TCL_OK) {
        SetResult("offset must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_set_2d_inplace(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor, nb1, offset);
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

int ml_CpyCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "CpyCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_cpy(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_CpyInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "CpyInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_cpy_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_ContCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "ContCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_cont(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_ReshapeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "ReshapeCmd\n"));
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

    struct ggml_tensor *tensor = ggml_reshape(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_Reshape1DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "Reshape1DCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle ne0");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_a_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *a = ml_GetInternalFromTensor(tensor_a_handle);
    if (!a) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    uint64_t ne0;
    if (Tcl_GetLongFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_reshape_1d(ctx->ggml_ctx, a->ggml_tensor, ne0);
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

int ml_Reshape2DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "Reshape2DCmd\n"));
    CheckArgs(5, 5, 1, "context_handle tensor_handle ne0 ne1");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_a_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *a = ml_GetInternalFromTensor(tensor_a_handle);
    if (!a) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    uint64_t ne0;
    if (Tcl_GetLongFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 must be a long integer");
        return TCL_ERROR;
    }
    uint64_t ne1;
    if (Tcl_GetLongFromObj(interp, objv[4], &ne1) != TCL_OK) {
        SetResult("ne1 must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_reshape_2d(ctx->ggml_ctx, a->ggml_tensor, ne0, ne1);
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

int ml_Reshape3DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "Reshape3DCmd\n"));
    CheckArgs(6, 6, 1, "context_handle tensor_handle ne0 ne1 ne2");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_a_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *a = ml_GetInternalFromTensor(tensor_a_handle);
    if (!a) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    uint64_t ne0;
    if (Tcl_GetLongFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 must be a long integer");
        return TCL_ERROR;
    }
    uint64_t ne1;
    if (Tcl_GetLongFromObj(interp, objv[4], &ne1) != TCL_OK) {
        SetResult("ne1 must be a long integer");
        return TCL_ERROR;
    }
    uint64_t ne2;
    if (Tcl_GetLongFromObj(interp, objv[5], &ne2) != TCL_OK) {
        SetResult("ne2 must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_reshape_3d(ctx->ggml_ctx, a->ggml_tensor, ne0, ne1, ne2);
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

int ml_Reshape4DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "Reshape4DCmd\n"));
    CheckArgs(7, 7, 1, "context_handle tensor_handle ne0 ne1 ne2 ne3");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_a_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *a = ml_GetInternalFromTensor(tensor_a_handle);
    if (!a) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    uint64_t ne0;
    if (Tcl_GetLongFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 must be a long integer");
        return TCL_ERROR;
    }
    uint64_t ne1;
    if (Tcl_GetLongFromObj(interp, objv[4], &ne1) != TCL_OK) {
        SetResult("ne1 must be a long integer");
        return TCL_ERROR;
    }
    uint64_t ne2;
    if (Tcl_GetLongFromObj(interp, objv[5], &ne2) != TCL_OK) {
        SetResult("ne2 must be a long integer");
        return TCL_ERROR;
    }
    uint64_t ne3;
    if (Tcl_GetLongFromObj(interp, objv[6], &ne3) != TCL_OK) {
        SetResult("ne3 must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_reshape_4d(ctx->ggml_ctx, a->ggml_tensor, ne0, ne1, ne2, ne3);
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

int ml_View1DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "View1DCmd\n"));
    CheckArgs(5, 5, 1, "context_handle tensor_handle ne0 offset");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_a_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *a = ml_GetInternalFromTensor(tensor_a_handle);
    if (!a) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    int64_t ne0;
    if (Tcl_GetLongFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 must be a long integer");
        return TCL_ERROR;
    }
    size_t offset;
    if (Tcl_GetLongFromObj(interp, objv[4], &offset) != TCL_OK) {
        SetResult("offset must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_view_1d(ctx->ggml_ctx, a->ggml_tensor, ne0, offset);
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

int ml_View2DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "View2DCmd\n"));
    CheckArgs(7, 7, 1, "context_handle tensor_handle ne0 ne1 nb1 offset");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_a_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *a = ml_GetInternalFromTensor(tensor_a_handle);
    if (!a) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    int64_t ne0;
    if (Tcl_GetLongFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 must be a long integer");
        return TCL_ERROR;
    }
    int64_t ne1;
    if (Tcl_GetLongFromObj(interp, objv[4], &ne1) != TCL_OK) {
        SetResult("ne1 must be a long integer");
        return TCL_ERROR;
    }
    size_t nb1;
    if (Tcl_GetLongFromObj(interp, objv[5], &nb1) != TCL_OK) {
        SetResult("nb1 must be a long integer");
        return TCL_ERROR;
    }
    size_t offset;
    if (Tcl_GetLongFromObj(interp, objv[6], &offset) != TCL_OK) {
        SetResult("offset must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_view_2d(ctx->ggml_ctx, a->ggml_tensor, ne0, ne1, nb1, offset);
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

int ml_View3DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "View3DCmd\n"));
    CheckArgs(9, 9, 1, "context_handle tensor_handle ne0 ne1 ne2 nb1 nb2 offset");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_a_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *a = ml_GetInternalFromTensor(tensor_a_handle);
    if (!a) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    int64_t ne0;
    if (Tcl_GetLongFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 must be a long integer");
        return TCL_ERROR;
    }
    int64_t ne1;
    if (Tcl_GetLongFromObj(interp, objv[4], &ne1) != TCL_OK) {
        SetResult("ne1 must be a long integer");
        return TCL_ERROR;
    }
    int64_t ne2;
    if (Tcl_GetLongFromObj(interp, objv[5], &ne2) != TCL_OK) {
        SetResult("ne2 must be a long integer");
        return TCL_ERROR;
    }
    size_t nb1;
    if (Tcl_GetLongFromObj(interp, objv[6], &nb1) != TCL_OK) {
        SetResult("nb1 must be a long integer");
        return TCL_ERROR;
    }
    size_t nb2;
    if (Tcl_GetLongFromObj(interp, objv[7], &nb2) != TCL_OK) {
        SetResult("nb2 must be a long integer");
        return TCL_ERROR;
    }
    size_t offset;
    if (Tcl_GetLongFromObj(interp, objv[8], &offset) != TCL_OK) {
        SetResult("offset must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_view_3d(ctx->ggml_ctx, a->ggml_tensor, ne0, ne1, ne2, nb1, nb2, offset);
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

int ml_View4DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "View4DCmd\n"));
    CheckArgs(11, 11, 1, "context_handle tensor_handle ne0 ne1 ne2 ne3 nb1 nb2 nb3 offset");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *tensor_a_handle = Tcl_GetString(objv[2]);
    ml_tensor_t *a = ml_GetInternalFromTensor(tensor_a_handle);
    if (!a) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }
    int64_t ne0;
    if (Tcl_GetLongFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 must be a long integer");
        return TCL_ERROR;
    }
    int64_t ne1;
    if (Tcl_GetLongFromObj(interp, objv[4], &ne1) != TCL_OK) {
        SetResult("ne1 must be a long integer");
        return TCL_ERROR;
    }
    int64_t ne2;
    if (Tcl_GetLongFromObj(interp, objv[5], &ne2) != TCL_OK) {
        SetResult("ne2 must be a long integer");
        return TCL_ERROR;
    }
    int64_t ne3;
    if (Tcl_GetLongFromObj(interp, objv[6], &ne3) != TCL_OK) {
        SetResult("ne3 must be a long integer");
        return TCL_ERROR;
    }
    size_t nb1;
    if (Tcl_GetLongFromObj(interp, objv[7], &nb1) != TCL_OK) {
        SetResult("nb1 must be a long integer");
        return TCL_ERROR;
    }
    size_t nb2;
    if (Tcl_GetLongFromObj(interp, objv[8], &nb2) != TCL_OK) {
        SetResult("nb2 must be a long integer");
        return TCL_ERROR;
    }
    size_t nb3;
    if (Tcl_GetLongFromObj(interp, objv[9], &nb3) != TCL_OK) {
        SetResult("nb3 must be a long integer");
        return TCL_ERROR;
    }
    size_t offset;
    if (Tcl_GetLongFromObj(interp, objv[10], &offset) != TCL_OK) {
        SetResult("offset must be a long integer");
        return TCL_ERROR;
    }
    struct ggml_tensor *tensor = ggml_view_4d(ctx->ggml_ctx, a->ggml_tensor, ne0, ne1, nb1, offset);
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

int ml_PermuteCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "PermuteCmd\n"));
    CheckArgs(7, 7, 1, "context_handle tensor_handle axis0 axis1 axis2 axis3");
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
    int axis0;
    if (Tcl_GetIntFromObj(interp, objv[3], &axis0) != TCL_OK) {
        SetResult("axis0 must be an integer");
        return TCL_ERROR;
    }
    int axis1;
    if (Tcl_GetIntFromObj(interp, objv[4], &axis1) != TCL_OK) {
        SetResult("axis1 must be an integer");
        return TCL_ERROR;
    }
    int axis2;
    if (Tcl_GetIntFromObj(interp, objv[5], &axis2) != TCL_OK) {
        SetResult("axis2 must be an integer");
        return TCL_ERROR;
    }
    int axis3;
    if (Tcl_GetIntFromObj(interp, objv[6], &axis3) != TCL_OK) {
        SetResult("axis3 must be an integer");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_permute(ctx->ggml_ctx, tensor_ptr->ggml_tensor, axis0, axis1, axis2, axis3);
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

int ml_TransposeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "TransposeCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_transpose(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_GetRowsCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GetRowsCmd\n"));
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

    struct ggml_tensor *tensor = ggml_get_rows(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_GetRowsBackCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "GetRowsCmd\n"));
    CheckArgs(5, 5, 1, "context_handle tensor_a tensor_b tensor_c");
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
    const char *tensor_c_handle = Tcl_GetString(objv[4]);
    ml_tensor_t *c = ml_GetInternalFromTensor(tensor_c_handle);
    if (!c) {
        SetResult("tensor c handle not found");
        return TCL_ERROR;
    }

    struct ggml_tensor *tensor = ggml_get_rows_back(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor, c->ggml_tensor);
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

int ml_DiagCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DiagCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_diag(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_DiagMaskInfCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DiagMaskInfCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle n_past");
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
    int n_past;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_past) != TCL_OK) {
        SetResult("n_past must be an integer");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_diag_mask_inf(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_past);
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

int ml_DiagMaskInfInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DiagMaskInfInplaceCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle n_past");
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
    int n_past;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_past) != TCL_OK) {
        SetResult("n_past must be an integer");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_diag_mask_inf_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_past);
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

int ml_DiagMaskZeroCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DiagMaskZeroCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle n_past");
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
    int n_past;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_past) != TCL_OK) {
        SetResult("n_past must be an integer");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_diag_mask_zero(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_past);
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

int ml_DiagMaskZeroInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DiagMaskZeroInplaceCmd\n"));
    CheckArgs(4, 4, 1, "context_handle tensor_handle n_past");
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
    int n_past;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_past) != TCL_OK) {
        SetResult("n_past must be an integer");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_diag_mask_zero_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_past);
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

int ml_SoftMaxCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SoftMaxCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_soft_max(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_SoftMaxInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SoftMaxInplaceCmd\n"));
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

    struct ggml_tensor *output_tensor = ggml_soft_max_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor);
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

int ml_SoftMaxBackCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SoftMaxBackCmd\n"));
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

    struct ggml_tensor *tensor = ggml_soft_max_back(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_SoftMaxBackInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "SoftMaxBackInplaceCmd\n"));
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

    struct ggml_tensor *tensor = ggml_soft_max_back_inplace(ctx->ggml_ctx, a->ggml_tensor, b->ggml_tensor);
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

int ml_RopeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RopeCmd\n"));
    CheckArgs(7, 7, 1, "context_handle tensor_handle n_past n_dims mode n_ctx");
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
    int n_past;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_past) != TCL_OK) {
        SetResult("n_past must be an integer");
        return TCL_ERROR;
    }
    int n_dims;
    if (Tcl_GetIntFromObj(interp, objv[4], &n_dims) != TCL_OK) {
        SetResult("n_dims must be an integer");
        return TCL_ERROR;
    }
    int mode;
    if (Tcl_GetIntFromObj(interp, objv[5], &mode) != TCL_OK) {
        SetResult("mode must be an integer");
        return TCL_ERROR;
    }
    int n_ctx;
    if (Tcl_GetIntFromObj(interp, objv[6], &n_ctx) != TCL_OK) {
        SetResult("n_ctx must be an integer");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_rope(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_past, n_dims, mode, n_ctx);
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

int ml_RopeInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RopeInplaceCmd\n"));
    CheckArgs(7, 7, 1, "context_handle tensor_handle n_past n_dims mode n_ctx");
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
    int n_past;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_past) != TCL_OK) {
        SetResult("n_past must be an integer");
        return TCL_ERROR;
    }
    int n_dims;
    if (Tcl_GetIntFromObj(interp, objv[4], &n_dims) != TCL_OK) {
        SetResult("n_dims must be an integer");
        return TCL_ERROR;
    }
    int mode;
    if (Tcl_GetIntFromObj(interp, objv[5], &mode) != TCL_OK) {
        SetResult("mode must be an integer");
        return TCL_ERROR;
    }
    int n_ctx;
    if (Tcl_GetIntFromObj(interp, objv[6], &n_ctx) != TCL_OK) {
        SetResult("n_ctx must be an integer");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_rope_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_past, n_dims, mode, n_ctx);
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

int ml_RopeCustomCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RopeCustomCmd\n"));
    CheckArgs(9, 9, 1, "context_handle tensor_handle n_past n_dims mode n_ctx freq_base freq_scale");
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
    int n_past;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_past) != TCL_OK) {
        SetResult("n_past must be an integer");
        return TCL_ERROR;
    }
    int n_dims;
    if (Tcl_GetIntFromObj(interp, objv[4], &n_dims) != TCL_OK) {
        SetResult("n_dims must be an integer");
        return TCL_ERROR;
    }
    int mode;
    if (Tcl_GetIntFromObj(interp, objv[5], &mode) != TCL_OK) {
        SetResult("mode must be an integer");
        return TCL_ERROR;
    }
    int n_ctx;
    if (Tcl_GetIntFromObj(interp, objv[6], &n_ctx) != TCL_OK) {
        SetResult("n_ctx must be an integer");
        return TCL_ERROR;
    }
    float freq_base;
    if (Tcl_GetDoubleFromObj(interp, objv[7], &freq_base) != TCL_OK) {
        SetResult("freq_base must be a double");
        return TCL_ERROR;
    }
    float freq_scale;
    if (Tcl_GetDoubleFromObj(interp, objv[8], &freq_scale) != TCL_OK) {
        SetResult("freq_scale must be a double");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_rope_custom(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_past, n_dims, mode, n_ctx, freq_base, freq_scale);
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

int ml_RopeCustomInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RopeCustomInplaceCmd\n"));
    CheckArgs(9, 9, 1, "context_handle tensor_handle n_past n_dims mode n_ctx freq_base freq_scale");
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
    int n_past;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_past) != TCL_OK) {
        SetResult("n_past must be an integer");
        return TCL_ERROR;
    }
    int n_dims;
    if (Tcl_GetIntFromObj(interp, objv[4], &n_dims) != TCL_OK) {
        SetResult("n_dims must be an integer");
        return TCL_ERROR;
    }
    int mode;
    if (Tcl_GetIntFromObj(interp, objv[5], &mode) != TCL_OK) {
        SetResult("mode must be an integer");
        return TCL_ERROR;
    }
    int n_ctx;
    if (Tcl_GetIntFromObj(interp, objv[6], &n_ctx) != TCL_OK) {
        SetResult("n_ctx must be an integer");
        return TCL_ERROR;
    }
    float freq_base;
    if (Tcl_GetDoubleFromObj(interp, objv[7], &freq_base) != TCL_OK) {
        SetResult("freq_base must be a double");
        return TCL_ERROR;
    }
    float freq_scale;
    if (Tcl_GetDoubleFromObj(interp, objv[8], &freq_scale) != TCL_OK) {
        SetResult("freq_scale must be a double");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_rope_custom_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_past, n_dims, mode, n_ctx, freq_base, freq_scale);
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

int ml_RopeXposInplaceCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RopeXposInplaceCmd\n"));
    CheckArgs(7, 7, 1, "context_handle tensor_handle n_past n_dims base down");
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
    int n_past;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_past) != TCL_OK) {
        SetResult("n_past must be an integer");
        return TCL_ERROR;
    }
    int n_dims;
    if (Tcl_GetIntFromObj(interp, objv[4], &n_dims) != TCL_OK) {
        SetResult("n_dims must be an integer");
        return TCL_ERROR;
    }
    float base;
    if (Tcl_GetDoubleFromObj(interp, objv[5], &base) != TCL_OK) {
        SetResult("base must be a float");
        return TCL_ERROR;
    }
    int down;
    if (Tcl_GetBooleanFromObj(interp, objv[6], &down) != TCL_OK) {
        SetResult("down must be a boolean");
        return TCL_ERROR;
    }
    struct ggml_tensor *output_tensor = ggml_rope_xpos_inplace(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_past, n_dims, base, down);
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

int ml_RopeBackCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "RopeBackCmd\n"));
    CheckArgs(11, 11, 1, "context_handle tensor_handle n_past n_dims mode n_ctx freq_base freq_scale xpos_base xpos_down");
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
    int n_past;
    if (Tcl_GetIntFromObj(interp, objv[3], &n_past) != TCL_OK) {
        SetResult("n_past must be an integer");
        return TCL_ERROR;
    }
    int n_dims;
    if (Tcl_GetIntFromObj(interp, objv[4], &n_dims) != TCL_OK) {
        SetResult("n_dims must be an integer");
        return TCL_ERROR;
    }
    int mode;
    if (Tcl_GetIntFromObj(interp, objv[5], &mode) != TCL_OK) {
        SetResult("mode must be an integer");
        return TCL_ERROR;
    }
    int n_ctx;
    if (Tcl_GetIntFromObj(interp, objv[6], &n_ctx) != TCL_OK) {
        SetResult("n_ctx must be an integer");
        return TCL_ERROR;
    }
    float freq_base;
    if (Tcl_GetDoubleFromObj(interp, objv[7], &freq_base) != TCL_OK) {
        SetResult("freq_base must be a double");
        return TCL_ERROR;
    }
    float freq_scale;
    if (Tcl_GetDoubleFromObj(interp, objv[8], &freq_scale) != TCL_OK) {
        SetResult("freq_scale must be a double");
        return TCL_ERROR;
    }
    float xpos_base;
    if (Tcl_GetDoubleFromObj(interp, objv[9], &xpos_base) != TCL_OK) {
        SetResult("xpos_base must be a double");
        return TCL_ERROR;
    }
    int xpos_down;
    if (Tcl_GetBooleanFromObj(interp, objv[10], &xpos_down) != TCL_OK) {
        SetResult("xpos_down must be a boolean");
        return TCL_ERROR;
    }

    struct ggml_tensor *output_tensor = ggml_rope_back(ctx->ggml_ctx, tensor_ptr->ggml_tensor, n_past, n_dims, mode, n_ctx, freq_base, freq_scale, xpos_base, xpos_down);
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
