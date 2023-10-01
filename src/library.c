/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */
#include "library.h"
#include <stdio.h>
#include <string.h>
#include <ggml.h>

#define XSTR(s) STR(s)
#define STR(s) #s

#ifdef DEBUG
# define DBG(x) x
#else
# define DBG(x)
#endif

#define CheckArgs(min,max,n,msg) \
                 if ((objc < min) || (objc >max)) { \
                     Tcl_WrongNumArgs(interp, n, objv, msg); \
                     return TCL_ERROR; \
                 }

#define SetResult(str) Tcl_ResetResult(interp); \
                     Tcl_SetStringObj(Tcl_GetObjResult(interp), (str), -1)

#define CMD_CONTEXT_NAME(s, internal) sprintf((s), "_GGML_CTX_%p", (internal))
#define CMD_TENSOR_NAME(s, internal) sprintf((s), "_GGML_T_%p", (internal))
#define CMD_CGRAPH_NAME(s, internal) sprintf((s), "_GGML_CG_%p", (internal))

static int ml_ModuleInitialized;

static Tcl_HashTable ml_ContextToInternal_HT;
static Tcl_Mutex ml_ContextToInternal_HT_Mutex;

static Tcl_HashTable ml_CGraphToInternal_HT;
static Tcl_Mutex ml_CGraphToInternal_HT_Mutex;

static Tcl_HashTable ml_TensorToInternal_HT;
static Tcl_Mutex ml_TensorToInternal_HT_Mutex;

typedef struct ml_context_s ml_context_t;

typedef struct ml_tensor_s {
    struct ggml_tensor *ggml_tensor;
    ml_context_t *ctx;
    struct ml_tensor_s *next;
    struct ml_tensor_s *prev;
    char handle[30];
} ml_tensor_t;

typedef struct {
    ml_context_t *ctx;
    struct ggml_cgraph *ggml_cgraph;
    char handle[30];
} ml_cgraph_t;

struct ml_context_s {
    size_t mem_size;
    char *mem_buffer;
    struct ggml_context *ggml_ctx;
    ml_cgraph_t *gf;
    ml_cgraph_t *gb;
    ml_tensor_t *first_tensor_ptr;
    ml_tensor_t *last_tensor_ptr;
};

static int
ml_RegisterContext(const char *name, ml_context_t *internal) {

    Tcl_HashEntry *entryPtr;
    int newEntry;
    Tcl_MutexLock(&ml_ContextToInternal_HT_Mutex);
    entryPtr = Tcl_CreateHashEntry(&ml_ContextToInternal_HT, (char *) name, &newEntry);
    if (newEntry) {
        Tcl_SetHashValue(entryPtr, (ClientData) internal);
    }
    Tcl_MutexUnlock(&ml_ContextToInternal_HT_Mutex);

    DBG(fprintf(stderr, "--> RegisterContext: name=%s internal=%p %s\n", name, internal,
                newEntry ? "entered into" : "already in"));

    return newEntry;
}

static int
ml_UnregisterContext(const char *name) {

    Tcl_HashEntry *entryPtr;

    Tcl_MutexLock(&ml_ContextToInternal_HT_Mutex);
    entryPtr = Tcl_FindHashEntry(&ml_ContextToInternal_HT, (char *) name);
    if (entryPtr != NULL) {
        Tcl_DeleteHashEntry(entryPtr);
    }
    Tcl_MutexUnlock(&ml_ContextToInternal_HT_Mutex);

    DBG(fprintf(stderr, "--> UnregisterContext: name=%s entryPtr=%p\n", name, entryPtr));

    return entryPtr != NULL;
}

static ml_context_t *
ml_GetInternalFromContext(const char *name) {
    ml_context_t *internal = NULL;
    Tcl_HashEntry *entryPtr;

    Tcl_MutexLock(&ml_ContextToInternal_HT_Mutex);
    entryPtr = Tcl_FindHashEntry(&ml_ContextToInternal_HT, (char *) name);
    if (entryPtr != NULL) {
        internal = (ml_context_t *) Tcl_GetHashValue(entryPtr);
    }
    Tcl_MutexUnlock(&ml_ContextToInternal_HT_Mutex);

    return internal;
}

static int
ml_RegisterCGraph(const char *name, ml_cgraph_t *internal) {

    Tcl_HashEntry *entryPtr;
    int newEntry;
    Tcl_MutexLock(&ml_CGraphToInternal_HT_Mutex);
    entryPtr = Tcl_CreateHashEntry(&ml_CGraphToInternal_HT, (char *) name, &newEntry);
    if (newEntry) {
        Tcl_SetHashValue(entryPtr, (ClientData) internal);
    }
    Tcl_MutexUnlock(&ml_CGraphToInternal_HT_Mutex);

    DBG(fprintf(stderr, "--> RegisterCGraph: name=%s internal=%p %s\n", name, internal,
                newEntry ? "entered into" : "already in"));

    return newEntry;
}

static int
ml_UnregisterCGraph(const char *name) {

    Tcl_HashEntry *entryPtr;

    Tcl_MutexLock(&ml_CGraphToInternal_HT_Mutex);
    entryPtr = Tcl_FindHashEntry(&ml_CGraphToInternal_HT, (char *) name);
    if (entryPtr != NULL) {
        Tcl_DeleteHashEntry(entryPtr);
    }
    Tcl_MutexUnlock(&ml_CGraphToInternal_HT_Mutex);

    DBG(fprintf(stderr, "--> UnregisterCGraph: name=%s entryPtr=%p\n", name, entryPtr));

    return entryPtr != NULL;
}

static ml_cgraph_t *
ml_GetInternalFromCGraph(const char *name) {
    ml_cgraph_t *internal = NULL;
    Tcl_HashEntry *entryPtr;

    Tcl_MutexLock(&ml_CGraphToInternal_HT_Mutex);
    entryPtr = Tcl_FindHashEntry(&ml_CGraphToInternal_HT, (char *) name);
    if (entryPtr != NULL) {
        internal = (ml_cgraph_t *) Tcl_GetHashValue(entryPtr);
    }
    Tcl_MutexUnlock(&ml_CGraphToInternal_HT_Mutex);

    return internal;
}

static int
ml_RegisterTensor(const char *name, ml_tensor_t *internal) {

    Tcl_HashEntry *entryPtr;
    int newEntry;
    Tcl_MutexLock(&ml_TensorToInternal_HT_Mutex);
    entryPtr = Tcl_CreateHashEntry(&ml_TensorToInternal_HT, (char *) name, &newEntry);
    if (newEntry) {
        Tcl_SetHashValue(entryPtr, (ClientData) internal);
    }
    Tcl_MutexUnlock(&ml_TensorToInternal_HT_Mutex);

    DBG(fprintf(stderr, "--> RegisterTensor: name=%s internal=%p %s\n", name, internal,
                newEntry ? "entered into" : "already in"));

    return newEntry;
}

static int
ml_UnregisterTensor(const char *name) {

    Tcl_HashEntry *entryPtr;

    Tcl_MutexLock(&ml_TensorToInternal_HT_Mutex);
    entryPtr = Tcl_FindHashEntry(&ml_TensorToInternal_HT, (char *) name);
    if (entryPtr != NULL) {
        Tcl_DeleteHashEntry(entryPtr);
    }
    Tcl_MutexUnlock(&ml_TensorToInternal_HT_Mutex);

    DBG(fprintf(stderr, "--> UnregisterTensor: name=%s entryPtr=%p\n", name, entryPtr));

    return entryPtr != NULL;
}

static ml_tensor_t *
ml_GetInternalFromTensor(const char *name) {
    ml_tensor_t *internal = NULL;
    Tcl_HashEntry *entryPtr;

    Tcl_MutexLock(&ml_TensorToInternal_HT_Mutex);
    entryPtr = Tcl_FindHashEntry(&ml_TensorToInternal_HT, (char *) name);
    if (entryPtr != NULL) {
        internal = (ml_tensor_t *) Tcl_GetHashValue(entryPtr);
    }
    Tcl_MutexUnlock(&ml_TensorToInternal_HT_Mutex);

    return internal;
}


static int ml_CreateContextCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "CreateContextCmd\n"));
    CheckArgs(2, 2, 1, "config_dict");

    Tcl_Obj *memSizeKeyPtr = Tcl_NewStringObj("mem_size", -1);
    Tcl_IncrRefCount(memSizeKeyPtr);
    Tcl_Obj *memSizePtr;
    if (TCL_OK != Tcl_DictObjGet(interp, objv[1], memSizeKeyPtr, &memSizePtr)) {
        SetResult("error reading config dict");
        return TCL_ERROR;
    }
    Tcl_DecrRefCount(memSizeKeyPtr);

    size_t mem_size;
    if (memSizePtr) {
        if (Tcl_GetLongFromObj(interp, memSizePtr, &mem_size) != TCL_OK || mem_size <= 0) {
            SetResult("mem_size is not an integer > 0");
            return TCL_ERROR;
        }
    } else {
        SetResult("mem_size is not specified");
        return TCL_ERROR;
    }

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

    char handle[30];
    CMD_CONTEXT_NAME(handle, ctx);
    ml_RegisterContext(handle, ctx);

    SetResult(handle);
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
    if (!ml_UnregisterContext(handle)) {
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

static int ml_InsertTensorToList(ml_context_t *ctx, ml_tensor_t *internal) {
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

static int ml_GetGradCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_SetParamCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_SetF32Cmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_SetF321DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_GetF321DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_NumElementsCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_NewTensorCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_NewTensor1DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_NewTensor2DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_NewTensor3DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_NewTensor4DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_AddCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_MulCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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

static int ml_SumCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
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


static void ml_ExitHandler(ClientData unused) {
    Tcl_MutexLock(&ml_ContextToInternal_HT_Mutex);
    Tcl_DeleteHashTable(&ml_ContextToInternal_HT);
    Tcl_MutexUnlock(&ml_ContextToInternal_HT_Mutex);

    Tcl_MutexLock(&ml_CGraphToInternal_HT_Mutex);
    Tcl_DeleteHashTable(&ml_CGraphToInternal_HT);
    Tcl_MutexUnlock(&ml_CGraphToInternal_HT_Mutex);

    Tcl_MutexLock(&ml_TensorToInternal_HT_Mutex);
    Tcl_DeleteHashTable(&ml_TensorToInternal_HT);
    Tcl_MutexUnlock(&ml_TensorToInternal_HT_Mutex);

}


void ml_InitModule() {
    if (!ml_ModuleInitialized) {

        Tcl_MutexLock(&ml_ContextToInternal_HT_Mutex);
        Tcl_InitHashTable(&ml_ContextToInternal_HT, TCL_STRING_KEYS);
        Tcl_MutexUnlock(&ml_ContextToInternal_HT_Mutex);

        Tcl_MutexLock(&ml_CGraphToInternal_HT_Mutex);
        Tcl_InitHashTable(&ml_CGraphToInternal_HT, TCL_STRING_KEYS);
        Tcl_MutexUnlock(&ml_CGraphToInternal_HT_Mutex);

        Tcl_MutexLock(&ml_TensorToInternal_HT_Mutex);
        Tcl_InitHashTable(&ml_TensorToInternal_HT, TCL_STRING_KEYS);
        Tcl_MutexUnlock(&ml_TensorToInternal_HT_Mutex);

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
    Tcl_CreateObjCommand(interp, "::ggml::build_forward_ctx", ml_BuildForwardCtxCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::graph_compute", ml_GraphComputeCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::build_backward_ctx", ml_BuildBackwardCtxCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::graph_reset", ml_GraphResetCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_grad", ml_GetGradCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_param", ml_SetParamCmd, NULL, NULL);
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
    Tcl_CreateObjCommand(interp, "::ggml::graph_dump_dot", ml_GraphDumpDotCmd, NULL, NULL);

    return Tcl_PkgProvide(interp, "ggml", XSTR(PROJECT_VERSION));
}

#ifdef USE_NAVISERVER
int Ns_ModuleInit(const char *server, const char *module) {
    Ns_TclRegisterTrace(server, (Ns_TclTraceProc *) Ggml_Init, server, NS_TCL_TRACE_CREATE);
    return NS_OK;
}
#endif
