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

typedef struct ml_tensor_s {
    struct ml_tensor_s *next;
    struct ml_tensor_s *prev;
    struct ggml_tensor *ggml_tensor;
    char handle[30];
} ml_tensor_t;

typedef struct ml_context_t_ ml_context_t;

typedef struct {
    ml_context_t *ctx;
    struct ggml_cgraph *ggml_cgraph;
    char handle[30];
} ml_cgraph_t;

struct ml_context_t_ {
    size_t mem_size;
    char *mem_buffer;
    struct ggml_context *ggml_ctx;
    ml_cgraph_t *cgraph;
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

    size_t mem_size = 16*1024*1024;

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
    ctx->cgraph = NULL;
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

    if (ctx->cgraph) {
        if (!ml_UnregisterCGraph(ctx->cgraph->handle)) {
            SetResult("unregister cgraph name failed");
            return TCL_ERROR;
        }
        Tcl_Free((char *) ctx->cgraph);
        ctx->cgraph = NULL;
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
    ctx->cgraph = cgraph_ptr;

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

enum ggml_type ml_GetType(const char *type_str) {
    // TODO: add more types
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

    fprintf(stderr, "tensor_ptr->ggml_tensor=%p value=%f\n", tensor_ptr->ggml_tensor, value);
    ggml_set_f32(tensor_ptr->ggml_tensor, value);

    SetResult(tensor_ptr->handle);
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

static int ml_NewTensor1DCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "NewTensor1DCmd\n"));
    CheckArgs(4, 4, 1, "context_handle type ne0");
    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }
    const char *type_str = Tcl_GetString(objv[2]);
    int ne0;
    if (Tcl_GetIntFromObj(interp, objv[3], &ne0) != TCL_OK) {
        return TCL_ERROR;
    }

    enum ggml_type type = ml_GetType(type_str);
    struct ggml_tensor *tensor = ggml_new_tensor_1d(ctx->ggml_ctx, type, ne0);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
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
    const char *type_str = Tcl_GetString(objv[2]);
    int ne0;
    if (Tcl_GetIntFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 is not an integer");
        return TCL_ERROR;
    }
    int ne1;
    if (Tcl_GetIntFromObj(interp, objv[4], &ne1) != TCL_OK) {
        SetResult("ne1 is not an integer");
        return TCL_ERROR;
    }

    enum ggml_type type = ml_GetType(type_str);
    struct ggml_tensor *tensor = ggml_new_tensor_2d(ctx->ggml_ctx, type, ne0, ne1);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
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
    const char *type_str = Tcl_GetString(objv[2]);
    int ne0;
    if (Tcl_GetIntFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 is not an integer");
        return TCL_ERROR;
    }
    int ne1;
    if (Tcl_GetIntFromObj(interp, objv[4], &ne1) != TCL_OK) {
        SetResult("ne1 is not an integer");
        return TCL_ERROR;
    }
    int ne2;
    if (Tcl_GetIntFromObj(interp, objv[5], &ne2) != TCL_OK) {
        SetResult("ne2 is not an integer");
        return TCL_ERROR;
    }

    enum ggml_type type = ml_GetType(type_str);
    struct ggml_tensor *tensor = ggml_new_tensor_3d(ctx->ggml_ctx, type, ne0, ne1, ne2);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
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
    const char *type_str = Tcl_GetString(objv[2]);
    int ne0;
    if (Tcl_GetIntFromObj(interp, objv[3], &ne0) != TCL_OK) {
        SetResult("ne0 is not an integer");
        return TCL_ERROR;
    }
    int ne1;
    if (Tcl_GetIntFromObj(interp, objv[4], &ne1) != TCL_OK) {
        SetResult("ne1 is not an integer");
        return TCL_ERROR;
    }
    int ne2;
    if (Tcl_GetIntFromObj(interp, objv[5], &ne2) != TCL_OK) {
        SetResult("ne2 is not an integer");
        return TCL_ERROR;
    }
    int ne3;
    if (Tcl_GetIntFromObj(interp, objv[6], &ne3) != TCL_OK) {
        SetResult("ne3 is not an integer");
        return TCL_ERROR;
    }

    enum ggml_type type = ml_GetType(type_str);
    struct ggml_tensor *tensor = ggml_new_tensor_4d(ctx->ggml_ctx, type, ne0, ne1, ne2, ne3);
    if (!tensor) {
        SetResult("tensor allocation failed");
        return TCL_ERROR;
    }

    ml_tensor_t *tensor_ptr = (ml_tensor_t *) Tcl_Alloc(sizeof(ml_tensor_t));
    tensor_ptr->ggml_tensor = tensor;
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
    tensor_ptr->next = NULL;
    tensor_ptr->prev = NULL;
    ml_InsertTensorToList(ctx, tensor_ptr);

    CMD_TENSOR_NAME(tensor_ptr->handle, tensor_ptr);
    ml_RegisterTensor(tensor_ptr->handle, tensor_ptr);

    SetResult(tensor_ptr->handle);
    return TCL_OK;
}

static void ml_ExitHandler(ClientData unused) {
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
    Tcl_CreateObjCommand(interp, "::ggml::set_param", ml_SetParamCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_f32", ml_SetF32Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_f32_1d", ml_GetF321DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_1d", ml_NewTensor1DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_2d", ml_NewTensor2DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_3d", ml_NewTensor3DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_4d", ml_NewTensor4DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::add", ml_AddCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::mul", ml_MulCmd, NULL, NULL);

    return Tcl_PkgProvide(interp, "ggml", XSTR(PROJECT_VERSION));
}

#ifdef USE_NAVISERVER
int Ns_ModuleInit(const char *server, const char *module) {
    Ns_TclRegisterTrace(server, (Ns_TclTraceProc *) Bcrypt_Init, server, NS_TCL_TRACE_CREATE);
    return NS_OK;
}
#endif
