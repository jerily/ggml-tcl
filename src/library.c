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

static int ml_ModuleInitialized;

static Tcl_HashTable ml_ContextToInternal_HT;
static Tcl_Mutex ml_ContextToInternal_HT_Mutex;

static Tcl_HashTable ml_TensorToInternal_HT;
static Tcl_Mutex ml_TensorToInternal_HT_Mutex;

typedef struct ml_tensor_s {
    struct ml_tensor_s *next;
    struct ml_tensor_s *prev;
    struct ggml_tensor *ggml_tensor;
    char handle[30];
} ml_tensor_t;

typedef struct {
    size_t mem_size;
    char *mem_buffer;
    struct ggml_context *ggml_ctx;
    ml_tensor_t *first_tensor_ptr;
    ml_tensor_t *last_tensor_ptr;
} ml_context_t;

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
        .mem_buffer = ctx->mem_buffer,      // if NULL, memory will be allocated internally
        .no_alloc   = 1,                             // don't allocate memory for the tensor data
    };

    // memory allocation happens here
    struct ggml_context *ggml_ctx = ggml_init(params);
    ctx->ggml_ctx = ggml_ctx;

    char handle[30];
    CMD_CONTEXT_NAME(handle, ctx);
    ml_RegisterContext(handle, ctx);

    SetResult(handle);
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
    ml_InsertTensorToList(ctx, tensor_ptr);

    CMD_TENSOR_NAME(tensor_ptr->handle, tensor_ptr);
    ml_RegisterTensor(tensor_ptr->handle, tensor_ptr);

    SetResult(tensor_ptr->handle);
    return TCL_OK;
}

static int ml_DestroyContextCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DestroyContextCmd\n"));
    CheckArgs(2, 2, 1, "handle");
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

    Tcl_Free(ctx->mem_buffer);
    ggml_free(ctx->ggml_ctx);
    Tcl_Free((char *) ctx);

    return TCL_OK;
}


static void ml_ExitHandler(ClientData unused) {
}


void ml_InitModule() {
    if (!ml_ModuleInitialized) {

        Tcl_MutexLock(&ml_ContextToInternal_HT_Mutex);
        Tcl_InitHashTable(&ml_ContextToInternal_HT, TCL_STRING_KEYS);
        Tcl_MutexUnlock(&ml_ContextToInternal_HT_Mutex);

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
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_1d", ml_NewTensor1DCmd, NULL, NULL);

    return Tcl_PkgProvide(interp, "ggml", XSTR(PROJECT_VERSION));
}

#ifdef USE_NAVISERVER
int Ns_ModuleInit(const char *server, const char *module) {
    Ns_TclRegisterTrace(server, (Ns_TclTraceProc *) Bcrypt_Init, server, NS_TCL_TRACE_CREATE);
    return NS_OK;
}
#endif
