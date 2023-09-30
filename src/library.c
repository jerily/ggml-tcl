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

static int ggml_ModuleInitialized;

static Tcl_HashTable ggml_ContextToInternal_HT;
static Tcl_Mutex ggml_ContextToInternal_HT_Mutex;

static int
ggml_RegisterContext(const char *name, struct ggml_context *internal) {

    Tcl_HashEntry *entryPtr;
    int newEntry;
    Tcl_MutexLock(&ggml_ContextToInternal_HT_Mutex);
    entryPtr = Tcl_CreateHashEntry(&ggml_ContextToInternal_HT, (char *) name, &newEntry);
    if (newEntry) {
        Tcl_SetHashValue(entryPtr, (ClientData) internal);
    }
    Tcl_MutexUnlock(&ggml_ContextToInternal_HT_Mutex);

    DBG(fprintf(stderr, "--> RegisterContext: name=%s internal=%p %s\n", name, internal,
                newEntry ? "entered into" : "already in"));

    return newEntry;
}

static int
ggml_UnregisterContext(const char *name) {

    Tcl_HashEntry *entryPtr;

    Tcl_MutexLock(&ggml_ContextToInternal_HT_Mutex);
    entryPtr = Tcl_FindHashEntry(&ggml_ContextToInternal_HT, (char *) name);
    if (entryPtr != NULL) {
        Tcl_DeleteHashEntry(entryPtr);
    }
    Tcl_MutexUnlock(&ggml_ContextToInternal_HT_Mutex);

    DBG(fprintf(stderr, "--> UnregisterContext: name=%s entryPtr=%p\n", name, entryPtr));

    return entryPtr != NULL;
}

static struct ggml_context *
ggml_GetInternalFromContext(const char *name) {
    struct ggml_context *internal = NULL;
    Tcl_HashEntry *entryPtr;

    Tcl_MutexLock(&ggml_ContextToInternal_HT_Mutex);
    entryPtr = Tcl_FindHashEntry(&ggml_ContextToInternal_HT, (char *) name);
    if (entryPtr != NULL) {
        internal = (struct ggml_context *) Tcl_GetHashValue(entryPtr);
    }
    Tcl_MutexUnlock(&ggml_ContextToInternal_HT_Mutex);

    return internal;
}


static int ggml_CreateContextCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "CreateContextCmd\n"));
    CheckArgs(2, 2, 1, "config_dict");

    size_t mem_size = 16*1024*1024;
    struct ggml_init_params params = {
        .mem_size   = mem_size,                      // bytes
        .mem_buffer = Tcl_Alloc(mem_size),      // if NULL, memory will be allocated internally
        .no_alloc   = 1,                             // don't allocate memory for the tensor data
    };

    // memory allocation happens here
    struct ggml_context * ctx = ggml_init(params);

    char handle[30];
    CMD_CONTEXT_NAME(handle, ctx);
    ggml_RegisterContext(handle, ctx);

    SetResult(handle);
    return TCL_OK;

}


static int ggml_DestroyContextCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "DestroyContextCmd\n"));
    CheckArgs(2, 2, 1, "handle");
    const char *handle = Tcl_GetString(objv[1]);
    struct ggml_context *ctx = ggml_GetInternalFromContext(handle);
    if (!ctx) {
        SetResult("ggml context handle not found");
        return TCL_ERROR;
    }
    if (!ggml_UnregisterContext(handle)) {
        SetResult("unregister server name failed");
        return TCL_ERROR;
    }

    Tcl_Free(ggml_get_mem_buffer(ctx));
    ggml_free(ctx);


}


static void ggml_ExitHandler(ClientData unused) {
}


void ggml_InitModule() {
    if (!ggml_ModuleInitialized) {
        Tcl_MutexLock(&ggml_ContextToInternal_HT_Mutex);
        Tcl_InitHashTable(&ggml_ContextToInternal_HT, TCL_STRING_KEYS);
        Tcl_MutexUnlock(&ggml_ContextToInternal_HT_Mutex);

        ggml_ModuleInitialized = 1;
        DBG(fprintf(stderr, "ggml-tcl module initialized\n"));
    }
}

int Ggml_Init(Tcl_Interp *interp) {
    if (Tcl_InitStubs(interp, "8.6", 0) == NULL) {
        return TCL_ERROR;
    }

    ggml_InitModule();

    Tcl_CreateNamespace(interp, "::ggml", NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::create_context", ggml_CreateContextCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::destroy_context", ggml_DestroyContextCmd, NULL, NULL);

    return Tcl_PkgProvide(interp, "ggml", XSTR(PROJECT_VERSION));
}

#ifdef USE_NAVISERVER
int Ns_ModuleInit(const char *server, const char *module) {
    Ns_TclRegisterTrace(server, (Ns_TclTraceProc *) Bcrypt_Init, server, NS_TCL_TRACE_CREATE);
    return NS_OK;
}
#endif
