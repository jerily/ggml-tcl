#include <tcl.h>
#include "common.h"

Tcl_HashTable ml_ContextToInternal_HT;
Tcl_Mutex ml_ContextToInternal_HT_Mutex;

Tcl_HashTable ml_CGraphToInternal_HT;
Tcl_Mutex ml_CGraphToInternal_HT_Mutex;

Tcl_HashTable ml_TensorToInternal_HT;
Tcl_Mutex ml_TensorToInternal_HT_Mutex;

void ml_InitContextHT() {
    Tcl_MutexLock(&ml_ContextToInternal_HT_Mutex);
    Tcl_InitHashTable(&ml_ContextToInternal_HT, TCL_STRING_KEYS);
    Tcl_MutexUnlock(&ml_ContextToInternal_HT_Mutex);
}

void ml_DeleteContextHT() {
    Tcl_MutexLock(&ml_ContextToInternal_HT_Mutex);
    Tcl_DeleteHashTable(&ml_ContextToInternal_HT);
    Tcl_MutexUnlock(&ml_ContextToInternal_HT_Mutex);
}

void ml_InitCGraphHT() {

    Tcl_MutexLock(&ml_CGraphToInternal_HT_Mutex);
    Tcl_InitHashTable(&ml_CGraphToInternal_HT, TCL_STRING_KEYS);
    Tcl_MutexUnlock(&ml_CGraphToInternal_HT_Mutex);
}

void ml_DeleteCGraphHT() {
    Tcl_MutexLock(&ml_CGraphToInternal_HT_Mutex);
    Tcl_DeleteHashTable(&ml_CGraphToInternal_HT);
    Tcl_MutexUnlock(&ml_CGraphToInternal_HT_Mutex);
}

void ml_InitTensorHT() {

    Tcl_MutexLock(&ml_TensorToInternal_HT_Mutex);
    Tcl_InitHashTable(&ml_TensorToInternal_HT, TCL_STRING_KEYS);
    Tcl_MutexUnlock(&ml_TensorToInternal_HT_Mutex);
}

void ml_DeleteTensorHT() {
    Tcl_MutexLock(&ml_TensorToInternal_HT_Mutex);
    Tcl_DeleteHashTable(&ml_TensorToInternal_HT);
    Tcl_MutexUnlock(&ml_TensorToInternal_HT_Mutex);
}

/*static*/ int
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

/*static*/ int
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

/*static*/ ml_context_t *
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

/*static*/ int
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

/*static*/ int
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

/*static*/ ml_cgraph_t *
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

/*static*/ int
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

/*static*/ int
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

/*static*/ ml_tensor_t *
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
