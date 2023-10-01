/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */
#ifndef GGML_TCL_COMMON_H
#define GGML_TCL_COMMON_H

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

EXTERN void ml_InitContextHT();
EXTERN void ml_DeleteContextHT();
EXTERN void ml_InitCGraphHT();
EXTERN void ml_DeleteCGraphHT();
EXTERN void ml_InitTensorHT();
EXTERN void ml_DeleteTensorHT();

EXTERN int ml_RegisterContext(const char *name, ml_context_t *internal);
EXTERN int ml_UnregisterContext(const char *name);
EXTERN ml_context_t *ml_GetInternalFromContext(const char *name);
EXTERN int ml_RegisterCGraph(const char *name, ml_cgraph_t *internal);
EXTERN int ml_UnregisterCGraph(const char *name);
EXTERN ml_cgraph_t *ml_GetInternalFromCGraph(const char *name);
EXTERN int ml_RegisterTensor(const char *name, ml_tensor_t *internal);
EXTERN int ml_UnregisterTensor(const char *name);
EXTERN ml_tensor_t *ml_GetInternalFromTensor(const char *name);

#endif //GGML_TCL_COMMON_H
