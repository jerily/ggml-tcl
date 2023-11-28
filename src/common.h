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

#define GGML_TCL_CMD(x) int (x)(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[])

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

typedef struct ml_cgraph_s {
    ml_context_t *ctx;
    struct ggml_cgraph *ggml_cgraph;
    struct ml_cgraph_s *next;
    struct ml_cgraph_s *prev;
    char handle[30];
} ml_cgraph_t;

struct ml_context_s {
    char *mem_buffer;
    struct ggml_context *ggml_ctx;
    struct gguf_context *gguf_ctx;
    ml_cgraph_t *first_graph_ptr;
    ml_cgraph_t *last_graph_ptr;
    ml_tensor_t *first_tensor_ptr;
    ml_tensor_t *last_tensor_ptr;
    char handle[30];
};

void ml_InitContextHT();
void ml_DeleteContextHT();
void ml_InitCGraphHT();
void ml_DeleteCGraphHT();
void ml_InitTensorHT();
void ml_DeleteTensorHT();

int ml_RegisterContext(const char *name, ml_context_t *internal);
int ml_UnregisterContext(const char *name);
ml_context_t *ml_GetInternalFromContext(const char *name);
int ml_RegisterCGraph(const char *name, ml_cgraph_t *internal);
int ml_UnregisterCGraph(const char *name);
ml_cgraph_t *ml_GetInternalFromCGraph(const char *name);
int ml_RegisterTensor(const char *name, ml_tensor_t *internal);
int ml_UnregisterTensor(const char *name);
ml_tensor_t *ml_GetInternalFromTensor(const char *name);

#endif //GGML_TCL_COMMON_H
