/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */
#ifndef GGML_TCL_CONTEXT_H
#define GGML_TCL_CONTEXT_H

#define GGML_TCL_CMD(x) int (x)(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[])

GGML_TCL_CMD(ml_CreateContextCmd);
GGML_TCL_CMD(ml_DestroyContextCmd);
GGML_TCL_CMD(ml_LoadContextFromFileCmd);
GGML_TCL_CMD(ml_UsedMemCmd);
GGML_TCL_CMD(ml_GetMaxTensorSizeCmd);
GGML_TCL_CMD(ml_GetMemSizeCmd);

#endif //GGML_TCL_CONTEXT_H
