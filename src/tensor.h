/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */
#ifndef GGML_TCL_TENSOR_H
#define GGML_TCL_TENSOR_H

#define GGML_TCL_CMD(x) int (x)(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[])

GGML_TCL_CMD(ml_GetGradCmd);
GGML_TCL_CMD(ml_SetParamCmd);
GGML_TCL_CMD(ml_NumElementsCmd);
GGML_TCL_CMD(ml_NewTensorCmd);
GGML_TCL_CMD(ml_NewTensor1DCmd);
GGML_TCL_CMD(ml_NewTensor2DCmd);
GGML_TCL_CMD(ml_NewTensor3DCmd);
GGML_TCL_CMD(ml_NewTensor4DCmd);

GGML_TCL_CMD(ml_NewI32Cmd);
GGML_TCL_CMD(ml_NewF32Cmd);
GGML_TCL_CMD(ml_DupTensorCmd);
GGML_TCL_CMD(ml_ViewTensorCmd);
GGML_TCL_CMD(ml_SetZeroCmd);
GGML_TCL_CMD(ml_SetI32Cmd);
GGML_TCL_CMD(ml_SetF32Cmd);
GGML_TCL_CMD(ml_GetI321DCmd);
GGML_TCL_CMD(ml_SetI321DCmd);
GGML_TCL_CMD(ml_GetF321DCmd);
GGML_TCL_CMD(ml_SetF321DCmd);

//GGML_TCL_CMD(ml_GetTensorCmd);
//GGML_TCL_CMD(ml_GetNameCmd);
//GGML_TCL_CMD(ml_SetNameCmd);
//GGML_TCL_CMD(ml_FormatNameCmd);


// operations on tensors with backpropagation
//GGML_TCL_CMD(ml_DupCmd);
//GGML_TCL_CMD(ml_DupInplaceCmd);
GGML_TCL_CMD(ml_AddCmd);
//GGML_TCL_CMD(ml_AddInplaceCmd);
//GGML_TCL_CMD(ml_Add1Cmd);
//GGML_TCL_CMD(ml_Add1InplaceCmd);
//GGML_TCL_CMD(ml_AccCmd);
//GGML_TCL_CMD(ml_AccInplaceCmd);
//GGML_TCL_CMD(ml_SubCmd);
//GGML_TCL_CMD(ml_SubInplaceCmd);
GGML_TCL_CMD(ml_MulCmd);
//GGML_TCL_CMD(ml_MulInplaceCmd);
//GGML_TCL_CMD(ml_DivCmd);
//GGML_TCL_CMD(ml_DivInplaceCmd);
//GGML_TCL_CMD(ml_SqrCmd);
//GGML_TCL_CMD(ml_SqrInplaceCmd);
//GGML_TCL_CMD(ml_SqrtCmd);
//GGML_TCL_CMD(ml_SqrtInplaceCmd);
//GGML_TCL_CMD(ml_LogCmd);
//GGML_TCL_CMD(ml_LogInplaceCmd);

// return scalar
GGML_TCL_CMD(ml_SumCmd);

#endif //GGML_TCL_TENSOR_H
