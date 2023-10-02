/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */
#ifndef GGML_TCL_CONTEXT_H
#define GGML_TCL_CONTEXT_H

EXTERN int ml_CreateContextCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
EXTERN int ml_DestroyContextCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
EXTERN int ml_LoadContextFromFileCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
EXTERN int ml_UsedMemCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
EXTERN int ml_GetMaxTensorSizeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
EXTERN int ml_GetMemSizeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);

#endif //GGML_TCL_CONTEXT_H
