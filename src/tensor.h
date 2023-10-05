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
GGML_TCL_CMD(ml_DupCmd);
GGML_TCL_CMD(ml_DupInplaceCmd);
GGML_TCL_CMD(ml_AddCmd);
GGML_TCL_CMD(ml_AddInplaceCmd);
GGML_TCL_CMD(ml_Add1Cmd);
GGML_TCL_CMD(ml_Add1InplaceCmd);
//GGML_TCL_CMD(ml_AccCmd);
//GGML_TCL_CMD(ml_AccInplaceCmd);
GGML_TCL_CMD(ml_SubCmd);
GGML_TCL_CMD(ml_SubInplaceCmd);
GGML_TCL_CMD(ml_MulCmd);
GGML_TCL_CMD(ml_MulInplaceCmd);
GGML_TCL_CMD(ml_DivCmd);
GGML_TCL_CMD(ml_DivInplaceCmd);
GGML_TCL_CMD(ml_SqrCmd);
GGML_TCL_CMD(ml_SqrInplaceCmd);
GGML_TCL_CMD(ml_SqrtCmd);
GGML_TCL_CMD(ml_SqrtInplaceCmd);
GGML_TCL_CMD(ml_LogCmd);
GGML_TCL_CMD(ml_LogInplaceCmd);
// return scalar
GGML_TCL_CMD(ml_SumCmd);
GGML_TCL_CMD(ml_SumRowsCmd);
GGML_TCL_CMD(ml_MeanCmd);
GGML_TCL_CMD(ml_ArgmaxCmd);
// if a is the same shape as b, and a is not parameter, return a
// otherwise, return a new tensor: repeat(a) to fit in b
GGML_TCL_CMD(ml_RepeatCmd);
GGML_TCL_CMD(ml_RepeatBackCmd);
GGML_TCL_CMD(ml_ConcatCmd);
GGML_TCL_CMD(ml_AbsCmd);
GGML_TCL_CMD(ml_AbsInplaceCmd);
GGML_TCL_CMD(ml_SgnCmd);
GGML_TCL_CMD(ml_SgnInplaceCmd);
GGML_TCL_CMD(ml_NegCmd);
GGML_TCL_CMD(ml_NegInplaceCmd);
GGML_TCL_CMD(ml_StepCmd);
GGML_TCL_CMD(ml_StepInplaceCmd);
GGML_TCL_CMD(ml_TanhCmd);
GGML_TCL_CMD(ml_TanhInplaceCmd);
GGML_TCL_CMD(ml_EluCmd);
GGML_TCL_CMD(ml_EluInplaceCmd);
GGML_TCL_CMD(ml_ReluCmd);
GGML_TCL_CMD(ml_ReluInplaceCmd);
GGML_TCL_CMD(ml_GeluCmd);
GGML_TCL_CMD(ml_GeluInplaceCmd);
GGML_TCL_CMD(ml_GeluQuickCmd);
GGML_TCL_CMD(ml_GeluQuickInplaceCmd);
GGML_TCL_CMD(ml_SiluCmd);
GGML_TCL_CMD(ml_SiluInplaceCmd);
// a - x
// b - dy
GGML_TCL_CMD(ml_SiluBackCmd);
// normalize along rows
GGML_TCL_CMD(ml_NormCmd);
GGML_TCL_CMD(ml_NormInplaceCmd);
GGML_TCL_CMD(ml_RmsNormCmd);
GGML_TCL_CMD(ml_RmsNormInplaceCmd);

// group normalize along ne0*ne1*n_groups
// used in stable-diffusion
GGML_TCL_CMD(ml_GroupNormCmd);
GGML_TCL_CMD(ml_GroupNormInplaceCmd);

// a - x
// b - dy
GGML_TCL_CMD(ml_RmsNormBackCmd);

// A: n columns, m rows
// B: n columns, p rows  (i.e. we transpose it internally)
// result is m columns, p rows
GGML_TCL_CMD(ml_MulMatCmd);

// A: m columns, n rows,
// B: p columns, n rows,
// result is m columns, p rows
GGML_TCL_CMD(ml_OutProdCmd);

GGML_TCL_CMD(ml_ScaleCmd);
GGML_TCL_CMD(ml_ScaleInplaceCmd);

// b -> view(a,offset,nb1,nb2,3), return modified a
GGML_TCL_CMD(ml_SetCmd);
// b -> view(a,offset,nb1,nb2,3), return view(a)
GGML_TCL_CMD(ml_SetInplaceCmd);
GGML_TCL_CMD(ml_Set1DCmd);
GGML_TCL_CMD(ml_Set1DInplaceCmd);
// b -> view(a,offset,nb1,nb2,3), return modified a
GGML_TCL_CMD(ml_Set2DCmd);
// b -> view(a,offset,nb1,nb2,3), return view(a)
GGML_TCL_CMD(ml_Set2DInplaceCmd);

// a -> b, return view(b)
GGML_TCL_CMD(ml_CpyCmd);
// a -> b, in-place, return view(b)
GGML_TCL_CMD(ml_CpyInplaceCmd);
// make contiguous
GGML_TCL_CMD(ml_ContCmd);
GGML_TCL_CMD(ml_ContInplaceCmd);
// return view(a), b specifies the new shape
GGML_TCL_CMD(ml_ReshapeCmd);
// return view(a)
GGML_TCL_CMD(ml_Reshape1DCmd);
GGML_TCL_CMD(ml_Reshape2DCmd);
GGML_TCL_CMD(ml_Reshape3DCmd);
GGML_TCL_CMD(ml_Reshape4DCmd);
// offset in bytes
GGML_TCL_CMD(ml_View1DCmd);
GGML_TCL_CMD(ml_View2DCmd);
GGML_TCL_CMD(ml_View3DCmd);
GGML_TCL_CMD(ml_View4DCmd);
GGML_TCL_CMD(ml_PermuteCmd);
GGML_TCL_CMD(ml_TransposeCmd);
GGML_TCL_CMD(ml_GetRowsCmd);
GGML_TCL_CMD(ml_GetRowsBackCmd);
GGML_TCL_CMD(ml_DiagCmd);
// set elements above the diagonal to -INF
GGML_TCL_CMD(ml_DiagMaskInfCmd);
GGML_TCL_CMD(ml_DiagMaskInfInplaceCmd);
// set elements above the diagonal to 0
GGML_TCL_CMD(ml_DiagMaskZeroCmd);
GGML_TCL_CMD(ml_DiagMaskZeroInplaceCmd);
GGML_TCL_CMD(ml_SoftMaxCmd);
GGML_TCL_CMD(ml_SoftMaxInplaceCmd);
GGML_TCL_CMD(ml_SoftMaxBackCmd);
GGML_TCL_CMD(ml_SoftMaxBackInplaceCmd);
// rotary position embedding
// if mode & 1 == 1, skip n_past elements
// if mode & 2 == 1, GPT-NeoX style
// if mode & 4 == 1, ChatGLM style
GGML_TCL_CMD(ml_RopeCmd);
// in-place, returns view(a)
GGML_TCL_CMD(ml_RopeInplaceCmd);
// custom RoPE
GGML_TCL_CMD(ml_RopeCustomCmd);
GGML_TCL_CMD(ml_RopeCustomInplaceCmd);
// xPos RoPE, in-place, returns view(a)
GGML_TCL_CMD(ml_RopeXposInplaceCmd);
GGML_TCL_CMD(ml_RopeBackCmd);
// alibi position embedding
// in-place, returns view(a)
GGML_TCL_CMD(ml_AlibiCmd);
// clamp
// in-place, returns view(a)
GGML_TCL_CMD(ml_ClampCmd);
GGML_TCL_CMD(ml_Conv1DCmd);
// conv_1d with padding = half
// alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
GGML_TCL_CMD(ml_Conv1DPhCmd);
GGML_TCL_CMD(ml_ConvTranspose1DCmd);
GGML_TCL_CMD(ml_Conv2DCmd);
// kernel size is a->ne[0] x a->ne[1]
// stride is equal to kernel size
// padding is zero
// example:
// a:     16   16    3  768
// b:   1024 1024    3    1
// res:   64   64  768    1
// used in sam
GGML_TCL_CMD(ml_Conv2DSkP0Cmd);
// kernel size is a->ne[0] x a->ne[1]
// stride is 1
// padding is half
// example:
// a:      3    3    256  256
// b:     64   64    256    1
// res:   64   64    256    1
// used in sam
GGML_TCL_CMD(ml_Conv2DS1PhCmd);
GGML_TCL_CMD(ml_ConvTranspose2DP0Cmd);
GGML_TCL_CMD(ml_Pool1DCmd);
GGML_TCL_CMD(ml_Pool2DCmd);
// nearest interpolate
// used in stable-diffusion
GGML_TCL_CMD(ml_UpscaleCmd);
GGML_TCL_CMD(ml_FlashAttnCmd);
GGML_TCL_CMD(ml_FlashAttnBackCmd);
GGML_TCL_CMD(ml_FlashFFCmd);
// partition into non-overlapping windows with padding if needed
// example:
// a:   768   64   64    1
// w:    14
// res: 768   14   14    25
// used in sam
GGML_TCL_CMD(ml_WinPartCmd);
GGML_TCL_CMD(ml_WinUnpartCmd);
GGML_TCL_CMD(ml_UnaryCmd);
GGML_TCL_CMD(ml_UnaryInplaceCmd);
// used in sam
GGML_TCL_CMD(ml_GetRelPosCmd);
// used in sam
GGML_TCL_CMD(ml_AddRelPosCmd);
GGML_TCL_CMD(ml_AddRelPosInplaceCmd);
GGML_TCL_CMD(ml_CrossEntropyLossCmd);
GGML_TCL_CMD(ml_CrossEntropyLossBackCmd);

#endif //GGML_TCL_TENSOR_H
