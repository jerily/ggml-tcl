/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */
#include "library.h"
#include <stdio.h>
#include <string.h>
#include <ggml.h>
#include "common.h"
#include "context.h"
#include "tensor.h"
#include "cgraph.h"
#include "opt.h"

#define XSTR(s) STR(s)
#define STR(s) #s

static int ml_ModuleInitialized;

static void ml_ExitHandler(ClientData unused) {
    ml_DeleteContextHT();
    ml_DeleteCGraphHT();
    ml_DeleteTensorHT();
}


void ml_InitModule() {
    if (!ml_ModuleInitialized) {

        ml_InitContextHT();
        ml_InitCGraphHT();
        ml_InitTensorHT();

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
//    Tcl_CreateObjCommand(interp, "::ggml::write_context_to_file", ml_WriteContextToFileCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::load_context_from_file", ml_LoadContextFromFileCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::used_mem", ml_UsedMemCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_max_tensor_size", ml_GetMaxTensorSizeCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_mem_size", ml_GetMemSizeCmd, NULL, NULL);

    Tcl_CreateObjCommand(interp, "::ggml::build_forward_expand", ml_BuildForwardExpandCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::build_backward_expand", ml_BuildBackwardExpandCmd, NULL, NULL);

    Tcl_CreateObjCommand(interp, "::ggml::new_graph", ml_NewGraphCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_graph_custom", ml_NewGraphCustomCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::graph_compute", ml_GraphComputeCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::graph_reset", ml_GraphResetCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::graph_dump_dot", ml_GraphDumpDotCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::graph_cpy", ml_GraphCpyCmd, NULL, NULL);

    Tcl_CreateObjCommand(interp, "::ggml::opt_default_params", ml_OptDefaultParamsCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::opt", ml_OptCmd, NULL, NULL);

    Tcl_CreateObjCommand(interp, "::ggml::set_param", ml_SetParamCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_grad", ml_GetGradCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::nelements", ml_NumElementsCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor", ml_NewTensorCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_1d", ml_NewTensor1DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_2d", ml_NewTensor2DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_3d", ml_NewTensor3DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_tensor_4d", ml_NewTensor4DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_i32", ml_NewI32Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::new_f32", ml_NewF32Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::dup_tensor", ml_DupTensorCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::view_tensor", ml_ViewTensorCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_zero", ml_SetZeroCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_i32", ml_SetI32Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_f32", ml_SetF32Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_i32_1d", ml_GetI321DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_i32_1d", ml_SetI321DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_f32_1d", ml_GetF321DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_f32_1d", ml_SetF321DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::dup", ml_DupCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::dup_inplace", ml_DupInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::add", ml_AddCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::add_inplace", ml_AddInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::add1", ml_Add1Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::add1_inplace", ml_Add1InplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sub", ml_SubCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sub_inplace", ml_SubInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::mul", ml_MulCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::mul_inplace", ml_MulInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::div", ml_DivCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::div_inplace", ml_DivInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sqr", ml_SqrCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sqr_inplace", ml_SqrInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sqrt", ml_SqrtCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sqrt_inplace", ml_SqrtInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::log", ml_LogCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::log_inplace", ml_LogInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sum", ml_SumCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sum_rows", ml_SumRowsCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::mean", ml_MeanCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::argmax", ml_ArgmaxCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::repeat", ml_RepeatCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::repeat_back", ml_RepeatBackCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::concat", ml_ConcatCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::abs", ml_AbsCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sgn", ml_SgnCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::sgn_inplace", ml_SgnInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::neg", ml_NegCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::neg_inplace", ml_NegInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::step", ml_StepCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::step_inplace", ml_StepInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::tanh", ml_TanhCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::tanh_inplace", ml_TanhInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::elu", ml_EluCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::elu_inplace", ml_EluInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::relu", ml_ReluCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::relu_inplace", ml_ReluInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::gelu", ml_GeluCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::gelu_inplace", ml_GeluInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::gelu_quick", ml_GeluQuickCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::gelu_quick_inplace", ml_GeluQuickInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::silu", ml_SiluCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::silu_inplace", ml_SiluInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::silu_back", ml_SiluBackCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::norm", ml_NormCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::norm_inplace", ml_NormInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::rms_norm", ml_RmsNormCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::rms_norm_inplace", ml_RmsNormInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::group_norm", ml_GroupNormCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::group_norm_inplace", ml_GroupNormInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::rms_norm_back", ml_RmsNormBackCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::mul_mat", ml_MulMatCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::out_prod", ml_OutProdCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::scale", ml_ScaleCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::scale_inplace", ml_ScaleInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set", ml_SetCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_inplace", ml_SetInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_1d", ml_Set1DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_1d_inplace", ml_Set1DInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_2d", ml_Set2DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::set_2d_inplace", ml_Set2DInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::cpy", ml_CpyCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::cpy_inplace", ml_CpyInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::cont", ml_ContCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::cont_inplace", ml_ContInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::reshape", ml_ReshapeCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::reshape_1d", ml_Reshape1DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::reshape_2d", ml_Reshape2DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::reshape_3d", ml_Reshape3DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::reshape_4d", ml_Reshape4DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::view_1d", ml_View1DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::view_2d", ml_View2DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::view_3d", ml_View3DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::view_4d", ml_View4DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::permute", ml_PermuteCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::transpose", ml_TransposeCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_rows", ml_GetRowsCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_rows_back", ml_GetRowsBackCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::diag", ml_DiagCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::diag_mask_inf", ml_DiagMaskInfCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::diag_mask_inf_inplace", ml_DiagMaskInfInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::diag_mask_zero", ml_DiagMaskZeroCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::diag_mask_zero_inplace", ml_DiagMaskZeroInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::soft_max", ml_SoftMaxCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::soft_max_inplace", ml_SoftMaxInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::soft_max_back", ml_SoftMaxBackCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::soft_max_back_inplace", ml_SoftMaxBackInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::rope", ml_RopeCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::rope_inplace", ml_RopeInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::rope_custom", ml_RopeCustomCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::rope_custom_inplace", ml_RopeCustomInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::rope_xpos_inplace", ml_RopeXposInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::rope_back", ml_RopeBackCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::alibi", ml_AlibiCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::clamp", ml_ClampCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::conv_1d", ml_Conv1DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::conv_1d_ph", ml_Conv1DPhCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::conv_transpose_1d", ml_ConvTranspose1DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::conv_2d", ml_Conv2DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::conv_2d_sk_p0", ml_Conv2DSkP0Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::conv_2d_s1_ph", ml_Conv2DS1PhCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::conv_transpose_2d_p0", ml_ConvTranspose2DP0Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::pool_1d", ml_Pool1DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::pool_2d", ml_Pool2DCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::upscale", ml_UpscaleCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::flash_attn", ml_FlashAttnCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::flash_attn_back", ml_FlashAttnBackCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::flash_ff", ml_FlashFFCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::win_part", ml_WinPartCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::win_unpart", ml_WinUnpartCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::unary", ml_UnaryCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::unary_inplace", ml_UnaryInplaceCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::cross_entropy_loss", ml_CrossEntropyLossCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::cross_entropy_loss_back", ml_CrossEntropyLossBackCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::get_rel_pos", ml_GetRelPosCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::add_rel_pos", ml_AddRelPosCmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::ggml::add_rel_pos_inplace", ml_AddRelPosInplaceCmd, NULL, NULL);

    return Tcl_PkgProvide(interp, "ggml", XSTR(PROJECT_VERSION));
}

#ifdef USE_NAVISERVER
int Ns_ModuleInit(const char *server, const char *module) {
    Ns_TclRegisterTrace(server, (Ns_TclTraceProc *) Ggml_Init, server, NS_TCL_TRACE_CREATE);
    return NS_OK;
}
#endif
