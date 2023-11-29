/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */

#include <tcl.h>
#include <ggml.h>
#include <stdlib.h>
#include "opt.h"

static Tcl_Obj *ml_NewFloatStringObj(float value) {
    char buffer[32];
    int len = snprintf(buffer, sizeof(buffer), "%f", value);
    return Tcl_NewStringObj(buffer, len);
}

static int ml_GetFloatFromObj(Tcl_Interp *interp, Tcl_Obj *objPtr, float *valuePtr) {
    int length;
    const char *str = Tcl_GetStringFromObj(objPtr, &length);
    char *endptr;
    *valuePtr = strtof(str, &endptr);
    if (endptr == str) {
        SetResult("could not convert to float");
        return TCL_ERROR;
    }
    return TCL_OK;
}

static const char *opt_types[] = {
        "ADAM",
        "LBFGS",
        NULL
};

enum ggml_opt_type ml_GetOptType(Tcl_Interp *interp, Tcl_Obj *objPtr) {
    int typeIndex;
    if (TCL_OK == Tcl_GetIndexFromObj(interp, objPtr, opt_types, "ggml_opt_type", 0, &typeIndex)) {
        return (enum ggml_opt_type) typeIndex;
    }
    return GGML_OPT_ADAM;
}

static const char *linesearch_methods[] = {
        "LINESEARCH_BACKTRACKING_ARMIJO",
        "LINESEARCH_BACKTRACKING_WOLFE",
        "LINESEARCH_BACKTRACKING_STRONG_WOLFE",
        NULL
};

enum ggml_linesearch ml_GetLinesearchMethod(Tcl_Interp *interp, Tcl_Obj *objPtr) {
    int typeIndex;
    if (TCL_OK == Tcl_GetIndexFromObj(interp, objPtr, linesearch_methods, "ggml_linesearch", 0, &typeIndex)) {
        return (enum ggml_linesearch) typeIndex;
    }
    return GGML_LINESEARCH_BACKTRACKING_WOLFE;
}

int ml_OptDefaultParamsCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "OptDefaultParamsCmd\n"));
    CheckArgs(2, 2, 1, "opt_type");

    enum ggml_opt_type opt_type = ml_GetOptType(interp, objv[1]);

    Tcl_Obj *dict_ptr = Tcl_NewDictObj();
    Tcl_IncrRefCount(dict_ptr);

    // create a dictionary with the default parameters for the given opt_type
    struct ggml_opt_params opt_params = ggml_opt_default_params(opt_type);

    if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("type", -1), Tcl_NewStringObj(opt_types[opt_type], -1))) {
        Tcl_DecrRefCount(dict_ptr);
        SetResult("could not add type key to dictionary");
        return TCL_ERROR;
    }

    if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("graph_size", -1), Tcl_NewLongObj(opt_params.graph_size))) {
        Tcl_DecrRefCount(dict_ptr);
        SetResult("could not add graph_size key to dictionary");
        return TCL_ERROR;
    }

    if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("n_threads", -1), Tcl_NewIntObj(opt_params.n_threads))) {
        Tcl_DecrRefCount(dict_ptr);
        SetResult("could not add n_threads key to dictionary");
        return TCL_ERROR;
    }

    if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("past", -1), Tcl_NewIntObj(opt_params.past))) {
        Tcl_DecrRefCount(dict_ptr);
        SetResult("could not add past key to dictionary");
        return TCL_ERROR;
    }

    if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("delta", -1), ml_NewFloatStringObj(opt_params.delta))) {
        Tcl_DecrRefCount(dict_ptr);
        SetResult("could not add delta key to dictionary");
        return TCL_ERROR;
    }

    if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("max_no_improvement", -1), Tcl_NewIntObj(opt_params.max_no_improvement))) {
        Tcl_DecrRefCount(dict_ptr);
        SetResult("could not add max_no_improvement key to dictionary");
        return TCL_ERROR;
    }

    if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("print_forward_graph", -1), Tcl_NewBooleanObj(opt_params.print_forward_graph))) {
        Tcl_DecrRefCount(dict_ptr);
        SetResult("could not add print_forward_graph key to dictionary");
        return TCL_ERROR;
    }

    if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("print_backward_graph", -1), Tcl_NewBooleanObj(opt_params.print_backward_graph))) {
        Tcl_DecrRefCount(dict_ptr);
        SetResult("could not add print_backward_graph key to dictionary");
        return TCL_ERROR;
    }

    if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("n_gradient_accumulation", -1), Tcl_NewIntObj(opt_params.n_gradient_accumulation))) {
        Tcl_DecrRefCount(dict_ptr);
        SetResult("could not add n_gradient_accumulation key to dictionary");
        return TCL_ERROR;
    }

    switch (opt_type) {
        case GGML_OPT_ADAM: {
            Tcl_Obj *adam_dict_ptr = Tcl_NewDictObj();
            Tcl_IncrRefCount(adam_dict_ptr);

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("n_iter", -1), Tcl_NewIntObj(opt_params.adam.n_iter))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add n_iter key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("sched", -1), ml_NewFloatStringObj(opt_params.adam.sched))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add sched key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("decay", -1), ml_NewFloatStringObj(opt_params.adam.decay))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add decay key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("decay_min_ndim", -1), Tcl_NewIntObj(opt_params.adam.decay_min_ndim))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add decay_min_ndim key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("alpha", -1), ml_NewFloatStringObj(opt_params.adam.alpha))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add alpha key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("beta1", -1), ml_NewFloatStringObj(opt_params.adam.beta1))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add beta1 key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("beta2", -1), ml_NewFloatStringObj(opt_params.adam.beta2))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add beta2 key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("eps", -1), ml_NewFloatStringObj(opt_params.adam.eps))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add eps key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("eps_f", -1), ml_NewFloatStringObj(opt_params.adam.eps_f))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add eps_f key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("eps_g", -1), ml_NewFloatStringObj(opt_params.adam.eps_g))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add eps_g key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, adam_dict_ptr, Tcl_NewStringObj("gclip", -1), ml_NewFloatStringObj(opt_params.adam.gclip))) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add gclip key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("adam", -1), adam_dict_ptr)) {
                Tcl_DecrRefCount(adam_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add adam key to dictionary");
                return TCL_ERROR;
            }

            Tcl_DecrRefCount(adam_dict_ptr);
        }
        break;
        case GGML_OPT_LBFGS: {
            Tcl_Obj *lbfgs_dict_ptr = Tcl_NewDictObj();
            Tcl_IncrRefCount(lbfgs_dict_ptr);

            if (TCL_OK != Tcl_DictObjPut(interp, lbfgs_dict_ptr, Tcl_NewStringObj("m", -1), Tcl_NewIntObj(opt_params.lbfgs.m))) {
                Tcl_DecrRefCount(lbfgs_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add m key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, lbfgs_dict_ptr, Tcl_NewStringObj("n_iter", -1), Tcl_NewIntObj(opt_params.lbfgs.n_iter))) {
                Tcl_DecrRefCount(lbfgs_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add n_iter key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, lbfgs_dict_ptr, Tcl_NewStringObj("max_linesearch", -1), Tcl_NewIntObj(opt_params.lbfgs.max_linesearch))) {
                Tcl_DecrRefCount(lbfgs_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add max_linesearch key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, lbfgs_dict_ptr, Tcl_NewStringObj("eps", -1), ml_NewFloatStringObj(opt_params.lbfgs.eps))) {
                Tcl_DecrRefCount(lbfgs_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add eps key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, lbfgs_dict_ptr, Tcl_NewStringObj("ftol", -1), ml_NewFloatStringObj(opt_params.lbfgs.ftol))) {
                Tcl_DecrRefCount(lbfgs_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add ftol key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, lbfgs_dict_ptr, Tcl_NewStringObj("wolfe", -1), ml_NewFloatStringObj(opt_params.lbfgs.wolfe))) {
                Tcl_DecrRefCount(lbfgs_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add wolfe key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, lbfgs_dict_ptr, Tcl_NewStringObj("min_step", -1), ml_NewFloatStringObj(opt_params.lbfgs.min_step))) {
                Tcl_DecrRefCount(lbfgs_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add min_step key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, lbfgs_dict_ptr, Tcl_NewStringObj("max_step", -1), ml_NewFloatStringObj(opt_params.lbfgs.max_step))) {
                Tcl_DecrRefCount(lbfgs_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add max_step key to dictionary");
                return TCL_ERROR;
            }

            if (TCL_OK != Tcl_DictObjPut(interp, dict_ptr, Tcl_NewStringObj("lbfgs", -1), lbfgs_dict_ptr)) {
                Tcl_DecrRefCount(lbfgs_dict_ptr);
                Tcl_DecrRefCount(dict_ptr);
                SetResult("could not add lbfgs key to dictionary");
                return TCL_ERROR;
            }

            Tcl_DecrRefCount(lbfgs_dict_ptr);

        }
        break;
        default:
            SetResult("unknown opt_type");
            Tcl_DecrRefCount(dict_ptr);
            return TCL_ERROR;
    }

    Tcl_SetObjResult(interp, dict_ptr);
    Tcl_DecrRefCount(dict_ptr);
    return TCL_OK;
}

static int ml_GetOptParamsFromDict(Tcl_Interp *interp, Tcl_Obj *dict_ptr, struct ggml_opt_params *opt_params) {

    Tcl_Obj *type_key_ptr = Tcl_NewStringObj("type", -1);
    Tcl_IncrRefCount(type_key_ptr);
    Tcl_Obj *type_ptr = NULL;
    if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, type_key_ptr, &type_ptr)) {
        Tcl_DecrRefCount(type_key_ptr);
        SetResult("type key not found");
        return TCL_ERROR;
    }
    Tcl_DecrRefCount(type_key_ptr);

    opt_params->type = ml_GetOptType(interp, type_ptr);

    // read the following from "dict_ptr":
    // - size_t graph_size
    // - int n_threads
    // - int past
    // - float delta
    // - int max_no_improvement
    // - bool print_forward_graph
    // - bool print_backward_graph
    // - int n_gradient_accumulation

    Tcl_Obj *graph_size_key_ptr = Tcl_NewStringObj("graph_size", -1);
    Tcl_IncrRefCount(graph_size_key_ptr);
    Tcl_Obj *graph_size_ptr = NULL;
    if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, graph_size_key_ptr, &graph_size_ptr)) {
        Tcl_DecrRefCount(graph_size_key_ptr);
        SetResult("graph_size key not found");
        return TCL_ERROR;
    }
    Tcl_DecrRefCount(graph_size_key_ptr);

    if (TCL_OK != Tcl_GetLongFromObj(interp, graph_size_ptr, &opt_params->graph_size) || opt_params->graph_size <= 0) {
        SetResult("graph_size is not an integer > 0");
        return TCL_ERROR;
    }

    Tcl_Obj *n_threads_key_ptr = Tcl_NewStringObj("n_threads", -1);
    Tcl_IncrRefCount(n_threads_key_ptr);
    Tcl_Obj *n_threads_ptr = NULL;
    if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, n_threads_key_ptr, &n_threads_ptr)) {
        Tcl_DecrRefCount(n_threads_key_ptr);
        SetResult("n_threads key not found");
        return TCL_ERROR;
    }
    Tcl_DecrRefCount(n_threads_key_ptr);

    if (TCL_OK != Tcl_GetIntFromObj(interp, n_threads_ptr, &opt_params->n_threads) || opt_params->n_threads <= 0) {
        SetResult("n_threads is not an integer > 0");
        return TCL_ERROR;
    }

    Tcl_Obj *past_key_ptr = Tcl_NewStringObj("past", -1);
    Tcl_IncrRefCount(past_key_ptr);
    Tcl_Obj *past_ptr = NULL;
    if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, past_key_ptr, &past_ptr)) {
        Tcl_DecrRefCount(past_key_ptr);
        SetResult("past key not found");
        return TCL_ERROR;
    }
    Tcl_DecrRefCount(past_key_ptr);

    if (TCL_OK != Tcl_GetIntFromObj(interp, past_ptr, &opt_params->past)) {
        SetResult("past is not an integer");
        return TCL_ERROR;
    }

    Tcl_Obj *delta_key_ptr = Tcl_NewStringObj("delta", -1);
    Tcl_IncrRefCount(delta_key_ptr);
    Tcl_Obj *delta_ptr = NULL;
    if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, delta_key_ptr, &delta_ptr)) {
        Tcl_DecrRefCount(delta_key_ptr);
        SetResult("delta key not found");
        return TCL_ERROR;
    }
    Tcl_DecrRefCount(delta_key_ptr);

    if (TCL_OK != ml_GetFloatFromObj(interp, delta_ptr, &opt_params->delta)) {
        SetResult("delta is not a double > 0");
        return TCL_ERROR;
    }

    Tcl_Obj *max_no_improvement_key_ptr = Tcl_NewStringObj("max_no_improvement", -1);
    Tcl_IncrRefCount(max_no_improvement_key_ptr);
    Tcl_Obj *max_no_improvement_ptr = NULL;
    if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, max_no_improvement_key_ptr, &max_no_improvement_ptr)) {
        Tcl_DecrRefCount(max_no_improvement_key_ptr);
        SetResult("max_no_improvement key not found");
        return TCL_ERROR;
    }
    Tcl_DecrRefCount(max_no_improvement_key_ptr);

    if (TCL_OK != Tcl_GetIntFromObj(interp, max_no_improvement_ptr, &opt_params->max_no_improvement) || opt_params->max_no_improvement <= 0) {
        SetResult("max_no_improvement is not an integer > 0");
        return TCL_ERROR;
    }

    Tcl_Obj *print_forward_graph_key_ptr = Tcl_NewStringObj("print_forward_graph", -1);
    Tcl_IncrRefCount(print_forward_graph_key_ptr);
    Tcl_Obj *print_forward_graph_ptr = NULL;
    if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, print_forward_graph_key_ptr, &print_forward_graph_ptr)) {
        Tcl_DecrRefCount(print_forward_graph_key_ptr);
        SetResult("print_forward_graph key not found");
        return TCL_ERROR;
    }
    Tcl_DecrRefCount(print_forward_graph_key_ptr);

    if (TCL_OK != Tcl_GetBooleanFromObj(interp, print_forward_graph_ptr, &opt_params->print_forward_graph)) {
        SetResult("print_forward_graph is not a boolean");
        return TCL_ERROR;
    }

    Tcl_Obj *print_backward_graph_key_ptr = Tcl_NewStringObj("print_backward_graph", -1);
    Tcl_IncrRefCount(print_backward_graph_key_ptr);
    Tcl_Obj *print_backward_graph_ptr = NULL;
    if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, print_backward_graph_key_ptr, &print_backward_graph_ptr)) {
        Tcl_DecrRefCount(print_backward_graph_key_ptr);
        SetResult("print_backward_graph key not found");
        return TCL_ERROR;
    }
    Tcl_DecrRefCount(print_backward_graph_key_ptr);

    if (TCL_OK != Tcl_GetBooleanFromObj(interp, print_backward_graph_ptr, &opt_params->print_backward_graph)) {
        SetResult("print_backward_graph is not a boolean");
        return TCL_ERROR;
    }

    Tcl_Obj *n_gradient_accumulation_key_ptr = Tcl_NewStringObj("n_gradient_accumulation", -1);
    Tcl_IncrRefCount(n_gradient_accumulation_key_ptr);
    Tcl_Obj *n_gradient_accumulation_ptr = NULL;
    if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, n_gradient_accumulation_key_ptr, &n_gradient_accumulation_ptr)) {
        Tcl_DecrRefCount(n_gradient_accumulation_key_ptr);
        SetResult("n_gradient_accumulation key not found");
        return TCL_ERROR;
    }
    Tcl_DecrRefCount(n_gradient_accumulation_key_ptr);

    if (TCL_OK != Tcl_GetIntFromObj(interp, n_gradient_accumulation_ptr, &opt_params->n_gradient_accumulation) || opt_params->n_gradient_accumulation <= 0) {
        SetResult("n_gradient_accumulation is not an integer > 0");
        return TCL_ERROR;
    }

    switch (opt_params->type) {
        case GGML_OPT_ADAM: {
            
            Tcl_Obj *adam_key_ptr = Tcl_NewStringObj("adam", -1);
            Tcl_IncrRefCount(adam_key_ptr);
            Tcl_Obj *adam_dict_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, adam_key_ptr, &adam_dict_ptr)) {
                Tcl_DecrRefCount(adam_key_ptr);
                SetResult("adam key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(adam_key_ptr);
            
            // read the following from "adam_dict_ptr":
            // - int n_iter;
            // - float sched; // schedule multiplier (fixed, decay or warmup)
            // - float decay; // weight decay for AdamW, use 0.0f to disable
            // - int   decay_min_ndim; // minimum number of tensor dimension to apply weight decay
            // - float alpha; // learning rate
            // - float beta1;
            // - float beta2;
            // - float eps;   // epsilon for numerical stability
            // - float eps_f; // epsilon for convergence test
            // - float eps_g; // epsilon for convergence test
            // - float gclip; // gradient clipping

            Tcl_Obj *n_iter_key_ptr = Tcl_NewStringObj("n_iter", -1);
            Tcl_IncrRefCount(n_iter_key_ptr);
            Tcl_Obj *n_iter_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, n_iter_key_ptr, &n_iter_ptr)) {
                Tcl_DecrRefCount(n_iter_key_ptr);
                SetResult("n_iter key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(n_iter_key_ptr);

            if (TCL_OK != Tcl_GetIntFromObj(interp, n_iter_ptr, &opt_params->adam.n_iter) || opt_params->adam.n_iter <= 0) {
                SetResult("n_iter is not an integer > 0");
                return TCL_ERROR;
            }

            Tcl_Obj *sched_key_ptr = Tcl_NewStringObj("sched", -1);
            Tcl_IncrRefCount(sched_key_ptr);
            Tcl_Obj *sched_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, sched_key_ptr, &sched_ptr)) {
                Tcl_DecrRefCount(sched_key_ptr);
                SetResult("sched key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(sched_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, sched_ptr, &opt_params->adam.sched)) {
                SetResult("sched is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *decay_key_ptr = Tcl_NewStringObj("decay", -1);
            Tcl_IncrRefCount(decay_key_ptr);
            Tcl_Obj *decay_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, decay_key_ptr, &decay_ptr)) {
                Tcl_DecrRefCount(decay_key_ptr);
                SetResult("decay key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(decay_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, decay_ptr, &opt_params->adam.decay)) {
                SetResult("decay is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *decay_min_ndim_key_ptr = Tcl_NewStringObj("decay_min_ndim", -1);
            Tcl_IncrRefCount(decay_min_ndim_key_ptr);
            Tcl_Obj *decay_min_ndim_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, decay_min_ndim_key_ptr, &decay_min_ndim_ptr)) {
                Tcl_DecrRefCount(decay_min_ndim_key_ptr);
                SetResult("decay_min_ndim key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(decay_min_ndim_key_ptr);

            if (TCL_OK != Tcl_GetIntFromObj(interp, decay_min_ndim_ptr, &opt_params->adam.decay_min_ndim) || opt_params->adam.decay_min_ndim <= 0) {
                SetResult("decay_min_ndim is not an integer > 0");
                return TCL_ERROR;
            }

            Tcl_Obj *alpha_key_ptr = Tcl_NewStringObj("alpha", -1);
            Tcl_IncrRefCount(alpha_key_ptr);
            Tcl_Obj *alpha_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, alpha_key_ptr, &alpha_ptr)) {
                Tcl_DecrRefCount(alpha_key_ptr);
                SetResult("alpha key not found");
                return TCL_ERROR;
            }

            if (TCL_OK != ml_GetFloatFromObj(interp, alpha_ptr, &opt_params->adam.alpha)) {
                SetResult("alpha is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *beta1_key_ptr = Tcl_NewStringObj("beta1", -1);
            Tcl_IncrRefCount(beta1_key_ptr);
            Tcl_Obj *beta1_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, beta1_key_ptr, &beta1_ptr)) {
                Tcl_DecrRefCount(beta1_key_ptr);
                SetResult("beta1 key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(beta1_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, beta1_ptr, &opt_params->adam.beta1)) {
                SetResult("beta1 is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *beta2_key_ptr = Tcl_NewStringObj("beta2", -1);
            Tcl_IncrRefCount(beta2_key_ptr);
            Tcl_Obj *beta2_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, beta2_key_ptr, &beta2_ptr)) {
                Tcl_DecrRefCount(beta2_key_ptr);
                SetResult("beta2 key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(beta2_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, beta2_ptr, &opt_params->adam.beta2)) {
                SetResult("beta2 is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *eps_key_ptr = Tcl_NewStringObj("eps", -1);
            Tcl_IncrRefCount(eps_key_ptr);
            Tcl_Obj *eps_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, eps_key_ptr, &eps_ptr)) {
                Tcl_DecrRefCount(eps_key_ptr);
                SetResult("eps key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(eps_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, eps_ptr, &opt_params->adam.eps)) {
                SetResult("eps is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *eps_f_key_ptr = Tcl_NewStringObj("eps_f", -1);
            Tcl_IncrRefCount(eps_f_key_ptr);
            Tcl_Obj *eps_f_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, eps_f_key_ptr, &eps_f_ptr)) {
                Tcl_DecrRefCount(eps_f_key_ptr);
                SetResult("eps_f key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(eps_f_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, eps_f_ptr, &opt_params->adam.eps_f)) {
                SetResult("eps_f is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *eps_g_key_ptr = Tcl_NewStringObj("eps_g", -1);
            Tcl_IncrRefCount(eps_g_key_ptr);
            Tcl_Obj *eps_g_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, eps_g_key_ptr, &eps_g_ptr)) {
                Tcl_DecrRefCount(eps_g_key_ptr);
                SetResult("eps_g key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(eps_g_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, eps_g_ptr, &opt_params->adam.eps_g)) {
                SetResult("eps_g is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *gclip_key_ptr = Tcl_NewStringObj("gclip", -1);
            Tcl_IncrRefCount(gclip_key_ptr);
            Tcl_Obj *gclip_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, adam_dict_ptr, gclip_key_ptr, &gclip_ptr)) {
                Tcl_DecrRefCount(gclip_key_ptr);
                SetResult("gclip key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(gclip_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, gclip_ptr, &opt_params->adam.gclip)) {
                SetResult("gclip is not a double");
                return TCL_ERROR;
            }
        }
            break;
        case GGML_OPT_LBFGS: {
            
            Tcl_Obj *lbfgs_key_ptr = Tcl_NewStringObj("lbfgs", -1);
            Tcl_IncrRefCount(lbfgs_key_ptr);
            Tcl_Obj *lbfgs_dict_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, dict_ptr, lbfgs_key_ptr, &lbfgs_dict_ptr)) {
                Tcl_DecrRefCount(lbfgs_key_ptr);
                SetResult("lbfgs key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(lbfgs_key_ptr);
            
            // read the following from "lbfgs_dict_ptr":
            // - int m; // number of corrections to approximate the inv. Hessian
            // - int n_iter;
            // - int max_linesearch;
            // - float eps;      // convergence tolerance
            // - float ftol;     // line search tolerance
            // - float wolfe;
            // - float min_step;
            // - float max_step;
            // - enum ggml_linesearch linesearch;

            Tcl_Obj *m_key_ptr = Tcl_NewStringObj("m", -1);
            Tcl_IncrRefCount(m_key_ptr);
            Tcl_Obj *m_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, lbfgs_key_ptr, m_key_ptr, &m_ptr)) {
                Tcl_DecrRefCount(m_key_ptr);
                SetResult("m key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(m_key_ptr);

            if (TCL_OK != Tcl_GetIntFromObj(interp, m_ptr, &opt_params->lbfgs.m) || opt_params->lbfgs.m <= 0) {
                SetResult("m is not an integer > 0");
                return TCL_ERROR;
            }

            Tcl_Obj *n_iter_key_ptr = Tcl_NewStringObj("n_iter", -1);
            Tcl_IncrRefCount(n_iter_key_ptr);
            Tcl_Obj *n_iter_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, lbfgs_key_ptr, n_iter_key_ptr, &n_iter_ptr)) {
                Tcl_DecrRefCount(n_iter_key_ptr);
                SetResult("n_iter key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(n_iter_key_ptr);

            if (TCL_OK != Tcl_GetIntFromObj(interp, n_iter_ptr, &opt_params->lbfgs.n_iter) || opt_params->lbfgs.n_iter <= 0) {
                SetResult("n_iter is not an integer > 0");
                return TCL_ERROR;
            }

            Tcl_Obj *max_linesearch_key_ptr = Tcl_NewStringObj("max_linesearch", -1);
            Tcl_IncrRefCount(max_linesearch_key_ptr);
            Tcl_Obj *max_linesearch_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, lbfgs_key_ptr, max_linesearch_key_ptr, &max_linesearch_ptr)) {
                Tcl_DecrRefCount(max_linesearch_key_ptr);
                SetResult("max_linesearch key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(max_linesearch_key_ptr);

            if (TCL_OK != Tcl_GetIntFromObj(interp, max_linesearch_ptr, &opt_params->lbfgs.max_linesearch) || opt_params->lbfgs.max_linesearch <= 0) {
                SetResult("max_linesearch is not an integer > 0");
                return TCL_ERROR;
            }

            Tcl_Obj *eps_key_ptr = Tcl_NewStringObj("eps", -1);
            Tcl_IncrRefCount(eps_key_ptr);
            Tcl_Obj *eps_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, lbfgs_key_ptr, eps_key_ptr, &eps_ptr)) {
                Tcl_DecrRefCount(eps_key_ptr);
                SetResult("eps key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(eps_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, eps_ptr, &opt_params->lbfgs.eps)) {
                SetResult("eps is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *ftol_key_ptr = Tcl_NewStringObj("ftol", -1);
            Tcl_IncrRefCount(ftol_key_ptr);
            Tcl_Obj *ftol_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, lbfgs_key_ptr, ftol_key_ptr, &ftol_ptr)) {
                Tcl_DecrRefCount(ftol_key_ptr);
                SetResult("ftol key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(ftol_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, ftol_ptr, &opt_params->lbfgs.ftol)) {
                SetResult("ftol is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *wolfe_key_ptr = Tcl_NewStringObj("wolfe", -1);
            Tcl_IncrRefCount(wolfe_key_ptr);
            Tcl_Obj *wolfe_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, lbfgs_key_ptr, wolfe_key_ptr, &wolfe_ptr)) {
                Tcl_DecrRefCount(wolfe_key_ptr);
                SetResult("wolfe key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(wolfe_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, wolfe_ptr, &opt_params->lbfgs.wolfe)) {
                SetResult("wolfe is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *min_step_key_ptr = Tcl_NewStringObj("min_step", -1);
            Tcl_IncrRefCount(min_step_key_ptr);
            Tcl_Obj *min_step_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, lbfgs_key_ptr, min_step_key_ptr, &min_step_ptr)) {
                Tcl_DecrRefCount(min_step_key_ptr);
                SetResult("min_step key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(min_step_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, min_step_ptr, &opt_params->lbfgs.min_step)) {
                SetResult("min_step is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *max_step_key_ptr = Tcl_NewStringObj("max_step", -1);
            Tcl_IncrRefCount(max_step_key_ptr);
            Tcl_Obj *max_step_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, lbfgs_key_ptr, max_step_key_ptr, &max_step_ptr)) {
                Tcl_DecrRefCount(max_step_key_ptr);
                SetResult("max_step key not found");
                return TCL_ERROR;
            }
            Tcl_DecrRefCount(max_step_key_ptr);

            if (TCL_OK != ml_GetFloatFromObj(interp, max_step_ptr, &opt_params->lbfgs.max_step)) {
                SetResult("max_step is not a double");
                return TCL_ERROR;
            }

            Tcl_Obj *linesearch_key_ptr = Tcl_NewStringObj("linesearch", -1);
            Tcl_IncrRefCount(linesearch_key_ptr);
            Tcl_Obj *linesearch_ptr = NULL;
            if (TCL_OK != Tcl_DictObjGet(interp, lbfgs_key_ptr, linesearch_key_ptr, &linesearch_ptr)) {
                Tcl_DecrRefCount(linesearch_key_ptr);
                SetResult("linesearch key not found");
                return TCL_ERROR;
            }

            opt_params->lbfgs.linesearch = ml_GetLinesearchMethod(interp, linesearch_ptr);
        }
            break;
        default:
            SetResult("unknown opt type");
            return TCL_ERROR;
    }

    return TCL_OK;
}

int ml_OptCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    DBG(fprintf(stderr, "OptCmd\n"));
    CheckArgs(4, 4, 1, "context_handle opt_params_dict tensor_handle");

    const char *context_handle = Tcl_GetString(objv[1]);
    ml_context_t *ctx = ml_GetInternalFromContext(context_handle);
    if (!ctx) {
        SetResult("context handle not found");
        return TCL_ERROR;
    }

    const char *tensor_handle = Tcl_GetString(objv[3]);
    ml_tensor_t *tensor_ptr = ml_GetInternalFromTensor(tensor_handle);
    if (!tensor_ptr) {
        SetResult("tensor handle not found");
        return TCL_ERROR;
    }

    struct ggml_opt_params opt_params;

    if (TCL_OK != ml_GetOptParamsFromDict(interp, objv[2], &opt_params)) {
        return TCL_ERROR;
    }

    ggml_opt(ctx->ggml_ctx, opt_params, tensor_ptr->ggml_tensor);

    return TCL_OK;
}