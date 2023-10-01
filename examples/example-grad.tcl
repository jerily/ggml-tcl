proc check_gradient {op_name ctx0 x f ndims nargs eps max_error_abs max_error_rel {nthreads 10}} {
    set gf [::ggml::build_forward_ctx $ctx0 $f]
    set gb [::ggml::build_backward_ctx $ctx0 $gf false]
    ::ggml::graph_compute $gf $nthreads
    ::ggml::graph_reset $gf
    ::ggml::set_f32 [::ggml::get_grad $f] 1.0
    ::ggml::graph_compute $gb $nthreads

}