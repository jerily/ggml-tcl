# ggml-tcl

TCL bindings for [ggml](https://github.com/ggerganov/ggml),
a tensor library for machine learning

**Note that this project is under active development.**

## Build Dependency

Download latest stable version on Linux:
```bash
git clone https://github.com/ggerganov/ggml.git
cd ggml
mkdir build
cd build
cmake .. \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BUILD_TESTS=OFF \
  -DGGML_BUILD_EXAMPLES=OFF
make
make install
```

## Installation

```bash
git clone https://github.com/jerily/ggml-tcl.git
cd ggml-tcl
mkdir build
cd build
# change "TCL_LIBRARY_DIR" and "TCL_INCLUDE_DIR" to the correct paths
cmake .. \
  -DTCL_LIBRARY_DIR=/usr/local/lib \
  -DTCL_INCLUDE_DIR=/usr/local/include
make
make install
```

## TCL Commands

```
::ggml::create_context mem_size
::ggml::destroy_context context_handle
::ggml::load_context_from_file filename
::ggml::used_mem context_handle
::ggml::get_max_tensor_size context_handle
::ggml::get_mem_size context_handle

::ggml::build_forward_expand context_handle tensor_handle
::ggml::build_backward_expand context_handle forward_cgraph_handle keep_gradient_graph

::ggml::graph_compute cgraph_handle nthreads
::ggml::graph_reset cgraph_handle
::ggml::graph_dump_dot gb_handle fg_handle filename

::ggml::set_param
::ggml::get_grad
::ggml::nelements
::ggml::new_tensor
::ggml::new_tensor_1d
::ggml::new_tensor_2d
::ggml::new_tensor_3d
::ggml::new_tensor_4d
::ggml::new_i32
::ggml::new_f32
::ggml::dup_tensor
::ggml::view_tensor
::ggml::set_zero
::ggml::set_i32
::ggml::set_f32
::ggml::get_i32_1d
::ggml::set_i32_1d
::ggml::get_f32_1d
::ggml::set_f32_1d
::ggml::dup
::ggml::dup_inplace
::ggml::add
::ggml::add_inplace
::ggml::add1
::ggml::add1_inplace
::ggml::sub
::ggml::sub_inplace
::ggml::mul
::ggml::mul_inplace
::ggml::div
::ggml::div_inplace
::ggml::sqr
::ggml::sqr_inplace
::ggml::sqrt
::ggml::sqrt_inplace
::ggml::log
::ggml::log_inplace
::ggml::sum
::ggml::sum_rows
::ggml::mean
::ggml::argmax
::ggml::repeat
::ggml::repeat_back
::ggml::concat
::ggml::abs
::ggml::sgn
::ggml::sgn_inplace
::ggml::neg
::ggml::neg_inplace
::ggml::step
::ggml::step_inplace
::ggml::tanh
::ggml::tanh_inplace
::ggml::elu
::ggml::elu_inplace
::ggml::relu
::ggml::relu_inplace
::ggml::gelu
::ggml::gelu_inplace
::ggml::gelu_quick
::ggml::gelu_quick_inplace
::ggml::silu
::ggml::silu_inplace
::ggml::silu_back
::ggml::norm
::ggml::norm_inplace
::ggml::rms_norm
::ggml::rms_norm_inplace
::ggml::group_norm
::ggml::group_norm_inplace
::ggml::rms_norm_back
::ggml::mul_mat
::ggml::out_prod
::ggml::scale
::ggml::scale_inplace
::ggml::set
::ggml::set_inplace
::ggml::set_1d
::ggml::set_1d_inplace
::ggml::set_2d
::ggml::set_2d_inplace
::ggml::cpy
::ggml::cpy_inplace
::ggml::cont
::ggml::cont_inplace
::ggml::reshape
::ggml::reshape_1d
::ggml::reshape_2d
::ggml::reshape_3d
::ggml::reshape_4d
::ggml::view_1d
::ggml::view_2d
::ggml::view_3d
::ggml::view_4d
::ggml::permute
::ggml::transpose
::ggml::get_rows
::ggml::get_rows_back
::ggml::diag
::ggml::diag_mask_inf
::ggml::diag_mask_inf_inplace
::ggml::diag_mask_zero
::ggml::diag_mask_zero_inplace
::ggml::soft_max
::ggml::soft_max_inplace
::ggml::soft_max_back
::ggml::soft_max_back_inplace
::ggml::rope
::ggml::rope_inplace
::ggml::rope_custom
::ggml::rope_custom_inplace
::ggml::rope_xpos_inplace
::ggml::rope_back
::ggml::alibi
::ggml::clamp
::ggml::conv_1d
::ggml::conv_1d_ph
::ggml::conv_transpose_1d
::ggml::conv_2d
::ggml::conv_2d_sk_p0
::ggml::conv_2d_s1_ph
::ggml::conv_transpose_2d_p0
::ggml::pool_1d
::ggml::pool_2d
::ggml::upscale
::ggml::flash_attn
::ggml::flash_attn_back
::ggml::flash_ff
::ggml::win_part
::ggml::win_unpart
::ggml::unary
::ggml::unary_inplace
::ggml::cross_entropy_loss
::ggml::cross_entropy_loss_back
::ggml::get_rel_pos
::ggml::add_rel_pos
::ggml::add_rel_pos_inplace
```