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

* **::ggml::create_context** *mem_size*
* **::ggml::destroy_context** *context_handle*
* **::ggml::load_context_from_file** *filename*
* **::ggml::used_mem** *context_handle*
* **::ggml::get_max_tensor_size** *context_handle*
* **::ggml::get_mem_size** *context_handle*

* **::ggml::build_forward_expand** *cgraph_handle* *tensor_handle*
* **::ggml::build_backward_expand** *context_handle* *forward_cgraph_handle* *backward_cgraph_handle* *keep_gradient_graph*

* **::ggml::new_graph** *context_handle*
* **::ggml::new_graph_custom** *context_handle* *grads* *?size?*
* **::ggml::graph_compute** *cgraph_handle* *nthreads*
* **::ggml::graph_reset** *cgraph_handle*
* **::ggml::graph_dump_dot** *gb_handle* *fg_handle* *output_filename*
* **::ggml::graph_cpy** *src_cgraph_handle* *dst_cgraph_handle*

* **::ggml::set_param** *context_handle* *tensor_handle*
* **::ggml::get_grad** *tensor_handle*
* **::ggml::nelements** *tensor_handle*
* **::ggml::new_tensor** *context_handle* *type* *ndims* *ne_list*
* **::ggml::new_tensor_1d** *context_handle* *type* *ne0*
* **::ggml::new_tensor_2d** *context_handle* *type* *ne0* *ne1*
* **::ggml::new_tensor_3d** *context_handle* *type* *ne0* *ne1* *ne2*
* **::ggml::new_tensor_4d** *context_handle* *type* *ne0* *ne1* *ne2* *ne3*
* **::ggml::new_i32** *context_handle* *int32_value*
* **::ggml::new_f32** *context_handle* *float32_value*
* **::ggml::dup_tensor** *context_handle* *tensor_handle*
* **::ggml::view_tensor** *context_handle* *tensor_handle*
* **::ggml::set_zero** *tensor_handle*
* **::ggml::set_i32** *tensor_handle* *int32_value*
* **::ggml::set_f32** *tensor_handle* *float32_value*
* **::ggml::get_i32_1d** *tensor_handle* *index*
* **::ggml::set_i32_1d** *tensor_handle* *index* *int32_value*
* **::ggml::get_f32_1d** *tensor_handle* *index*
* **::ggml::set_f32_1d** *tensor_handle* *index* *float32_value*
* **::ggml::dup** *context_handle* *tensor_handle*
* **::ggml::dup_inplace** *context_handle* *tensor_handle*
* **::ggml::add** *context_handle* *tensor_a* *tensor_b*
* **::ggml::add_inplace** *context_handle* *tensor_a* *tensor_b*
* **::ggml::add1** *context_handle* *tensor_a* *tensor_b*
* **::ggml::add1_inplace** *context_handle* *tensor_a* *tensor_b*
* **::ggml::sub** *context_handle* *tensor_a* *tensor_b*
* **::ggml::sub_inplace** *context_handle* *tensor_a* *tensor_b*
* **::ggml::mul** *context_handle* *tensor_a* *tensor_b*
* **::ggml::mul_inplace** *context_handle* *tensor_a* *tensor_b*
* **::ggml::div** *context_handle* *tensor_a* *tensor_b*
* **::ggml::div_inplace** *context_handle* *tensor_a* *tensor_b*
* **::ggml::sqr** *context_handle* *tensor_handle*
* **::ggml::sqr_inplace** *context_handle* *tensor_handle*
* **::ggml::sqrt** *context_handle* *tensor_handle*
* **::ggml::sqrt_inplace** *context_handle* *tensor_handle*
* **::ggml::log** *context_handle* *tensor_handle*
* **::ggml::log_inplace** *context_handle* *tensor_handle*
* **::ggml::sum** *context_handle* *tensor_handle*
* **::ggml::sum_rows** *context_handle* *tensor_handle*
* **::ggml::mean** *context_handle* *tensor_handle*
* **::ggml::argmax** *context_handle* *tensor_handle*
* **::ggml::repeat** *context_handle* *tensor_a* *tensor_b*
* **::ggml::repeat_back** *context_handle* *tensor_a* *tensor_b*
* **::ggml::concat** *context_handle* *tensor_a* *tensor_b*
* **::ggml::abs** *context_handle* *tensor_handle*
* **::ggml::sgn** *context_handle* *tensor_handle*
* **::ggml::sgn_inplace** *context_handle* *tensor_handle*
* **::ggml::neg** *context_handle* *tensor_handle*
* **::ggml::neg_inplace** *context_handle* *tensor_handle*
* **::ggml::step** *context_handle* *tensor_handle*
* **::ggml::step_inplace** *context_handle* *tensor_handle*
* **::ggml::tanh** *context_handle* *tensor_handle*
* **::ggml::tanh_inplace** *context_handle* *tensor_handle*
* **::ggml::elu** *context_handle* *tensor_handle*
* **::ggml::elu_inplace** *context_handle* *tensor_handle*
* **::ggml::relu** *context_handle* *tensor_handle*
* **::ggml::relu_inplace** *context_handle* *tensor_handle*
* **::ggml::gelu** *context_handle* *tensor_handle*
* **::ggml::gelu_inplace** *context_handle* *tensor_handle*
* **::ggml::gelu_quick** *context_handle* *tensor_handle*
* **::ggml::gelu_quick_inplace** *context_handle* *tensor_handle*
* **::ggml::silu** *context_handle* *tensor_handle*
* **::ggml::silu_inplace** *context_handle* *tensor_handle*
* **::ggml::silu_back** *context_handle* *tensor_handle*
* **::ggml::norm** *context_handle* *tensor_handle* *eps*
* **::ggml::norm_inplace** *context_handle* *tensor_handle* *eps*
* **::ggml::rms_norm** *context_handle* *tensor_handle* *eps*
* **::ggml::rms_norm_inplace** *context_handle* *tensor_handle* *eps*
* **::ggml::group_norm** *context_handle* *tensor_handle* *eps*
* **::ggml::group_norm_inplace** *context_handle* *tensor_handle* *eps*
* **::ggml::rms_norm_back** *context_handle* *tensor_a* *tensor_b* *eps*
* **::ggml::mul_mat** *context_handle* *tensor_a* *tensor_b*
* **::ggml::out_prod** *context_handle* *tensor_a* *tensor_b*
* **::ggml::scale** *context_handle* *tensor_a* *tensor_b*
* **::ggml::scale_inplace** *context_handle* *tensor_a* *tensor_b*
* **::ggml::set** *context_handle* *tensor_a* *tensor_b* *nb1* *nb2* *nb3* *offset*
* **::ggml::set_inplace** *context_handle* *tensor_a* *tensor_b* *nb1* *nb2* *nb3* *offset*
* **::ggml::set_1d** *context_handle* *tensor_a* *tensor_b* *offset*
* **::ggml::set_1d_inplace** *context_handle* *tensor_a* *tensor_b* *offset*
* **::ggml::set_2d** *context_handle* *tensor_a* *tensor_b* *nb1* *offset*
* **::ggml::set_2d_inplace** *context_handle* *tensor_a* *tensor_b* *nb1* *offset*
* **::ggml::cpy** *context_handle* *tensor_a* *tensor_b*
* **::ggml::cpy_inplace** *context_handle* *tensor_a* *tensor_b*
* **::ggml::cont** *context_handle* *tensor_handle*
* **::ggml::cont_inplace** *context_handle* *tensor_handle*
* **::ggml::reshape** *context_handle* *tensor_a* *tensor_b*
* **::ggml::reshape_1d** *context_handle* *tensor_handle* *ne0*
* **::ggml::reshape_2d** *context_handle* *tensor_handle* *ne0* *ne1*
* **::ggml::reshape_3d** *context_handle* *tensor_handle* *ne0* *ne1* *ne2*
* **::ggml::reshape_4d** *context_handle* *tensor_handle* *ne0* *ne1* *ne2* *ne3*
* **::ggml::view_1d** *context_handle* *tensor_handle* *ne0* *offset*
* **::ggml::view_2d** *context_handle* *tensor_handle* *ne0* *ne1* *nb1* *offset*
* **::ggml::view_3d** *context_handle* *tensor_handle* *ne0* *ne1* *ne2* *nb1* *nb2* *offset*
* **::ggml::view_4d** *context_handle* *tensor_handle* *ne0* *ne1* *ne2* *ne3* *nb1* *nb2* *nb3* *offset*
* **::ggml::permute** *context_handle* *tensor_handle* *axis0* *axis1* *axis2* *axis3*
* **::ggml::transpose** *context_handle* *tensor_handle*
* **::ggml::get_rows** *context_handle* *tensor_a* *tensor_b*
* **::ggml::get_rows_back** *context_handle* *tensor_a* *tensor_b* *tensor_c*
* **::ggml::diag** *context_handle* *tensor_handle*
* **::ggml::diag_mask_inf** *context_handle* *tensor_handle* *n_past*
* **::ggml::diag_mask_inf_inplace** *context_handle* *tensor_handle* *n_past*
* **::ggml::diag_mask_zero** *context_handle* *tensor_handle* *n_past*
* **::ggml::diag_mask_zero_inplace** *context_handle* *tensor_handle* *n_past*
* **::ggml::soft_max** *context_handle* *tensor_handle*
* **::ggml::soft_max_inplace** *context_handle* *tensor_handle*
* **::ggml::soft_max_back** *context_handle* *tensor_a* *tensor_b*
* **::ggml::soft_max_back_inplace** *context_handle* *tensor_a* *tensor_b*
* **::ggml::rope** *context_handle* *tensor_a_handle* *tensor_b_handle* *n_dims* *mode* *n_ctx*
* **::ggml::rope_inplace** *context_handle* *tensor_a_handle* *tensor_b_handle* *n_dims* *mode* *n_ctx*
* **::ggml::rope_custom** *context_handle* *tensor_a_handle* *tensor_b_handle* *n_dims* *mode* *n_ctx* *n_orig_ctx* *freq_base* *freq_scale* *ext_factor* *attn_factor* *beta_fast* *beta_slow*
* **::ggml::rope_custom_inplace** *context_handle* *tensor_a_handle* *tensor_b_handle* *n_dims* *mode* *n_ctx* *n_orig_ctx* *freq_base* *freq_scale* *ext_factor* *attn_factor* *beta_fast* *beta_slow*
* **::ggml::rope_xpos_inplace** *context_handle* *tensor_handle* *n_past* *n_dims* *base* *down*
* **::ggml::rope_back** *context_handle* *tensor_a_handle* *tensor_b_handle* *n_dims* *mode* *n_ctx* *n_orig_ctx* *freq_base* *freq_scale* *ext_factor* *attn_factor* *beta_fast* *beta_slow* *xpos_base* *xpos_down*
* **::ggml::alibi** *context_handle* *tensor_handle* *n_past* *n_head* *bias_max*
* **::ggml::clamp** *context_handle* *tensor_handle* *min* *max*
* **::ggml::conv_1d** *context_handle* *tensor_a* *tensor_b* *stride* *padding* *dilation*
* **::ggml::conv_1d_ph** *context_handle* *tensor_a* *tensor_b* *stride* *dilation*
* **::ggml::conv_transpose_1d** *context_handle* *tensor_a* *tensor_b* *stride* *padding* *dilation*
* **::ggml::conv_2d** *context_handle* *tensor_a* *tensor_b* *s0* *s1* *p0* *p1* *d0* *d1*
* **::ggml::conv_2d_sk_p0** *context_handle* *tensor_a* *tensor_b*
* **::ggml::conv_2d_s1_ph** *context_handle* *tensor_a* *tensor_b*
* **::ggml::conv_transpose_2d_p0** *context_handle* *tensor_a* *tensor_b* *stride*
* **::ggml::pool_1d** *context_handle* *tensor_handle* *op_pool* *k0* *s0* *p0*
* **::ggml::pool_2d** *context_handle* *tensor_handle* *op_pool* *k0* *k1* *s0* *s1* *p0* *p1*
* **::ggml::upscale** *context_handle* *tensor_handle* *scale_factor*
* **::ggml::flash_attn** *context_handle* *tensor_q* *tensor_k* *tensor_v* *masked*
* **::ggml::flash_attn_back** *context_handle* *tensor_q* *tensor_k* *tensor_v* *tensor_d* *masked*
* **::ggml::flash_ff** *context_handle* *tensor_a* *tensor_b0* *tensor_b1* *tensor_c0* *tensor_c1*
* **::ggml::win_part** *context_handle* *tensor_handle* *w*
* **::ggml::win_unpart** *context_handle* *tensor_handle* *w0* *h0* *w*
* **::ggml::unary** *context_handle* *tensor_handle* *unary_op*
* **::ggml::unary_inplace** *context_handle* *tensor_handle* *unary_op*
* **::ggml::cross_entropy_loss** *context_handle* *tensor_a* *tensor_b*
* **::ggml::cross_entropy_loss_back** *context_handle* *tensor_a* *tensor_b* *tensor_c*
* **::ggml::get_rel_pos** *context_handle* *tensor_handle* *qh* *kh*
* **::ggml::add_rel_pos** *context_handle* *tensor_a* *tensor_pw* *tensor_ph*
* **::ggml::add_rel_pos_inplace** *context_handle* *tensor_a* *tensor_pw* *tensor_ph*
