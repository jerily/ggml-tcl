package require ggml

# create context
set ctx [::ggml::create_context [dict create]]
puts "ctx: $ctx"

set x [::ggml::new_tensor_1d $ctx F32 1]

# x is an input variable
::ggml::set_param $ctx $x

set a [::ggml::new_tensor_1d $ctx F32 1]
set b [::ggml::new_tensor_1d $ctx F32 1]
set x2 [::ggml::mul $ctx $x $x]
set f [::ggml::add $ctx [::ggml::mul $ctx $a $x2] $b]

# build forward computational graph
set cgraph [::ggml::build_forward_ctx $ctx $f]
puts "cgraph: $cgraph"

# set the input variable and parameter values
::ggml::set_f32 $x 2.0
::ggml::set_f32 $a 3.0
::ggml::set_f32 $b 4.0

# compute
set nthreads 10
::ggml::graph_compute $cgraph $nthreads

# get result
puts "f = [::ggml::get_f32_1d $f 0]"

# destroy context
::ggml::destroy_context $ctx