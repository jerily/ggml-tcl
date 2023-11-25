package require ggml

# create context
set mem_size [expr { 16*1024*1024 }]
set ctx [::ggml::create_context $mem_size]
puts "ctx: $ctx"

set x [::ggml::new_tensor_1d $ctx F32 1]

# x is an input variable
::ggml::set_param $ctx $x

set a [::ggml::new_tensor_1d $ctx F32 1]
set b [::ggml::new_tensor_1d $ctx F32 1]
set x2 [::ggml::mul $ctx $x $x]
set f [::ggml::add $ctx [::ggml::mul $ctx $a $x2] $b]

# build forward computational graph
set gf [::ggml::build_forward_expand $ctx $f]
puts "gf: $gf"

# set the input variable and parameter values
::ggml::set_f32 $x 2.0
::ggml::set_f32 $a 3.0
::ggml::set_f32 $b 4.0

# compute
set nthreads 10
::ggml::graph_compute $gf $nthreads

# get result
puts "f = [::ggml::get_f32_1d $f 0]"

# destroy context
::ggml::destroy_context $ctx