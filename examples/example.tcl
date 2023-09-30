package require ggml

set ctx [::ggml::create_context [dict create]]
puts "ctx: $ctx"
set x [::ggml::new_tensor_1d $ctx F32 1]
puts "x: $x"
::ggml::destroy_context $ctx