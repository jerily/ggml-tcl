package require ggml

set ctx [::ggml::create_context [dict create]]
puts "ctx: $ctx"
::ggml::destroy_context $ctx