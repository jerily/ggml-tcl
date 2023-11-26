package require ggml

if { [llength $argv] != 1 } {
    puts "Usage: $argv0 <filename.gguf>"
    exit 1
}

set filename [lindex $argv 0]
set ctx [::ggml::load_context_from_file $filename]
puts ctx=$ctx
puts used_mem=[::ggml::used_mem $ctx]
puts max_tensor_size=[::ggml::get_max_tensor_size $ctx]
puts mem_size=[::ggml::get_mem_size $ctx]
::ggml::destroy_context $ctx