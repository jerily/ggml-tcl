package require ggml

set filename "/home/phi/data/llama-2-7b.Q4_0.gguf"
set ctx [::ggml::load_context_from_file $filename]
puts ctx=$ctx
::ggml::destroy_context $ctx