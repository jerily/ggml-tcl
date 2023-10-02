package require ggml

set filename "/home/phi/data/llama-2-7b.Q4_0.gguf"
set ctx [::ggml::load_context_from_file $filename]
puts ctx=$ctx
puts used_mem=[::ggml::used_mem $ctx]
puts max_tensor_size=[::ggml::get_max_tensor_size $ctx]
puts mem_size=[::ggml::get_mem_size $ctx]
::ggml::destroy_context $ctx