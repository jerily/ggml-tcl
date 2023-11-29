package require ggml

proc get_random_tensor_f32 {ctx0 n_dims ne_lst fmin fmax} {
    set result [::ggml::new_tensor $ctx0 F32 $n_dims $ne_lst]

    switch $n_dims {
        1 {
            for {set i0 0} {$i0 < [lindex $ne_lst 0]} {incr i0} {
                ::ggml::set_f32_1d $result $i0 [expr { $fmin + rand() * ($fmax - $fmin) }]
            }
        }
        2 {
            for {set i1 0} {$i1 < [lindex $ne_lst 1]} {incr i1} {
                for {set i0 0} {$i0 < [lindex $ne_lst 0]} {incr i0} {
                    set ne0 [lindex $ne_lst 0]
                    set i [expr { $i1 * $ne0 + $i0 }]
                    ::ggml::set_f32_1d $result $i [expr { $fmin + rand() * ($fmax - $fmin) }]
                }
            }
        }
        3 {
            for {set i2 0} {$i2 < [lindex $ne_lst 2]} {incr i2} {
                for {set i1 0} {$i1 < [lindex $ne_lst 1]} {incr i1} {
                    for {set i0 0} {$i0 < [lindex $ne_lst 0]} {incr i0} {
                        set ne0 [lindex $ne_lst 0]
                        set ne1 [lindex $ne_lst 1]
                        set i [expr { $i2 * $ne1 * $ne0 + $i1 * $ne0 + $i0 }]
                        ::ggml::set_f32_1d $result $i [expr { $fmin + rand() * ($fmax - $fmin) }]
                    }
                }
            }
        }
        4 {
            for {set i3 0} {$i3 < [lindex $ne_lst 3]} {incr i3} {
                for {set i2 0} {$i2 < [lindex $ne_lst 2]} {incr i2} {
                    for {set i1 0} {$i1 < [lindex $ne_lst 1]} {incr i1} {
                        for {set i0 0} {$i0 < [lindex $ne_lst 0]} {incr i0} {
                            set ne0 [lindex $ne_lst 0]
                            set ne1 [lindex $ne_lst 1]
                            set ne2 [lindex $ne_lst 2]
                            set i [expr { $i3 * $ne2 * $ne1 * $ne0 + $i2 * $ne1 * $ne0 + $i1 * $ne0 + $i0 }]
                            ::ggml::set_f32_1d $result $i [expr { $fmin + rand() * ($fmax - $fmin) }]
                        }
                    }
                }
            }
        }
    }
    return $result
}

# create context
set mem_size [expr { 1024*1024*1024 }]
set ctx [::ggml::create_context $mem_size]

set ne1_lst [list 4 128 1 1]
set ne2_lst [list 4 256 1 1]
set ne3_lst [list 128 256 1 1]

set a [get_random_tensor_f32 $ctx 2 $ne1_lst -1 +1]
set b [get_random_tensor_f32 $ctx 2 $ne2_lst -1 +1]
::ggml::set_param $ctx $a
::ggml::set_param $ctx $b

set c [get_random_tensor_f32 $ctx 2 $ne3_lst -1 +1]

set ab [::ggml::mul_mat $ctx $a $b]
set d [::ggml::sub $ctx $c $ab]
set e [::ggml::sum $ctx [::ggml::sqr $ctx $d]]

set ge [::ggml::new_graph_custom $ctx true]
::ggml::build_forward_expand $ge $e
::ggml::graph_reset $ge

::ggml::graph_compute $ge 1

set fe [::ggml::get_f32_1d $e 0]
puts "fe = $fe"

set opt_params [::ggml::opt_default_params ADAM]
puts "opt_params = $opt_params"
::ggml::opt $ctx $opt_params $e
::ggml::graph_reset $ge
::ggml::graph_compute $ge 1

set fe_opt [::ggml::get_f32_1d $e 0]
puts "original e = $fe, optimized e = $fe_opt"

::ggml::destroy_context $ctx

