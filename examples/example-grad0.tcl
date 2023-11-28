# https://github.com/ggerganov/llama.cpp/blob/master/tests/test-grad0.cpp

package require ggml

proc check_gradient {op_name ctx0 x f n_dims nargs eps max_error_abs max_error_rel {n_threads 10}} {
    set gf [::ggml::new_graph_custom $ctx0 true]
    set gb [::ggml::new_graph_custom $ctx0 true]
    ::ggml::build_forward_expand $gf $f
    ::ggml::graph_cpy $gf $gb
    ::ggml::build_backward_expand $ctx0 $gf $gb false


    ::ggml::graph_compute $gf $n_threads
    ::ggml::graph_reset $gf
    ::ggml::set_f32 [::ggml::get_grad $f] 1.0
    ::ggml::graph_compute $gb $n_threads

    #::ggml::graph_dump_dot $gf "" "example-grad0-forward.dot"
    #::ggml::graph_dump_dot $gb $gf "example-grad0-backward.dot"

    for {set i 0} {$i < $nargs} {incr i} {
        set nelements [::ggml::nelements [lindex $x $i]]

        for {set k 0} {$k < $nelements} {incr k} {
            # compute gradient using finite differences
            set x0 [::ggml::get_f32_1d [lindex $x $i] $k]
            set xm [expr { $x0 - $eps }]
            set xp [expr { $x0 + $eps }]

            ::ggml::set_f32_1d [lindex $x $i] $k $xp

            ::ggml::graph_compute $gf $n_threads

            set f0 [::ggml::get_f32_1d $f 0]

            ::ggml::set_f32_1d [lindex $x $i] $k $xm

            ::ggml::graph_compute $gf $n_threads

            set f1 [::ggml::get_f32_1d $f 0]
            set g0 [expr { ($f0 - $f1)/(2.0 * $eps) }]

            ::ggml::set_f32_1d [lindex $x $i] $k $x0

            # compute gradient using backward graph
            ::ggml::graph_reset $gf
            ::ggml::set_f32 [::ggml::get_grad $f] 1.0

            ::ggml::graph_compute $gb $n_threads

            set g1 [::ggml::get_f32_1d [::ggml::get_grad [lindex $x $i]] $k]

            set error_abs [expr { abs($g0 - $g1) }]
            set error_rel [expr { $g0 != 0 ? abs($g0 - $g1)/abs($g0) : 0 }]

            if { $error_abs > $max_error_abs || $error_rel > $max_error_rel } {
                puts "${op_name}: n_dims=${n_dims}, i=${i}, k=${k}, x0=${x0}, xm=${xm}, xp=${xp}, f0=${f0}, f1=${f1}, g0=${g0}, g1=${g1}, eps=${eps}, error_abs=${error_abs}, error_rel=${error_rel}"
                return false
            }
        }
    }

    return true
}

set ::MAX_INT 2147483647
proc ::tcl::mathfunc::irand {n} {
    if { $n == 0 } { return 0 }
    return [expr { int(rand() * ${::MAX_INT}) % $n }]
}

proc get_random_dims {n_dims} {
    set dims [list]
    for {set i 0} {$i < $n_dims} {incr i} {
        lappend dims [expr { 1 + irand(${n_dims}) }]
    }
    return $dims
}

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

proc main {argv0 argv} {

    if { [llength $argv] != 1 } {
        puts "usage: $argv0 <niter>"
        exit 1
    }

    set seed_iter 1
    set niter [lindex $argv 0]
    set mem_size [expr { 256*1024*1024 }]

    for {set iter 1} {$iter <= $niter} {incr iter} {
        expr { srand($seed_iter) }
        set seed_iter [expr { int(rand() * $::MAX_INT) }]
        set seed [expr { int(rand() * $::MAX_INT) }]

        puts "example-grad0: iter:${iter}/${niter}"
        set ctx0 [::ggml::create_context $mem_size]

        set ne_lst [get_random_dims 4]

        # add f32
        expr { srand($seed) }
        set nargs 2

        for {set n_dims 1} {$n_dims <= 4} {incr n_dims} {
            for {set i 0} {$i < $nargs} {incr i} {
                set tensor [get_random_tensor_f32 $ctx0 $n_dims $ne_lst -1.0 1.0]
                ::ggml::set_param $ctx0 $tensor
                lappend x $tensor
            }

            set f [::ggml::sum $ctx0 [::ggml::add $ctx0 [lindex $x 0] [lindex $x 1]]]

            check_gradient "add f32" $ctx0 $x $f $n_dims $nargs 0.001 0.002 0.002
        }
    }
}

main $argv0 $argv
