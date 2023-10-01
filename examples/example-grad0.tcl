proc check_gradient {op_name ctx0 x f ndims nargs eps max_error_abs max_error_rel {nthreads 10}} {
    set gf [::ggml::build_forward_ctx $ctx0 $f]
    set gb [::ggml::build_backward_ctx $ctx0 $gf false]
    ::ggml::graph_compute $gf $nthreads
    ::ggml::graph_reset $gf
    ::ggml::set_f32 [::ggml::get_grad $f] 1.0
    ::ggml::graph_compute $gb $nthreads

    for {set i 0} {$i < $nargs} {incr i} {
        set nelements = [::ggml::nelements [lindex $x $i]]

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

            set g1 [::ggml::get_f32_1d [::ggml::get_grad [lindex $x $i]] $k

            set error_abs [expr { abs($g0 - $g1) }]
            set error_rel [expr { $g0 != 0 ? abs($g0 - $g1)/abs($g0) : 0 }]

            if { $error_abs > $max_error_abs || $error_rel > $max_error_rel } {
                puts "${op_name}: ndims=${ndmis}, i=${i}, k=${k}, x0=${x0}, xm=${xm}, xp=${xp}, f0=${f0}, f1=${f1}, g0=${g0}, g1=${g1}, eps=${eps}, error_abs=${error_abs}, error_rel=${error_rel}"
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

proc get_random_dims {ndims} {
    set dims [list]
    for {set i 0} {$i < $ndims} {incr i} {
        lappend dims [expr { 1 + irand(${ndims}) }]
    }
    return $dims
}

proc main {argv} {

    if [llength $argv] != 1 {
        puts "usage: $argv0 <niter>"
        exit 1
    }

    set seed_iter 1
    set niter [lindex $argv 0]
    set mem_size [expr { 256*1024*1024 }]

    for {set iter 0} {$iter < $niter} {incr iter} {
            expr { srand($seed_iter) }
            set seed_iter [expr { int(rand() * $::MAX_INT) }]
            set seed [expr { int(rand() * $::MAX_INT) }]

            puts "example-grad0: iter:${iter}/${niter}"
            set ctx0 [::ggml::create_context [dict create mem_size $mem_size]]

            set dims [get_random_dims 4]

            struct ggml_tensor * x[MAX_NARGS];

            // add f32
            {
                expr { srand(seed) }
                set nargs 2

                for {set ndims 1} {$ndims <= 4} {incr ndims} {
                    for {set i 0} {$i < $nargs} {incr i} {
                        set tensor [get_random_tensor_f32 $ctx0 $ndims $ne -1.0 1.0]
                        ::ggml::set_param $ctx0 $tensor
                        lappend x $tensor
                    }

                    set f [::ggml::sum $ctx0 [::ggml::add $ctx0 [lindex $x 0] [lindex $x 1]]]

                    check_gradient "add f32" $ctx0 $x $f $ndims $nargs 0.001 0.002 0.002
                }
            }
        }
    }
}

main $argv
