# measure2.tcl: report-only measurement of one checkpoint plus a constraint fingerprint (weekend run 3, Part 2).
# measure.tcl (Part 1, flagged) + lib_run3.tcl's rl3_fingerprint + an optional reset of the setup clock
# uncertainty before measuring (for outputs of nudged inputs). NEWLY ASSEMBLED Tcl (flagged); lines from:
#   measure.tcl (all lines); run 1 recipes n_unc*: "set_clock_uncertainty -setup 0 [get_clocks]" (unc_reset step).
# usage: vivado -mode batch -source measure2.tcl -tclargs <dcp> <outdir> <tag> [full 0|1] [reset_unc 0|1]
source [file join [file dirname [info script]] lib_measure.tcl]
source [file join [file dirname [info script]] lib_run3.tcl]
set dcp [lindex $argv 0]
set out [lindex $argv 1]
set tag [lindex $argv 2]
set full [lindex $argv 3]
set reset_unc [lindex $argv 4]
if {$full eq ""} { set full 0 }
if {$reset_unc eq ""} { set reset_unc 0 }
set_param general.maxThreads 8
set t0 [clock milliseconds]
open_checkpoint $dcp
puts "RL_STEP open [expr {[clock milliseconds]-$t0}] ms"
if {$reset_unc} {
  set_clock_uncertainty -setup 0 [get_clocks]
  puts "RL_STEP unc_reset done"
}
measure_design $out $tag $full
rl3_fingerprint $out $tag
puts "RL_STEP total [expr {[clock milliseconds]-$t0}] ms"
close_design
