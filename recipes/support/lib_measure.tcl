# FRESHLY WRITTEN Tcl (not ported from any prior working source). Lower confidence than ported code.
# Every Vivado command used here has a help dump in notes/help_vivado/ (see notes/command_ledger.md).
#
# measure_design <outdir> <tag> <full>
#   Writes a fixed set of reports for the CURRENTLY OPEN design to <outdir>/reports/<tag>_*.
#   Each report is wrapped in catch so one failure cannot abort a run; failures are logged as MEASURE_ERROR.
#   Operating conditions are NOT changed anywhere in this project (power numbers use tool defaults).

proc rl_try {name script} {
  set t0 [clock milliseconds]
  if {[catch {uplevel 1 $script} err]} {
    puts "MEASURE_ERROR $name: $err"
  }
  puts "MEASURE_TIME $name [expr {[clock milliseconds]-$t0}] ms"
}

proc measure_design {out tag {full 0}} {
  set r [file join $out reports]
  file mkdir $r
  set t0 [clock milliseconds]
  rl_try timing_summary  { report_timing_summary -delay_type min_max -max_paths 10 -file [file join $r ${tag}_timing_summary.rpt] }
  rl_try utilization     { report_utilization -file [file join $r ${tag}_utilization.rpt] }
  rl_try control_sets    { report_control_sets -file [file join $r ${tag}_control_sets.rpt] }
  rl_try route_status    { report_route_status -file [file join $r ${tag}_route_status.rpt] }
  rl_try power           { report_power -file [file join $r ${tag}_power.rpt] }
  rl_try power_advisory  { report_power -advisory -file [file join $r ${tag}_power_advisory.rpt] }
  rl_try drc             { report_drc -file [file join $r ${tag}_drc.rpt] }
  rl_try clock_util      { report_clock_utilization -file [file join $r ${tag}_clock_utilization.rpt] }
  rl_try high_fanout     { report_high_fanout_nets -timing -max_nets 25 -file [file join $r ${tag}_high_fanout.rpt] }
  rl_try clocks          {
    set fh [open [file join $r ${tag}_clocks.txt] w]
    puts $fh "name\tperiod\tis_generated"
    foreach c [get_clocks] {
      puts $fh "[get_property NAME $c]\t[get_property PERIOD $c]\t[get_property IS_GENERATED $c]"
    }
    close $fh
  }
  rl_try counts {
    # direct evidence that gating / replication levers fired (filters verified by the baseline probe runs)
    set fh [open [file join $r ${tag}_counts.txt] w]
    puts $fh "n_cells_all\t[llength [get_cells -hierarchical]]"
    puts $fh "n_IS_CLOCK_GATED\t[llength [get_cells -hierarchical -filter {IS_CLOCK_GATED == 1}]]"
    puts $fh "n_replica_named_cells\t[llength [get_cells -hierarchical -filter {NAME =~ *replica*}]]"
    close $fh
  }
  if {$full} {
    rl_try methodology   { report_methodology -file [file join $r ${tag}_methodology.rpt] }
    rl_try congestion    { report_design_analysis -congestion -file [file join $r ${tag}_congestion.rpt] }
    rl_try complexity    { report_design_analysis -complexity -file [file join $r ${tag}_complexity.rpt] }
    rl_try logic_levels  { report_design_analysis -logic_level_distribution -file [file join $r ${tag}_logic_levels.rpt] }
  }
  puts "MEASURE_TOTAL $tag [expr {[clock milliseconds]-$t0}] ms"
}
