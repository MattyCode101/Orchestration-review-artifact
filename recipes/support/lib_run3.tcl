# lib_run3.tcl: helper procs for weekend run 3, Part 2. NEWLY WRITTEN Tcl (flagged). Sourced after run 1's
# lib_measure.tcl (uses its rl_try). Each Vivado command used here, and its source:
#   get_timing_paths -setup -max_paths, get_property SLACK  run 1 recipes (RL_SNAP line); help_get_timing_paths.txt
#   get_cells -hierarchical [-filter]                       run 1 lib_measure.tcl (counts); help_get_cells.txt
#   get_cells -of_objects <paths> -filter                   UG949 p416 recipe "get_cells -of [get_timing_paths ...] -filter"
#                                                           (via the backend flow shortlist); help_get_cells.txt
#   write_xdc -exclude_physical -no_tool_comments -force    help_write_xdc.txt (Vivado 2025.1): "an XDC file that
#                                                           contains only timing constraints"
#   report_clocks -file                                     help_report_clocks.txt

# One-line snapshot after a recipe step, so a step that silently did nothing is visible in the log:
# worst setup slack (as run 1's RL_SNAP), all cells, logical LUT and FF cells.
proc rl3_snap {label} {
  rl_try snap_$label {
    set p [get_timing_paths -max_paths 1 -setup]
    set nl [llength [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]]
    set nf [llength [get_cells -hierarchical -filter {REF_NAME =~ FD*}]]
    puts "RL_SNAP $label wns=[get_property SLACK $p] cells=[llength [get_cells -hierarchical]] luts=$nl ffs=$nf"
  }
}

# Leaf cells on the n worst setup paths that are not already DONT_TOUCH (the cells a recipe may protect or exclude).
proc rl3_crit_cells {n} {
  set paths [get_timing_paths -setup -max_paths $n]
  set cells [get_cells -of_objects $paths -filter {DONT_TOUCH == 0}]
  puts "RL_CRIT n_paths=[llength $paths] n_cells=[llength $cells]"
  return $cells
}

# Constraint fingerprint of the open design: timing-only XDC, clock report, and counts of the properties a
# recipe may set temporarily (they must be back to the input's values at the end).
proc rl3_fingerprint {out tag} {
  set r [file join $out reports]
  file mkdir $r
  rl_try fp_xdc    { write_xdc -exclude_physical -no_tool_comments -force [file join $r ${tag}_timing.xdc] }
  rl_try fp_clocks { report_clocks -file [file join $r ${tag}_report_clocks.rpt] }
  rl_try fp_props  {
    set fh [open [file join $r ${tag}_props.txt] w]
    puts $fh "n_DONT_TOUCH\t[llength [get_cells -hierarchical -filter {DONT_TOUCH == 1}]]"
    puts $fh "n_IS_LOC_FIXED\t[llength [get_cells -hierarchical -filter {IS_LOC_FIXED == 1}]]"
    puts $fh "n_IS_BEL_FIXED\t[llength [get_cells -hierarchical -filter {IS_BEL_FIXED == 1}]]"
    close $fh
  }
}

# NEW (Part 3): placement-preserving merge of exact duplicate drivers on a PLACED or ROUTED design. This is the step
# opt_design -merge_equivalent_drivers performs only on unplaced logic (Part 3 runs e1_*: 0 cells removed on placed and
# routed states). A duplicate is a LUT (or FF) cell with the same REF_NAME, INIT, pin-inversion properties and the same net
# on every input pin as another cell. Its leaf loads move to the kept cell's output net, and it is removed. The caller then
# runs route_design, which is re-entrant (UG904 p60-61), to route the moved loads. Commands: UG904 p257-261 "Modifying the
# Netlist" (disconnect_net, connect_net, remove_cell) and Vivado 2025.1 help (code/help_part3/): connect_net -hierarchical
# (adds pins and nets across hierarchy), disconnect_net -prune -pinlist, get_pins -leaf -of_objects, get_nets -segments.
# A duplicate is skipped when its net reaches a port, or feeds a CARRY*/MUXF* cell (dedicated LUT connections), or, with
# protect=1, when it, the kept cell or a load is in ::rl3_crit.
proc rl3_eco_merge {kind {protect 0}} {
  if {$kind eq "lut"} {
    set cells [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]; set opin O; set extra {}
  } else {
    set cells [get_cells -hierarchical -filter {REF_NAME =~ FD*}]; set opin Q
    set extra {IS_C_INVERTED IS_D_INVERTED IS_R_INVERTED IS_S_INVERTED IS_CLR_INVERTED IS_PRE_INVERTED}
  }
  array unset crit
  if {$protect} { foreach c $::rl3_crit { set crit($c) 1 } }
  array unset grp
  foreach c $cells {
    set k "[get_property REF_NAME $c]|[get_property INIT $c]"
    foreach p $extra { catch { append k "|$p=[get_property $p $c]" } }
    foreach p [lsort [get_pins -of_objects $c -filter {DIRECTION == IN}]] {
      append k "|[lindex [split $p /] end]=[get_nets -of_objects $p]"
    }
    lappend grp($k) $c
  }
  set merged 0; set skipped 0; set moved 0
  foreach k [array names grp] {
    set g $grp($k)
    if {[llength $g] < 2} { continue }
    set keep ""; set kn -1
    foreach c $g {
      set n [llength [get_pins -quiet -leaf -of_objects [get_nets -of_objects [get_pins $c/$opin]] -filter {DIRECTION == IN}]]
      if {$n > $kn} { set kn $n; set keep $c }
    }
    set knet [get_nets -of_objects [get_pins $keep/$opin]]
    foreach c $g {
      if {$c eq $keep} { continue }
      set dnet [get_nets -of_objects [get_pins $c/$opin]]
      set loads [get_pins -quiet -leaf -of_objects $dnet -filter {DIRECTION == IN}]
      set skip 0
      if {[llength [get_ports -quiet -of_objects [get_nets -quiet -segments $dnet]]] > 0} { set skip 1 }
      # pilot (e2_*, 3d-rendering, spam-filter): leaf loads inside a LUTRAM macro (e.g. RAM256X1S/RAMS64E_A/ADR4) cannot
      # be edited ("Netlist 29-352 Cannot modify ... contained in a macro primitive"), and connect_net then failed. Check
      # before any disconnect: skip loads on macro-internal cells (PRIMITIVE_LEVEL INTERNAL) and on clock buffers too.
      if {!$skip && [llength $loads] && [llength [get_cells -quiet -of_objects $loads -filter {REF_NAME =~ CARRY* || REF_NAME =~ MUXF* || REF_NAME =~ BUFG* || PRIMITIVE_LEVEL == INTERNAL}]] > 0} {
        set skip 1
      }
      if {!$skip && $protect} {
        if {[info exists crit($c)] || [info exists crit($keep)]} { set skip 1 }
        foreach lc [get_cells -quiet -of_objects $loads] { if {[info exists crit($lc)]} { set skip 1; break } }
      }
      if {$skip} { incr skipped; continue }
      foreach p $loads { disconnect_net -prune -pinlist $p }
      if {[llength $loads]} { connect_net -hierarchical -net $knet -objects $loads }
      remove_cell $c
      incr merged; incr moved [llength $loads]
    }
  }
  puts "RL_ECO $kind merged=$merged skipped=$skipped moved_loads=$moved"
}
