# Weekend run 3, Part 2 -- NEW Tcl (diagnostic only; the design is opened and closed, never written).
# Census of LUT cells, their placed BELs and LUT-combining properties. It explains why worker B's re-placement packs
# fewer LUT pairs after LATCH's phys_opt than from B2 (part2_log, Sat 00:30).
# Commands: open_checkpoint, get_cells -hier -filter, list_property (Vivado 2025.1 help). Filters on REF_NAME as in
# lib_run3.tcl (rl3_snap). A LUT cell on a *5LUT BEL shares its site with the 6LUT cell, which is what
# report_utilization's "adjusted for LUT combining" counts once.
# usage: vivado -mode batch -source diag_lut.tcl -tclargs <dcp>
set dcp [lindex $argv 0]
open_checkpoint $dcp
set luts [get_cells -hier -filter {REF_NAME =~ LUT*}]
puts "RL_DIAG n_lut_cells [llength $luts]"
foreach r {LUT1 LUT2 LUT3 LUT4 LUT5 LUT6 LUT6_2} {
  puts "RL_DIAG n_$r [llength [get_cells -hier -filter "REF_NAME == $r"]]"
}
puts "RL_DIAG bel_5lut [llength [get_cells -hier -filter {REF_NAME =~ LUT* && BEL =~ *5LUT}]]"
puts "RL_DIAG bel_6lut [llength [get_cells -hier -filter {REF_NAME =~ LUT* && BEL =~ *6LUT}]]"
set props [list_property [lindex $luts 0]]
puts "RL_DIAG lut_props $props"
foreach p $props {
  if {[string match -nocase *LUTNM* $p]} {
    if {[catch {set n [llength [get_cells -hier -filter "REF_NAME =~ LUT* && $p != \"\""]]} err]} {
      puts "RL_DIAG n_$p ERROR $err"
    } else {
      puts "RL_DIAG n_$p $n"
    }
  }
}
close_design
