# Read-only probe of the worst setup path of a routed checkpoint.
# usage: vivado -mode batch -source wpath.tcl -tclargs <dcp> <outfile>
set dcp [lindex $argv 0]
set out [lindex $argv 1]
open_checkpoint $dcp
set fh [open $out w]
set p [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
foreach prop {SLACK REQUIREMENT DATAPATH_DELAY DATAPATH_LOGIC_DELAY DATAPATH_NET_DELAY LOGIC_LEVELS SKEW STARTPOINT_PIN ENDPOINT_PIN STARTPOINT_CLOCK} {
    puts $fh "$prop\t[get_property $prop $p]"
}
# cell composition of the path
set cells [get_cells -quiet -of_objects $p]
set comp [dict create]
foreach c $cells {
    set t [get_property REF_NAME $c]
    dict incr comp $t
}
puts $fh "CELLS\t[llength $cells]\t$comp"
# tile span of placed path cells
set xs {}; set ys {}
foreach c $cells {
    set s [get_sites -quiet -of_objects $c]
    if {[llength $s] == 0} { continue }
    set tl [get_tiles -quiet -of_objects [lindex $s 0]]
    if {[llength $tl] == 0} { continue }
    if {[regexp {_X(\d+)Y(\d+)$} [lindex $tl 0] -> tx ty]} { lappend xs $tx; lappend ys $ty }
}
if {[llength $xs] > 0} {
    set xs [lsort -integer $xs]; set ys [lsort -integer $ys]
    puts $fh "TILE_SPAN\tX [lindex $xs 0]-[lindex $xs end]\tY [lindex $ys 0]-[lindex $ys end]\tdx [expr {[lindex $xs end]-[lindex $xs 0]}]\tdy [expr {[lindex $ys end]-[lindex $ys 0]}]"
}
# pblocks present in the checkpoint
foreach pb [get_pblocks -quiet] {
    puts $fh "PBLOCK\t$pb\tcells [llength [get_cells -quiet -of_objects $pb]]\tranges [get_property GRID_RANGES $pb]"
}
if {[llength [get_pblocks -quiet]] == 0} { puts $fh "PBLOCK\tnone" }
# top-level design stats for context
puts $fh "N_CELLS\t[llength [get_cells -hierarchical -quiet -filter {IS_PRIMITIVE}]]"
close $fh
