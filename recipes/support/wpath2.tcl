# Read-only probe: worst setup path + utilisation of a routed checkpoint.
# usage: vivado -mode batch -source wpath2.tcl -tclargs <dcp> <outfile>
set dcp [lindex $argv 0]
set out [lindex $argv 1]
open_checkpoint $dcp
set fh [open $out w]
set p [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
foreach prop {SLACK REQUIREMENT DATAPATH_DELAY DATAPATH_LOGIC_DELAY DATAPATH_NET_DELAY LOGIC_LEVELS SKEW STARTPOINT_PIN ENDPOINT_PIN} {
    puts $fh "$prop\t[get_property $prop $p]"
}
set cells [get_cells -quiet -of_objects $p]
set comp [dict create]
foreach c $cells { dict incr comp [get_property REF_NAME $c] }
puts $fh "CELLS\t[llength $cells]\t$comp"
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
    puts $fh "TILE_SPAN\tX [lindex $xs 0]-[lindex $xs end]\tY [lindex $ys 0]-[lindex $ys end]"
}
foreach pb [get_pblocks -quiet] {
    puts $fh "PBLOCK\t$pb\tranges [get_property GRID_RANGES $pb]"
}
if {[llength [get_pblocks -quiet]] == 0} { puts $fh "PBLOCK\tnone" }
set u [report_utilization -return_string]
foreach {key pat} {LUT {CLB LUTs} FF {CLB Registers} BRAM {Block RAM Tile} DSP {DSPs} URAM {URAM}} {
    set v "na"
    set re [format {^\|\s*%s\*?\s*\|\s*([0-9.]+)\s*\|} $pat]
    foreach line [split $u "\n"] {
        if {[regexp $re $line -> n]} { set v $n; break }
    }
    puts $fh "UTIL_$key\t$v"
}
close $fh
