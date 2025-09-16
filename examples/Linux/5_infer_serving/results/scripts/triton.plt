#!/usr/bin/env gnuplot

reset
set output eps_file
set terminal postscript "Helvetica,16" eps enhance color dl 2

set pointsize 1
set size 0.6,0.5
set nozeroaxis


set rmargin 2 #2
set lmargin 8 #5.5
set tmargin 0.7 #1.5
set bmargin 3.4 #2.5


### Key
set key inside left Left reverse top enhanced nobox
set key samplen 1.1 spacing 1.4 height 0.2 width 11 autotitles columnhead 
set key font ',18' noopaque #maxrows 1 at graph 0.02, graph 0.975  
# unset key

## Y-axis
set ylabel "CDF (%)" font ",20" offset 0.,0
set yrange [0:100]
set ytics 0,25,100
set ytics font ",20" #offset 0,0 #format "%.1f"
set ytics nomirror 


### X-axis
set xlabel "Latency (ms) " font ",20" offset 0,-.0
set xrange [0:16]
set xtics 0,4,20

set xtics font ",18" offset -0.8,0
set xtics nomirror 

STANDALONE_COLOR = "#1b7c3d"
TRITON_COLOR = "#376795"
TRITON_PRIORITY_COLOR = "#72bcd5"
XSched_COLOR = "#ec5d3b"



plot standalone_data u ($2):($1*100) t "Standalone" w l  lt 2 lw 5 lc rgb STANDALONE_COLOR, \
     triton_data u ($2):($1*100) t "Triton" w l  lt 2 lw 5 lc rgb TRITON_COLOR, \
     triton_priority_data u ($2):($1*100) t "T+Priority" w l  lt 2 lw 5 lc rgb TRITON_PRIORITY_COLOR, \
     xsched_data u ($2):($1*100) t "T+XSched" w l  lt 2 lw 5 lc rgb XSched_COLOR
    

