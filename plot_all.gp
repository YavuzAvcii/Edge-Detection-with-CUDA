reset

# --- Input Variables ---
file_shared  = "time_measurements/shared.dat"
file_global  = "time_measurements/global.dat"

# --- Output Variables (Shared) ---
out_block_s  = "time_measurements/graphs/1_block_vs_time_shared.png"
out_kernel_s = "time_measurements/graphs/2_kernel_vs_time_shared.png"
out_3d_s     = "time_measurements/graphs/3_combined_3d_shared.png"

# --- Output Variables (Global) ---
out_block_g  = "time_measurements/graphs/1_block_vs_time_global.png"
out_kernel_g = "time_measurements/graphs/2_kernel_vs_time_global.png"
out_3d_g     = "time_measurements/graphs/3_combined_3d_global.png"

# --- Output Variable (Comparison) ---
out_compare  = "time_measurements/graphs/comparison.png"

# --- General Style Settings ---
set terminal pngcairo size 800,600 enhanced font 'Verdana,10'
set grid
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5   # Blue, Circle
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5   # Red, Square

# ==========================================
# PART 1: SHARED MEMORY PLOTS
# ==========================================

# 1.1 Shared: Block Dim vs Time
set output out_block_s
set title "Shared Mem: Block Dimension vs Execution Time"
set xlabel "Block Dimension"
set ylabel "Execution Time (s)"
plot file_shared using 1:3 smooth unique title "Shared Time" with linespoints ls 1

# 1.2 Shared: Kernel Dim vs Time
set output out_kernel_s
set title "Shared Mem: Kernel Dimension vs Execution Time"
set xlabel "Kernel Dimension"
plot file_shared using 2:3 smooth unique title "Shared Time" with linespoints ls 1

# 1.3 Shared: 3D Surface
set output out_3d_s
set title "Shared Mem: 3D View"
set xlabel "Block Dim"
set ylabel "Kernel Dim"
set zlabel "Time (s)"
set view 60, 30 
set grid z vertical
set dgrid3d 30,30
set hidden3d
splot file_shared using 1:2:3 title "Shared Surface" with lines ls 1

# ==========================================
# PART 2: GLOBAL MEMORY PLOTS
# ==========================================

# 2.1 Global: Block Dim vs Time
# Note: standard 'plot' ignores dgrid3d, so we don't strictly need to unset it, 
# but it is good practice if you see weird artifacts. 
unset dgrid3d 

set output out_block_g
set title "Global Mem: Block Dimension vs Execution Time"
set xlabel "Block Dimension"
set ylabel "Execution Time (s)"
plot file_global using 1:3 smooth unique title "Global Time" with linespoints ls 2

# 2.2 Global: Kernel Dim vs Time
set output out_kernel_g
set title "Global Mem: Kernel Dimension vs Execution Time"
set xlabel "Kernel Dimension"
plot file_global using 2:3 smooth unique title "Global Time" with linespoints ls 2

# 2.3 Global: 3D Surface
set output out_3d_g
set title "Global Mem: 3D View"
set xlabel "Block Dim"
set ylabel "Kernel Dim"
set zlabel "Time (s)"
set dgrid3d 30,30
set hidden3d
splot file_global using 1:2:3 title "Global Surface" with lines ls 2

# ==========================================
# PART 3: COMPARISON (Block vs Time)
# ==========================================

unset dgrid3d # Turn off 3D grid for 2D plot

set output out_compare
set title "Comparison: Global vs Shared Memory (Block Dim)"
set xlabel "Block Dimension"
set ylabel "Execution Time (s)"

# Plot both files on one graph
plot file_shared using 1:3 smooth unique title "Shared Memory" with linespoints ls 1, \
     file_global using 1:3 smooth unique title "Global Memory" with linespoints ls 2
