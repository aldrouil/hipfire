#!/bin/bash
# power_snap.sh — methodology power/memory/clocks snapshot from sysfs (dGPU = card1).
# Prints one labeled block; caller redirects to meta log pre/post.
set -u
D=/sys/class/drm/card1/device
H=$D/hwmon/hwmon0
echo "power1_average_uW=$(cat $H/power1_average 2>/dev/null || echo NA)"
echo "power1_cap_uW=$(cat $H/power1_cap 2>/dev/null || echo NA)"
echo "temp_edge_mC=$(cat $H/temp1_input 2>/dev/null || echo NA)"
echo "temp_junction_mC=$(cat $H/temp2_input 2>/dev/null || echo NA)"
echo "temp_mem_mC=$(cat $H/temp3_input 2>/dev/null || echo NA)"
echo "sclk_Hz=$(cat $H/freq1_input 2>/dev/null || echo NA)"
echo "mclk_Hz=$(cat $H/freq2_input 2>/dev/null || echo NA)"
echo "perf_level=$(cat $D/power_dpm_force_performance_level 2>/dev/null || echo NA)"
echo "pp_sclk_star=$(grep -E '\*' $D/pp_dpm_sclk 2>/dev/null | head -1 || echo NA)"
echo "throttle_log=$(cat $D/thermal_throttling_logging 2>/dev/null | head -c 300 || echo NA)"
echo "vram_used_B=$(cat $D/mem_info_vram_used 2>/dev/null || echo NA)"
echo "vram_total_B=$(cat $D/mem_info_vram_total 2>/dev/null || echo NA)"
echo "gtt_used_B=$(cat $D/mem_info_gtt_used 2>/dev/null || echo NA)"
echo "gpu_busy_pct=$(cat $D/gpu_busy_percent 2>/dev/null || echo NA)"
echo "mem_busy_pct=$(cat $D/mem_busy_percent 2>/dev/null || echo NA)"
echo "host_MemAvailable_kB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')"
