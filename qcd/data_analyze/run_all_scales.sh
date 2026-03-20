#!/bin/bash
source /home/khw/Documents/Git_repository/qcd/data_analyze/.venv/bin/activate
cd /home/khw/Documents/Git_repository/qcd/data_analyze

# Configurations to run
configs=(
    "8,4,6.0"
    "16,8,6.0"
    "32,16,6.0"
    "16,8,5.7"
    "16,8,5.8"
    "16,8,5.9"
    "16,8,6.0"
    "16,8,6.5"
)

for config in "${configs[@]}"; do
    T=$(echo $config | cut -d, -f1)
    S=$(echo $config | cut -d, -f2)
    beta=$(echo $config | cut -d, -f3)
    
    echo "Running T=$T S=$S beta=$beta..."
    
    # Update config in scale_setting.py
    sed -i "s/^T        = .*/T        = $T/" scale_setting.py
    sed -i "s/^S        = .*/S        = $S/" scale_setting.py
    sed -i "s/^beta     = .*/beta     = $beta/" scale_setting.py
    
    python scale_setting.py 2>&1 | tail -20
    echo "---"
done
