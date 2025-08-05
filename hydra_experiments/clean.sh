#!/bin/bash

# Loop through all directories in the current folder
for dir in */; do
    # Remove trailing slash
    dir=${dir%/}
    
    # Check if this is a directory and doesn't have a hydra_experiments subfolder
    if [ -d "$dir" ] && [ ! -d "$dir/hydra_experiments" ] && [ ! -d "$dir/metrics_history" ]; then # If directory doesn't contain hydra_experiments or metrics_history subfolders
        # Print the directory name
        echo "Moving directory to trash: $dir"
        gio trash "$dir" # Move the directory to trash instead of removing it
    fi
done

echo "Cleanup complete."