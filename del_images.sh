#!/bin/bash

# Loop through all folders from 00001 to 00054
for i in $(seq -w 00001 00054); do
    # Define the source directory for the current folder
    SOURCE_DIR="image_folder/$i"
    
    # Start timer
    start_time=$(date +%s)
    
    # Delete all image files from the source directory and its subdirectories
    find "$SOURCE_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.gif" \) -exec rm -f {} \;
    
    # End timer
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    
    echo "All image files have been deleted from $SOURCE_DIR in $elapsed_time seconds."
done

