# Define the output file
output_file="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/mrna-rbp/combined_ENCORI_hg38_RBPTarget.txt"

# Create/empty the output file
> "$output_file"

# Define a file to save the names of combined files
combined_files_log="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/mrna-rbp/combined_files.txt"
> "$combined_files_log"  # Empty the log file

# Get the list of files starting with ENCORI_hg38_RBPTarget
files=(/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/mrna-rbp/all_rbp_files/ENCORI_hg38_RBPTarget*.txt)

# Process the first file, including the headers
if [[ -s "${files[0]}" ]]; then  # Check if the first file is not empty
    cat "${files[0]}" >> "$output_file"
    echo "${files[0]}" >> "$combined_files_log"  # Log the combined file name
    echo "Combined: ${files[0]}"  # Echo the name of the combined file

fi

# Loop through the remaining files and check for an empty last line
for file in "${files[@]:1}"; do
    if [[ -s "$file" && $(tail -n 1 "$file") == "" ]]; then  # Check if the file is not empty and ends with an empty line
        tail -n +5 "$file" >> "$output_file"  # Remove the first 4 lines
        echo "$file" >> "$combined_files_log"  # Log the combined file name
        echo "Combined: $file"  # Echo the name of the combined file

    fi
done

# Count the number of files processed
file_count=$(wc -l < "$combined_files_log")

echo "All files combined into $output_file"
echo "Total number of files processed: $file_count"
echo "Combined file names saved in $combined_files_log"
