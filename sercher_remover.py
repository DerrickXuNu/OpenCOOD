import os
import pandas as pd

def collect_yaml_files_by_folder(root_dir):
    yaml_files_by_folder = {}
    frame_counter = {}
    old_folder_name = ""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        yaml_files = [f.split(".")[0] for f in filenames if f.endswith('.yaml')]
        # Remove .png files from the directory
        for f in filenames:
            if f.endswith('.png'):
                os.remove(os.path.join(dirpath, f))
        
        if yaml_files and len(yaml_files) > 1:
            new_dir_path = dirpath.split("\\")[-1]

            current_folder = dirpath.split("\\")[:-1][-1]
            

            if old_folder_name != current_folder: 
                frame_counter[new_dir_path] = len(yaml_files)            
                old_folder_name = current_folder

            yaml_files_by_folder[new_dir_path] = yaml_files
    print('All images removed')       
    print('*'*20)
    print("Frame counts per folder:", frame_counter)
    print("Total frames:", sum(frame_counter.values()))
    print('*'*20)
    return yaml_files_by_folder

# Example usage:
root_directory = r"C:\Users\mohammed.amirifard\Desktop\Github_Projects\OpenCOOD\data\test\test_culver_city"
yaml_files = collect_yaml_files_by_folder(root_directory)

import csv

def save_dict_columnwise(data, output_file):
    # Get the maximum number of files in any folder
    max_files = max(len(files) for files in data.values())

    # Create header: 'Folder', 'File_1', 'File_2', ...
    header = ['Folder'] + [f'File_{i+1}' for i in range(max_files)]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for folder, files in data.items():
            row = [folder] + files + [''] * (max_files - len(files))  # pad with empty strings
            writer.writerow(row)
    print(f"Data saved to {output_file}")
# Example usage:
save_dict_columnwise(yaml_files, 'yaml_files_columnwise.csv')
