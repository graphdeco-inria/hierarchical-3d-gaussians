#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import argparse
import os

# if __name__ == 'main':
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', required=True, help="Chunks folder")
parser.add_argument('--dest_dir', required=True, help="Folder to which chunks.txt file will be written")
args = parser.parse_args()

chunks = os.listdir(args.base_dir)

chunks_data = []
for chunk in chunks:
    center_file_path = os.path.join(args.base_dir, chunk + "/center.txt")
    extents_file_path = os.path.join(args.base_dir, chunk + "/extent.txt")

    chunk = {
        "name":chunk,
        "center": [0,0,0],
        "extent": [0,0,0]
    }

    try:
        with open(center_file_path, 'r') as file:
            content = file.read()
            chunk["center"] = content.split(" ")
    except FileNotFoundError:
        print(f"File not found: {center_file_path}")

    try:
        with open(extents_file_path, 'r') as file:
            content = file.read()
            chunk["extent"] = content.split(" ")
    except FileNotFoundError:
        print(f"File not found: {extents_file_path}")

    chunks_data.append(chunk)

def write_chunks(data, output_directory):
    file_path = os.path.join(output_directory, "chunks.txt")
    try:
        with open(file_path, 'w') as file:
            ind = 0
            for chunk in data:
                line = chunk['name'] + " " + ' '.join(map(str, chunk['center'])) + " " +' '.join(map(str, chunk['extent'])) + "\n"
                
                if ind == len(data)-1:
                    line = line[:-1]

                # Write content to the file
                file.write(line)
                ind += 1
            print(f"Content written to {file_path}")

    except IOError:
        print(f"Error writing to {file_path}")

write_chunks(chunks_data, args.dest_dir)