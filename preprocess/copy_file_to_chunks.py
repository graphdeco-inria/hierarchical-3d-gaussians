import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', required=True, help="File to copy")
parser.add_argument('--chunks_path', required=True, help="Path containing folders 0_0/, ...")
parser.add_argument('--out_subdir', default="sparse/0", help="Copy file_path to chunks_path/x_y/out_subdir/")
args = parser.parse_args()

chunks = os.listdir(args.chunks_path)

for chunk in chunks:
    shutil.copy(args.file_path, os.path.join(args.chunks_path, chunk, args.out_subdir))
