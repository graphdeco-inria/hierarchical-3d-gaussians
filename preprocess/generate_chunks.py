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

import os, sys
import subprocess
import argparse
import time, platform

def submit_job(slurm_args):
    """Submit a job using sbatch and return the job ID."""    
    try:
        result = subprocess.run(slurm_args, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error when submitting a job: {e}")
        sys.exit(1)
    
    job = result.stdout.strip().split()[-1]
    print(f"submitted job {job}")
    return job

def is_job_finished(job):
    """Check if the job has finished using sacct."""
    result = subprocess.run(['sacct', '-j', job, '--format=State', '--noheader', '--parsable2'], capture_output=True, text=True)
    
    job_state = result.stdout.split('\n')[0]
    return job_state if job_state in {'COMPLETED', 'FAILED', 'CANCELLED'} else ""

def setup_dirs(images, colmap, chunks, project):
    images_dir = os.path.join(project, "camera_calibration", "rectified", "images") if images == "" else images
    colmap_dir = os.path.join(project, "camera_calibration", "aligned") if colmap == "" else colmap
    chunks_dir = os.path.join(project, "camera_calibration") if chunks == "" else chunks

    return images_dir, colmap_dir, chunks_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', required=True, help="images, colmap and chunks paths doesnt have to be set if you generated the colmap using generate_colmap script.")
    parser.add_argument('--images_dir', default="")
    parser.add_argument('--global_colmap_dir', default="")
    parser.add_argument('--chunks_dir', default="")
    parser.add_argument('--use_slurm', action="store_true", default=False)
    parser.add_argument('--skip_bundle_adjustment', action="store_true", default=False)
    parser.add_argument('--n_jobs', type=int, default=8, help="Run per chunk COLMAP in parallel on the same machine. Does not handle multi GPU systems. --use_slurm overrides this.")
    args = parser.parse_args()
    
    images_dir, colmap_dir, chunks_dir = setup_dirs(
        args.images_dir,
        args.global_colmap_dir, args.chunks_dir,
        args.project_dir
    ) 

    if args.use_slurm:
        slurm_args = [
            "sbatch" 
        ]
    submitted_jobs = []

    colmap_exe = "colmap.bat" if platform.system() == "Windows" else "colmap"
    start_time = time.time()

    ## First create raw_chunks, each chunk has its own colmap.
    print(f"chunking colmap from {colmap_dir} to {args.chunks_dir}/raw_chunks")
    make_chunk_args = [
            "python", f"preprocess/make_chunk.py",
            "--base_dir", os.path.join(colmap_dir, "sparse", "0"),
            "--images_dir", f"{images_dir}",
            "--output_path", f"{chunks_dir}/raw_chunks",
        ]
    try:
        subprocess.run(make_chunk_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing image_undistorter: {e}")
        sys.exit(1)

    ## Then we refine chunks with 2 rounds of bundle adjustment/triangulation
    print("Starting per chunk triangulation and bundle adjustment (if required)")
    n_processed = 0
    chunk_names = os.listdir(os.path.join(chunks_dir, "raw_chunks"))
    for chunk_name in chunk_names:
        in_dir = os.path.join(chunks_dir, "raw_chunks", chunk_name)
        out_dir = os.path.join(chunks_dir, "chunks", chunk_name)

        if args.use_slurm:
            # Process chunks in parallel
            job = submit_job(slurm_args + [
                f"--error={in_dir}/log.err", f"--output={in_dir}/log.out",
                "preprocess/prepare_chunk.slurm", in_dir, out_dir,images_dir,
                os.path.dirname(os.path.realpath(__file__))
                ])
            submitted_jobs.append(job)
        else:
            try:
                if len(submitted_jobs) >= args.n_jobs:
                    submitted_jobs.pop(0).communicate()
                intermediate_dir = os.path.join(in_dir, "bundle_adjustment")
                if os.path.exists(intermediate_dir):
                    print(f"{intermediate_dir} exists! Per chunk triangulation might crash!")
                prepare_chunk_args = [
                        "python", f"preprocess/prepare_chunk.py",
                        "--raw_chunk", in_dir, "--out_chunk", out_dir, 
                        "--images_dir", images_dir
                ]
                if args.skip_bundle_adjustment:
                    prepare_chunk_args.append("--skip_bundle_adjustment")
                job = subprocess.Popen(
                    prepare_chunk_args,
                    stderr=open(f"{in_dir}/log.err", 'w'), 
                    stdout=open(f"{in_dir}/log.out", 'w'),
                )
                submitted_jobs.append(job)
                n_processed += 1
                print(f"Launched triangulation for [{n_processed} / {len(chunk_names)} chunks].")
                print(f"Logs in {in_dir}/log.err (or .out)")
            except subprocess.CalledProcessError as e:
                print(f"Error executing prepare_chunk.py: {e}")
                sys.exit(1)


    if args.use_slurm:
        # Check every 10 sec all the jobs status
        all_finished = False
        all_status = []
        last_count = 0
        print(f"Waiting for chunks processed in parallel to be done ...")

        while not all_finished:
            # print("Checking status of all jobs...")
            all_status = [is_job_finished(id) for id in submitted_jobs if is_job_finished(id) != ""]
            if last_count != all_status.count("COMPLETED"):
                last_count = all_status.count("COMPLETED")
                print(f"processed [{last_count} / {len(chunk_names)} chunks].")

            all_finished = len(all_status) == len(submitted_jobs)
    
            if not all_finished:
                time.sleep(10)  # Wait before checking again
        
        if not all(status == "COMPLETED" for status in all_status):
            print("At least one job failed or was cancelled, check at error logs.")
    else:
        for job in submitted_jobs:
            job.communicate()

    # create chunks.txt file that concatenates all chunks center.txt and extent.txt files
    try:
        subprocess.run([
            "python", "preprocess/concat_chunks_info.py",
            "--base_dir", os.path.join(chunks_dir, "chunks"),
            "--dest_dir", colmap_dir
        ], check=True)
        n_processed += 1
    except subprocess.CalledProcessError as e:
        print(f"Error executing concat_chunks_info.sh: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"chunks successfully prepared in {(end_time - start_time)/60.0} minutes.")

