import os, sys, shutil
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
    print(f"RESULT {result}")

    # Extract job ID from sbatch output 
    job_id = result.stdout.strip().split()[-1]
    return job_id

def is_job_finished(job_id):
    """Check if the job has finished using sacct."""
    result = subprocess.run(['sacct', '-j', job_id, '--format=State', '--noheader', '--parsable2'], capture_output=True, text=True)
    #test = subprocess.run(['scontrol', 'show', 'jobid', job_id], capture_output=True, text=True)
    
    # Get job state
    job_state = result.stdout.split('\n')[0]
#    print(f"res {job_state}")
    return job_state if job_state in {'COMPLETED', 'FAILED', 'CANCELLED'} else ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--depths_dir', default="")    # can be set if depths were not generated using automatic generate_colmap script
    parser.add_argument('--global_colmap_dir', required=True)
    parser.add_argument('--chunks_dir', required=True)
    parser.add_argument('--use_slurm', action="store_true", default=False)
    # parser.add_argument('--colmap_exe', default="colmap.bat")
    args = parser.parse_args()
    preprocess_dir = os.path.dirname(os.path.realpath(__file__))

#    if args.use_slurm:
#        gpu='-C a100 -A hzb@a100'
#        slurm_args = [
#            "sbatch",
#             #gpu,
##            "--ntasks=1", "--nodes=1",
##            "--gres=gpu:1", "--cpus-per-task=20",
##            "--time=3:00:00"  
#        ]
    submitted_jobs_ids = []
    os_name = platform.system()

    colmap_exe = "colmap.bat" if os_name == "Windows" else "colmap"
    start_time = time.time()
    depths_dir = args.depths_dir if args.depths_dir != "" else os.path.join(args.images_dir, "..", "depths")

    print(f"chunking colmap from {args.global_colmap_dir} to {args.chunks_dir}/raw_chunks")
    make_chunk_args = [
            "python", f"{preprocess_dir}/make_chunk.py",
            "--base_dir", f"{args.global_colmap_dir}",
            "--images_dir", f"{args.images_dir}",
            "--output_path", f"{args.chunks_dir}/raw_chunks",
        ]
    try:
        subprocess.run(make_chunk_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing image_undistorter: {e}")
        sys.exit(1)

    print("TEST WITH ONLY 1 CHUNK")
    chunk_name = os.listdir(os.path.join(args.chunks_dir, "raw_chunks"))[0]
    in_dir = os.path.join(args.chunks_dir, "raw_chunks", chunk_name)
    bundle_adj_dir = os.path.join(args.chunks_dir, "raw_chunks", chunk_name, "bundle_adjustment")
    out_dir = os.path.join(args.chunks_dir, "chunks", chunk_name)

    if args.use_slurm:
        gpu='-C a100 -A hzb@a100'
        slurm_args = [
            "sbatch", f"--error={in_dir}/log.err",
            f"--output={in_dir}/log.out"
             #gpu,
#            "--ntasks=1", "--nodes=1",
#            "--gres=gpu:1", "--cpus-per-task=20",
#            "--time=3:00:00"  
        ]

        # Process chunks in parallel
#            str_args = " ".join(slurm_args + ["preprocess/prepare_chunk.slurm", in_dir, bundle_adj_dir, out_dir,args.images_dir, depths_dir, preprocess_dir])
            #print(f"STR ARGS {str_args}")
        job_id = submit_job(slurm_args + ["preprocess/prepare_chunk.slurm", in_dir, bundle_adj_dir, out_dir,args.images_dir, depths_dir, preprocess_dir])
#            job_id = submit_job(slurm_args, [
#                "preprocess/prepare_chunk.slurm",
#                in_dir, bundle_adj_dir, out_dir, 
#                args.images_dir, depths_dir, preprocess_dir
#                ])
            # job_id = submit_job(slurm_args, [
            #     f"{preprocess_dir}/prepare_chunk.py",
            #     "--in_dir", in_dir, "--bundle_adj_dir", bundle_adj_dir,"--out_dir", out_dir, 
            #     "--images_dir", args.images_dir, "--depths_dir", args.depths_dir,"--preprocess_dir", preprocess_dir, "--is_job"
            #     ])
        submitted_jobs_ids.append(job_id)
    
        # Check every 10 sec all the jobs status
        all_finished = False
        all_status = []
        time_limit = 180
        while not all_finished and time_limit:
            # print("Checking status of all jobs...")
            all_status = [is_job_finished(id) for id in submitted_jobs_ids if is_job_finished(id) != ""]
          
            all_finished = len(all_status) == len(submitted_jobs_ids)
            if not all_finished:
                time.sleep(10)  # Wait before checking again
                time_limit -= 10

        if not all(status == "COMPLETED" for status in all_status):
            print("At least one job failed or was cancelled, check at error logs.")
        print(f"STATUS WHEN DONE: {all_status}")
    end_time = time.time()
    print(f"chunks successfully prepared in {(end_time - start_time)/60.0} minutes.")

