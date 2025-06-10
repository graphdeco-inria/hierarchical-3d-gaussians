import os, sys
import subprocess
import argparse
import time
import platform
from pathlib import Path

def submit_job(slurm_args):
    """Submit a job using sbatch and return the job ID."""    
    try:
        result = subprocess.run(slurm_args, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error when submitting a job: {e}")
        sys.exit(1)
    # Extract job ID from sbatch output
    job_id = result.stdout.strip().split()[-1]
    print(f"submitted job {job_id}")

    return job_id

def is_job_finished(job_id):
    """Check if the job has finished using sacct."""
    result = subprocess.run(['sacct', '-j', job_id, '--format=State', '--noheader', '--parsable2'], capture_output=True, text=True)
    # Get job state
    job_state = result.stdout.split('\n')[0]
    return job_state if job_state in {'COMPLETED', 'FAILED', 'CANCELLED'} else ""

def setup_dirs(images, depths, masks, colmap, chunks, output, project):
    images_dir = "../rectified/images" if images == "" else images
    depths_dir = "../rectified/depths" if depths == "" else depths
    if masks == "":
        if os.path.exists(os.path.join(project, "camera_calibration/rectified/masks")):
            masks_dir = "../rectified/masks"
        else:
            masks_dir = ""
    else:
        masks_dir = masks
    colmap_dir = os.path.join(project, "camera_calibration", "aligned") if colmap == "" else colmap
    chunks_dir = os.path.join(project, "camera_calibration", "chunks") if chunks == "" else chunks
    output_dir = os.path.join(project, "output") if output == "" else output

    return images_dir, depths_dir, masks_dir, colmap_dir, chunks_dir, output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', required=True, help="Only the project dir has to be specified, other directories will be set according to the ones created using generate_colmap and generate_chunks scripts. They still can be explicitly specified.")
    parser.add_argument('--env_name', default="hierarchical_3d_gaussians")
    parser.add_argument('--extra_training_args', default="", help="Additional arguments that can be passed to training scripts. Not passed to slurm yet")
    parser.add_argument('--colmap_dir', default="")
    parser.add_argument('--images_dir', default="")
    parser.add_argument('--masks_dir', default="")
    parser.add_argument('--depths_dir', default="")
    parser.add_argument('--chunks_dir', default="")
    
    parser.add_argument('--output_dir', default="")
    parser.add_argument('--use_slurm', action="store_true", default=False)
    parser.add_argument('--skip_if_exists', action="store_true", default=False, help="Skip training a chunk if it already has a hierarchy")
    parser.add_argument('--keep_running', action="store_true", default=False, help="Keep running even if a chunk processing fails")
    args = parser.parse_args()
    print(args.extra_training_args)

    os_name = platform.system()
    f_path = Path(__file__)
    images_dir, depths_dir, masks_dir, colmap_dir, chunks_dir, output_dir = setup_dirs(
        args.images_dir, args.depths_dir,
        args.masks_dir, args.colmap_dir,
        args.chunks_dir, args.output_dir,
        args.project_dir
    )

    start_time = time.time()
    
    if not os.path.exists(output_dir):
        print(f"creating output dir: {output_dir}")
        os.makedirs(os.path.join(output_dir, "scaffold"))
        os.makedirs(os.path.join(output_dir, "trained_chunks"))

    slurm_args = ["sbatch"]

    ## First step is coarse optimization to generate a scaffold that will be used later.
    if args.skip_if_exists and os.path.exists(os.path.join(output_dir, "scaffold/point_cloud/iteration_30000/point_cloud.ply")):
        print("Skipping coarse")
    else:
        if args.use_slurm:
            if args.extra_training_args != "":
                print("\nThe script does not support passing extra_training_args to slurm!!\n")
            submitted_jobs_ids = []

            coarse_train = submit_job(slurm_args + [
                f"--error={output_dir}/scaffold/log.err", f"--output={output_dir}/scaffold/log.out",
                "coarse_train.slurm", args.env_name, colmap_dir, images_dir, output_dir, " --alpha_masks " + masks_dir
            ])
            print("waiting for coarse training to finish...")
            while is_job_finished(coarse_train) == "":
                time.sleep(10)
        else:
            train_coarse_args =  " ".join([
                "python", "train_coarse.py",
                "-s", colmap_dir,
                "--save_iterations", "-1",
                "-i", images_dir,
                "--skybox_num", "100000",
                "--model_path", os.path.join(output_dir, "scaffold")
            ])
            if masks_dir != "":
                train_coarse_args += " --alpha_masks " + masks_dir
            if args.extra_training_args != "": 
                train_coarse_args += " " + args.extra_training_args

            try:
                subprocess.run(train_coarse_args, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing train_coarse: {e}")
                sys.exit(1)


    if not os.path.isabs(images_dir):
        images_dir = os.path.join("../", images_dir)
    if not os.path.isabs(depths_dir):
        depths_dir = os.path.join("../", depths_dir)
    if masks_dir != "" and not os.path.isabs(masks_dir):
        masks_dir = os.path.join("../", masks_dir)

    ## Now we can train each chunks using the scaffold previously created
    train_chunk_args =  " ".join([
        "python", "-u train_single.py",
        "--save_iterations -1",
        f"-i {images_dir}", f"-d {depths_dir}",
        f"--scaffold_file {output_dir}/scaffold/point_cloud/iteration_30000",
        "--skybox_locked" 
    ])
    if masks_dir != "":
        train_chunk_args += " --alpha_masks " + masks_dir
    if args.extra_training_args != "": 
        train_chunk_args += " " + args.extra_training_args

    hierarchy_creator_args = "submodules/gaussianhierarchy/build/Release/GaussianHierarchyCreator.exe " if os_name == "Windows" else "submodules/gaussianhierarchy/build/GaussianHierarchyCreator "
    hierarchy_creator_args = os.path.join(f_path.parent.parent, hierarchy_creator_args)

    post_opt_chunk_args =  " ".join([
        "python", "-u train_post.py",
        "--iterations 15000", "--feature_lr 0.0005",
        "--opacity_lr 0.01", "--scaling_lr 0.001", "--save_iterations -1",
        f"-i {images_dir}",  f"--scaffold_file {output_dir}/scaffold/point_cloud/iteration_30000",
    ])
    if masks_dir != "":
        post_opt_chunk_args += " --alpha_masks " + masks_dir
    if args.extra_training_args != "": 
        post_opt_chunk_args += " " + args.extra_training_args

    
    chunk_names = os.listdir(chunks_dir)
    for chunk_name in chunk_names:
        source_chunk = os.path.join(chunks_dir, chunk_name)
        trained_chunk = os.path.join(output_dir, "trained_chunks", chunk_name)

        if args.skip_if_exists and os.path.exists(os.path.join(trained_chunk, "hierarchy.hier_opt")):
            print(f"Skipping {chunk_name}")
        else:
            ## Training can be done in parallel using slurm.
            if args.use_slurm:
                job_id = submit_job(slurm_args + [
                    f"--error={trained_chunk}/log.err", f"--output={trained_chunk}/log.out",
                    "train_chunk.slurm", source_chunk, output_dir, args.env_name, 
                    chunk_name, hierarchy_creator_args, images_dir,
                    depths_dir, " --alpha_masks " + masks_dir
                ])

                submitted_jobs_ids.append(job_id)
            else:
                print(f"Training chunk {chunk_name}")
                try:
                    subprocess.run(
                        train_chunk_args + " -s "+ source_chunk + 
                        " --model_path " + trained_chunk +
                        " --bounds_file "+ source_chunk,
                        shell=True, check=True
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error executing train_single: {e}")
                    if not args.keep_running:
                        sys.exit(1)

                # Generate a hierarchy within each chunks
            print(f"Generating hierarchy for chunk {chunk_name}")
            try:
                subprocess.run(
                hierarchy_creator_args + " ".join([
                        os.path.join(trained_chunk, "point_cloud/iteration_30000/point_cloud.ply"),
                        source_chunk,
                        trained_chunk,
                        os.path.join(output_dir, "scaffold/point_cloud/iteration_30000")
                    ]),
                    shell=True, check=True, text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error executing hierarchy_creator: {e}")
                if not args.keep_running:
                    sys.exit(1)

            # Post optimization on each chunks
            print(f"post optimizing chunk {chunk_name}")
            try:
                subprocess.run(
                    post_opt_chunk_args + " -s "+ source_chunk + 
                    " --model_path " + trained_chunk +
                    " --hierarchy " + os.path.join(trained_chunk, "hierarchy.hier"),
                    shell=True, check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error executing train_post: {e}")
                if not args.keep_running:
                    sys.exit(1) # TODO: log where it fails and don't add it to the consolidation and add a warning at the end

    if args.use_slurm:
        # Check every 10 sec all the jobs status
        all_finished = False
        all_status = []
        last_count = 0
        print(f"Waiting for chunks to be trained in parallel ...")

        while not all_finished:
            # print("Checking status of all jobs...")
            all_status = [is_job_finished(id) for id in submitted_jobs_ids if is_job_finished(id) != ""]
            if last_count != all_status.count("COMPLETED"):
                last_count = all_status.count("COMPLETED")
                print(f"processed [{last_count} / {len(chunk_names)} chunks].")

            all_finished = len(all_status) == len(submitted_jobs_ids)
    
            if not all_finished:
                time.sleep(10)  # Wait before checking again
        
        if not all(status == "COMPLETED" for status in all_status):
            print("At least one job failed or was cancelled, check at error logs.")

    end_time = time.time()
    print(f"Successfully trained in {(end_time - start_time)/60.0} minutes.")

    ## Consolidation to create final hierarchy
    hierarchy_merger_path = "submodules/gaussianhierarchy/build/Release/GaussianHierarchyMerger.exe" if os_name == "Windows" else "submodules/gaussianhierarchy/build/GaussianHierarchyMerger"
    hierarchy_merger_path = os.path.join(f_path.parent.parent, hierarchy_merger_path)

    consolidation_args = [
        hierarchy_merger_path, f"{output_dir}/trained_chunks",
        "0", chunks_dir, f"{output_dir}/merged.hier" 
    ]
  
    consolidation_args = consolidation_args + chunk_names
    print(f"Consolidation...")
    if args.use_slurm:
        consolidation = submit_job(slurm_args + [
                f"--error={output_dir}/consolidation_log.err", f"--output={output_dir}/consolidation_log.out",
                "consolidate.slurm"] + consolidation_args)        

        while is_job_finished(consolidation) == "":
            time.sleep(10)
    else:
        try:
            subprocess.run(consolidation_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing consolidation: {e}")
            sys.exit(1)

    end_time = time.time()
    print(f"Total time elapsed for training and consolidation {(end_time - start_time)/60.0} minutes.")
