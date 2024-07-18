import os, sys
import subprocess
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True)
    parser.add_argument('--images_dir', default="", help="Will be set to project_dir/camera_calibration/rectified/images if not set")
    parser.add_argument('--chunks_dir', default="", help="Will be set to project_dir/camera_calibration/chunks if not set")
    parser.add_argument('--depth_generator', default="Depth-Anything-V2", choices=["DPT", "Depth-Anything-V2"], help="depth generator can be DPT or Depth-Anything-V2, we suggest using Depth-Anything-V2.")
    args = parser.parse_args()
    
    if args.images_dir == "":
        args.images_dir = os.path.join(args.project_dir, "camera_calibration/rectified/images")

    if args.chunks_dir == "":
        args.chunks_dir = os.path.join(args.project_dir, "camera_calibration/chunks")

    print(f"generating depth maps using {args.depth_generator}.")
    start_time = time.time()

    # Generate depth maps
    generator_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "submodules", args.depth_generator)
    
    if args.depth_generator == "DPT":
        base_generator_args = [
            "python", f"{generator_dir}/run_monodepth.py",
            "-t", "dpt_large"
        ]
    else:
        base_generator_args = [
            "python", f"{generator_dir}/run.py",
            "--encoder", "vitl", "--pred-only", "--grayscale"
        ]

    images_dir = os.path.join(args.project_dir, "camera_calibration/rectified", "images")
    cam_dirs = os.listdir(images_dir)
    if all(os.path.isfile(os.path.join(images_dir, cam_dir)) for cam_dir in cam_dirs):
        cam_dirs = [""]
    for cam_dir in cam_dirs:
        full_cam_path = os.path.join(images_dir, cam_dir)
        print(f"Estimating depth for {full_cam_path}")
        full_depth_path = os.path.join(args.project_dir, "camera_calibration/rectified", "depths", cam_dir)
        if not os.path.isabs(full_cam_path):
            full_cam_path = os.path.join("../../", full_cam_path)
        if not os.path.isabs(full_depth_path):
            full_depth_path = os.path.join("../../", full_depth_path)
        os.makedirs(full_depth_path, exist_ok=True)
        if args.depth_generator == "DPT":
            generator_args = base_generator_args + [
                "-i", full_cam_path,
                "-o", full_depth_path
            ] 
        else:
            generator_args = base_generator_args + [
                "--img-path", full_cam_path,
                "--outdir", full_depth_path
            ] 
        try:
            subprocess.run(generator_args, check=True, cwd=generator_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error executing run_monodepth: {e}")
            sys.exit(1)

    # generate depth_params.json for each chunks
    print(f"generating depth_params.json for chunks {os.listdir(args.chunks_dir)}.")
    try:
        subprocess.run([
            "python", "preprocess/make_chunks_depth_scale.py", "--chunks_dir", f"{args.chunks_dir}", "--depths_dir", f"{os.path.join(args.project_dir, "camera_calibration/rectified", "depths")}"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing run_monodepth: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"Monocular depth estimation done in {(end_time - start_time)/60.0} minutes.")