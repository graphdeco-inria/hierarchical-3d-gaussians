import argparse
import database
from read_write_model import read_model
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--database_path', required=True)
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)
        
    cam_intrinsics, images_metas, _, = read_model(args.in_dir, ".bin")
    db = database.COLMAPDatabase.connect(args.database_path)
    db.create_tables()

    for key in cam_intrinsics:
        cam = cam_intrinsics[key]
        ## 1 pinhole, 5 OPENCV_FISHEYE, 3 ??, 4 OPENCV
        db.add_camera(1, cam.width, cam.height, cam.params, camera_id=key)

    for key in images_metas:
        image_meta = images_metas[key]
        db.add_image(image_meta.name, image_meta.camera_id, image_id=key)

    db.commit()
    # shutil.copy(f"{args.in_dir}/cameras.txt", f"{args.out_dir}/cameras.txt")

    print(0)