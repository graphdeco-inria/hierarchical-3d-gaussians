import argparse
import database
from read_write_model import read_model, CAMERA_MODEL_NAMES
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
        db.add_camera(CAMERA_MODEL_NAMES[cam.model].model_id, cam.width, cam.height, cam.params, camera_id=key)

    for key in images_metas:
        image_meta = images_metas[key]
        db.add_image(image_meta.name, image_meta.camera_id, image_id=key)

    db.commit()
    # shutil.copy(f"{args.in_dir}/cameras.txt", f"{args.out_dir}/cameras.txt")

    print(0)