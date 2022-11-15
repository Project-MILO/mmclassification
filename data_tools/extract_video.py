import os
from pathlib import Path


def extract(video_path, image_path, video_name):
    command = f"ffmpeg -i {video_path} -vf \"fps=2\" -qscale:v 2 {image_path}/{video_name}_%03d.png"
    os.system(command)


def main(in_path, out_path):
    video_paths = in_path.glob("*")
    for vid_path in video_paths:
        if ".mp4" not in str(vid_path):
            # TODO: Handle rotate videos
            continue
        vid_name = vid_path.name.replace(".mp4", "")
        image_path = out_path / vid_name
        print("Handle video path: ", vid_path)
        print("Handle save image path: ", image_path)
        image_path.mkdir(parents=True, exist_ok=True)
        extract(str(vid_path), image_path, vid_name)


if __name__ == "__main__":
    # in_path = Path("dataset/train_val_videos/val/fake")
    # in_path = Path("dataset/train/real/rotate")
    # in_path = Path("dataset/train/real")
    in_path = Path('/home/kiennt54/data/milo/liveness/public_test_2/videos')
    # out_path = Path("dataset/train_images_5fps/fake")
    # out_path = Path("dataset/train_images_5fps/real/rotate")
    out_path = Path('data/liveness_2fps/public_test')
    main(in_path, out_path)
