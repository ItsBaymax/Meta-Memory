import argparse
import os, os.path as osp

from PIL import Image
import numpy as np
import sys

# load this directory
sys.path.append(sys.path[0] + '/..')
from captioners.qwen_captioner import QwenCaptioner
from PIL import Image as PILImage

from langchain_huggingface import HuggingFaceEmbeddings
import glob
import json
import tqdm
import time


current_time = time.localtime()
formatted_time = time.strftime("%Y-%m-%d %H:%M", current_time)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run_video_in_segs(args):
    img_files = glob.glob(os.path.join("./data", "spacelocqa", args.seq_id, "camera", "*.png"))
    img_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))
    times = [float(x.split('/')[-1][:-4]) for x in img_files]

    poses_data = []
    with open(os.path.join("data", "spacelocqa", args.seq_id, "odometry.txt"), 'r') as file:
        lines = file.readlines()
        for line in lines:
            row = [float(value) for value in line.strip().split()]
            poses_data.append(row)

    segments = []
    current_segment = []
    time_start = times[0]
    for t, img_file, pose in zip(times, img_files, poses_data):
        file = {"image_path": img_file, "position": pose[1:4], "timestamp": pose[0]}
        if t - time_start > args.seconds_per_caption:
            # Then start over. Add the previous group. This item is the first of the new group
            segments.append(current_segment)
            current_segment = [file]
            time_start = t
        else:
            # Add current file to group
            current_segment.append(file)

    embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
    qwen_model = QwenCaptioner(model_name="Qwen/Qwen2.5-VL-7B-Instruct")

    captions_location = os.path.join("data", "spacelocqa", args.seq_id, "captions")
    os.makedirs(captions_location, exist_ok=True)

    outputs = []
    for i, file_names in tqdm.tqdm(enumerate(segments), total=len(segments)):  
        img_path = []
        images = []
        position = []
        timestamp = []

        for file in file_names:
            img_path.append(file["image_path"])
            img = np.array(Image.open(file["image_path"]))
            images.append(PILImage.fromarray(img.astype('uint8'), 'RGB'))
            position.append(file["position"])
            timestamp.append(file["timestamp"])
        
        position = np.array(position)
        timestamp = np.array(timestamp)

        # let's sample the images down to args.num_video_frames
        caption_images = img_path[::30//args.num_video_frames]

        if args.save_images:
            num_samples = 4
            total_images = len(images)
            if total_images <= num_samples:
                save_images = images
            else:
                indices = np.linspace(0, total_images - 1, num_samples, dtype=int)
                save_images = [images[i] for i in indices]

            out_images_path = os.path.join("data", "spacelocqa", args.seq_id, "images")
            os.makedirs(out_images_path, exist_ok=True)

            if save_images:
                img_width, img_height = save_images[0].size
            else:
                raise ValueError("The images list is empty.")
            
            new_img = Image.new('RGB', (img_width * 2, img_height * 2))
            for idx, img in enumerate(save_images):
                row = idx // 2
                col = idx % 2
                new_img.paste(img, (col * img_width, row * img_height))
            new_img = new_img.resize((img_width, img_height))
            new_img_path = os.path.join(out_images_path, f'sampled_images_{i}.png')
            new_img.save(new_img_path)

        
        out_text = qwen_model.video_caption(caption_images, args.query)[0]
        filename_start = os.path.basename(str(file_names[0]["timestamp"]))
        filename_end = os.path.basename(str(file_names[-1]["timestamp"]))

        text_embedding = embedder.embed_query(out_text)

        entity = {
            'id': file_names[0]["image_path"],
            'position': position.mean(axis=0),
            'theta': 3.14, # TEMPORARY: We are not using rotation information yet, so just leaving a placeholder
            'time': timestamp.mean(), # We are not using time information
            'caption': out_text,
            'file_start': filename_start,
            'file_end': filename_end,
            'text_embedding': text_embedding,
            'images_path': new_img_path
        }

        outputs.append(entity)

    # now save the outputs into a json
    with open(os.path.join(captions_location, f'captions_{args.captioner_name}_{args.seconds_per_caption}_secs.json'), 'w') as f:
        json.dump(outputs, f, cls=NumpyEncoder)


if __name__ == "__main__":

    default_query = "You are wandering around a university campus. Please provide a concise description of what you see in the few seconds of the video. \
Please output without line breaks, presenting it as a single continuous paragraph."

    parser = argparse.ArgumentParser()

    parser.add_argument("--seq_id", type=str, default="0")
    parser.add_argument("--captioner_name", type=str, default="Qwen2.5-VL-7B")
    parser.add_argument("--seconds_per_caption", type=int, default=3)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, default=default_query)
    parser.add_argument("--save_images", type=bool, default=True)
    args = parser.parse_args()

    run_video_in_segs(args)


