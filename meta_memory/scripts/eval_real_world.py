import json
import numpy as np

import re
import time
import sys
import os, sys

from dataclasses import asdict
import argparse

# load this directory
sys.path.append(sys.path[0] + '/..')

from agents.metamemory_agent import MetaMemoryAgent
from memory.memory import MemoryItem
from memory.milvus_memory import MilvusMemory


import time
current_time = time.localtime()
formatted_time = time.strftime("%y-%m-%d_%H:%M", current_time)


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)


def load_memory(args, start_time, end_time, ip_address='127.0.0.1'):
    reasoning_path = os.path.join(args.out_dir, 'reason', f'reasoning_{formatted_time}.txt')
    visual_prompt_path = os.path.join(args.out_dir, str(args.sequence_id), 'visual_prompt')
    memory = MilvusMemory("eval_memory", db_ip=ip_address, time_offset=start_time, save_path=visual_prompt_path, save_reasoning_path=reasoning_path)

    memory.reset()

    captions_path = os.path.join(args.data_dir, 'spacelocqa/real_world/captions', args.caption_file)
    with open(captions_path, 'r') as f:
        out = json.load(f)

    outputs = []

    # Compute start idx
    all_start_times = np.array([float(x['file_start'][:-4]) for x in out])
    diff = all_start_times - start_time
    start_idx = np.argmin(np.abs(diff))

    # Compute end idx
    all_end_times = np.array([float(x['file_end'][:-4]) for x in out])
    diff = all_end_times - end_time
    end_idx = np.argmin(np.abs(diff))

    for i in range(start_idx, end_idx+1):
        item = out[i]
        entity = {
            'position': item['position'],
            'theta': item['theta'], # ignoring rotation
            'caption': item['caption'],
            'images_path': item['images_path']
        }

        outputs.append(entity)

        entity = MemoryItem.from_dict(entity)
        memory.insert(entity, text_embedding=item['text_embedding'])

    return memory, outputs


def main(args):
    reasoning_path = os.path.join(args.out_dir, 'reason', f'reasoning_{formatted_time}.txt')
    os.makedirs(os.path.dirname(reasoning_path), exist_ok=True)
    agent = MetaMemoryAgent(temperature=args.temperature, save_reasoning_path=reasoning_path)

    # TODO: Set start_time and end_time to the timestamps of your collected data.
    start_time = 1752039473.892425
    end_time = 1752039628.782148

    memory, instance_captions = load_memory(args, start_time, end_time, ip_address=args.db_ip)
    agent.set_memory(memory)

    all_position = []
    with open(args.poses_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            row = [float(value) for value in line.strip().split()]
            all_position.append(tuple(row[1:3]))

    path_graph = memory._build_graph(all_position, 1.0)

    while True:
        question = input("Please input your query:")

        response = agent.query(question)
        predicted = asdict(response)
        if type(predicted['position']) == str:
            predicted['position'] = eval(predicted['position'])
        pred_pos = tuple(predicted['position'])

        waypoints, distance = memory._dijkstra(path_graph, (0, 0), pred_pos)

         # TODO: Send the waypoints as navigation goals to ROS. 
         # Please ensure that relocalization has been performed prior to this step—specifically, 
         # the robot's current coordinate frame must be aligned with the coordinate frame stored in Memory.

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='SpaceLocQA',
                        description='Runs LLMs on the QA dataset',)
    
    parser.add_argument("--qa_file", type=str, default="position_human_qa.json")
    parser.add_argument("--caption_file", type=str, default="captions_Qwen2.5-VL-7B_3_secs.json")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--poses_file_path", type=str, default="./data/spacelocqa/real_world/odometry.txt")
    parser.add_argument("--out_dir", type=str, default="./output/real_world")

    parser.add_argument("--temperature", type=float, default=0)

    parser.add_argument("--db_ip", type=str, default='127.0.0.1')

    args = parser.parse_args()
    main(args)