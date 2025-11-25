import json
import numpy as np
import tqdm

import re
import time
import sys
import os, sys

from dataclasses import asdict
import argparse
import traceback 

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

# we can have binary, position-based, time-based, or description-based. let's answer accordingly
def evaluate_output(qa_instance, predicted):

    out_error = {}
    q_type = qa_instance['type']
    if 'position' in q_type:
        answer = np.array(qa_instance['answers']['position'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['position']) == str:
            predicted['position'] = eval(predicted['position'])
        pred_pos = np.array(predicted['position'])

        dist = np.linalg.norm(answer - pred_pos)

        out_error['position_error'] = dist
    else:
        raise Exception("We do not support question type " + q_type)

    return out_error


def answer_squad_question(model, question, qa_instance):
    print(f'Question: {question}')

    parsed = None
    while True:
        try:
            start_time = time.time()
            response = model.query(question)
            end_time = time.time()

            elapsed = end_time - start_time

            parsed = asdict(response)

            out_error = evaluate_output(qa_instance, parsed)
            print("Time elapsed", elapsed)

        except Exception as e:
            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info()) 
            continue

        return_dict = {"response": parsed}
        return_dict.update(parsed)
        return_dict['error'] = out_error
        return_dict['elapsed'] = elapsed

        return return_dict


def load_memory(args, qa_instance, ip_address='127.0.0.1'):
    # Here we load everything needed to load a MilvusDB instance neatly
    start_time = qa_instance['start_time']
    end_time = qa_instance['end_time']

    reasoning_path = os.path.join(args.out_dir, str(args.sequence_id), 'reason', f'reasoning_{formatted_time}.txt')
    visual_prompt_path = os.path.join(args.out_dir, str(args.sequence_id), 'visual_prompt')
    memory = MilvusMemory(f"eval_memory_{args.sequence_id}", db_ip=ip_address, time_offset=start_time, save_path=visual_prompt_path, save_reasoning_path=reasoning_path)

    memory.reset()

    captions_path = os.path.join(args.data_dir, 'spacelocqa', str(args.sequence_id), 'captions', args.caption_file)
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
    reasoning_path = os.path.join(args.out_dir, str(args.sequence_id), 'reason', f'reasoning_{formatted_time}.txt')
    os.makedirs(os.path.dirname(reasoning_path), exist_ok=True)
    agent = MetaMemoryAgent(temperature=args.temperature, save_reasoning_path=reasoning_path)

    data_path = os.path.join(args.data_dir, 'spacelocqa', str(args.sequence_id), 'questions',  args.qa_file)
    data = json.load(open(data_path, 'r'))
    data = data['data']

    running_pos_error = 0
    num_position = 0
    basci_15 = 0
    local_15 = 0
    global_15 = 0
    basci_error = []
    local_error = []
    global_error = []
    
    responses = []
    for i in tqdm.tqdm(range(0, len(data)), total=len(data)):
        print(f"Evaluating {i} out of {len(data)}")

        qa_instance = data[i]
        question = qa_instance['question']
        id = qa_instance['id']

        if qa_instance['type'] != 'position':
            raise ValueError("Invalid qa_instance type. Expected 'position'.")

        ## load memory
        memory, instance_captions = load_memory(args, data[i], ip_address=args.db_ip)
        if len(instance_captions) == 0: # ISSUE
            print("Length of Instance Captions is 0. It should not be")
            import pdb; pdb.set_trace()

        print("HISTORY LENGTH", len(instance_captions))

        agent.set_memory(memory)

        out_dict = answer_squad_question(agent, question, qa_instance)

        out_dict['question'] = qa_instance['question']
        out_dict['id'] = id
        error_dict = out_dict['error']

        # keep track of how many of each. usually all CSVs are one type only
        num_position += 1
        running_pos_error += error_dict['position_error']

        if error_dict['position_error'] < 15:
            if qa_instance['category'] == "Basic":
                basci_15 += 1
                basci_error.append(error_dict['position_error'])
            elif qa_instance['category'] == "Local":
                local_15 += 1
                local_error.append(error_dict['position_error'])
            elif qa_instance['category'] == "Global":
                global_15 += 1
                global_error.append(error_dict['position_error'])

        print("Question:", question)
        if 'response' in out_dict:
            print("Response:", out_dict['response'])

        out_dict['result'] = {"Current Spatial Error": error_dict['position_error'],
                            "basci_15": basci_15,
                            "local_15": local_15,
                            "global_15": global_15,
                            "Average Spatial Error": running_pos_error/num_position
                             }
        
        print("Average Positional Error", running_pos_error/num_position)
        print("Current Positional Error", error_dict['position_error'])
        print()

        with open(reasoning_path, 'a') as f:
            f.write("Question:" + question + "\n")
            f.write("Response:" + str(out_dict['response']) + "\n")
            json.dump(out_dict['result'], f)
            f.write("\n--------------------------------------------------------------------------------------\n\n")
        responses.append(out_dict)


    def calculate_sr_spe(all_error: list):
        sum_value = 0
        sr_15 = 0
        for i in all_error:
            sum_value += 1-(i/15)
            if i <= 15:
                sr_15 += 1

        sr_15 = sr_15/15
        spe = sum_value/15
        
        return sr_15, spe
    
    basci_sr_15, basci_spe = calculate_sr_spe(basci_error)
    local_sr_15, local_spe = calculate_sr_spe(local_error)
    global_sr_15, global_spe = calculate_sr_spe(global_error)

    final_out_dict = {}
    final_out_dict['final_result'] = {"basci_15": basci_15,
                                      "local_15": local_15,
                                      "global_15": global_15,
                                      "Average Spatial Error": running_pos_error/num_position,
                                      "basci_sr_15": basci_sr_15,
                                      "basci_spe": basci_spe,
                                      "local_sr_15": local_sr_15,
                                      "local_spe": local_spe,
                                      "global_sr_15": global_sr_15,
                                      "global_spe": global_spe
                                      }
    responses.append(final_out_dict)
    # save all_questions into json
    out_json = {
        "version": 0.1,
        "responses": responses
    }

    # save the outputs
    out_path = os.path.join(args.out_dir, str(args.sequence_id), 'result')
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, f'{formatted_time}.json'), 'w') as f:
        # to_save = json.dumps(out_json, indent=4)
        json.dump(out_json, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='SpaceLocQA',
                        description='Runs LLMs on the QA dataset',)
    
    parser.add_argument("--sequence_id", type=int, default=0)
    parser.add_argument("--qa_file", type=str, default="position_human_qa.json")
    parser.add_argument("--caption_file", type=str, default="captions_Qwen2.5-VL-7B_3_secs.json")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./output")

    parser.add_argument("--temperature", type=float, default=0)

    parser.add_argument("--db_ip", type=str, default='127.0.0.1')

    args = parser.parse_args()
    main(args)