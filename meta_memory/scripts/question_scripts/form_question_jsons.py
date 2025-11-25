import json
import pandas as pd
import glob
import os
from time import strftime, localtime
import numpy as np

DATA_CSV = "./data/spacelocqa/spacelocqa.csv"
CAPTIONS_PATH = './data/spacelocqa/{seq_id}/captions/captions_Qwen2.5-VL-7B_3_secs.json'


def format_docs(doc):
    out_string = ""
    s = f"The robot was at an average position of {np.array(doc['position']).round(3).tolist()}."
    s += f"The robot saw the following: {doc['caption']}\n\n"
    out_string += s
    return out_string


def parse_answer(answer, context):

    q_type = answer['Type (binary, position, time, text)']
    text_answer = answer['Text answer']
    parsable_answer = answer['Parsable answer']

    out_dict = None
    out_dict = {'position': context['position']}
    
    # just in case things don't parse
    if out_dict is None:
        print("NOT EVERYTHING WAS PARSED CORRECTLY POSSIBLY!")
        print("Filling in un-parsable out_dict")
        out_dict = {
            'text': [text_answer],
            q_type: parsable_answer
        }
        
    return out_dict


# We read from the data.csv, parse the true info from the human generation, then create a new qa.json file
# 1 per sequence!
data = pd.read_csv(DATA_CSV)
files = glob.glob(os.path.join('./data', 'spacelocqa', '*', 'spacelocqa_unfilled.json'))
seq_ids = [int(x.split('/')[-2]) for x in files]

for i, seq_id in enumerate(seq_ids):
    print("On SeqID", seq_id)
    all_questions = [] # this is similar to how we create the new json

    # Load the json
    with open(files[i], 'r') as f:
        unfilled_qa = json.load(f)['data']
    try:
        with open(CAPTIONS_PATH.format(seq_id = seq_id)) as f:
            captions = json.load(f)
    except:
        print(f"ERROR. Questions for {seq_id} exists, however, captions do not exist. Will skip SeqID {seq_id}")
        continue

    # get the specific subset
    subset_df = data[(data["Seq ID"] == seq_id) & (data["Question"] != "") & (data['Question'].notna())]

    if len(subset_df) == 0:
        continue
    
    for qa_pair in unfilled_qa: 
        # qa_pair has keys: id, length_category, length, start_time, end_time, file_info={qa_start_filename, qa_end_filename}

        id = qa_pair['id']
        answers = subset_df[subset_df['UUID'] == id]

        caption_start_ids = [item['id'] for item in captions]
        caption_start_ids.sort(key=lambda x: float(x.split('/')[-1][:-4])) # should already be sorted
        caption_times = np.array([float(file.split('/')[-1][:-4]) for file in caption_start_ids])

        # note that there *could* be multiple answers per clip
        for _, answer in answers.iterrows():
            filled_qa = qa_pair.copy()
            text_answer_timestamp = answer['Timestamp with answer']
            question = answer['Question']
            q_type = answer['Type (binary, position, time, text)']
            q_category = answer['Question Category']

            # 1. Need to parse timestamp into raw time with a 3-sec before and after to get context_start_filename and context_end_filename
            # 2. Need to parse position answers
            # 3. Need to parse [minutes] ago into actual answer

            diff = caption_times - float(text_answer_timestamp)
            caption_idx = np.argmax(diff > 0) - 1

            context = format_docs(captions[caption_idx])
            context_starts = captions[caption_idx]['file_start']
            context_ends = captions[caption_idx]['file_end']
            
            current_time = localtime(filled_qa['end_time'])
            current_time = strftime('%Y-%m-%d %H:%M:%S', current_time)   
            start_time = localtime(filled_qa['start_time'])
            start_time = strftime('%Y-%m-%d %H:%M:%S', start_time) 

            # Fill in filled_qa properly
            filled_qa['question'] = question
            filled_qa['type'] = q_type
            filled_qa['context'] = context
            filled_qa['file_info']['context_start_filename'] = context_starts
            filled_qa['file_info']['context_end_filename'] = context_ends

            parsed_answer = parse_answer(answer, captions[caption_idx])
            filled_qa['answers'] = parsed_answer

            all_questions.append(filled_qa)


    # save all_questions into json
    out_json = {
        "version": 0.1,
        "data": all_questions
    }

    print(f"Saving data for sequence {seq_id} in ./data/spacelocqa/{seq_id}/questions/position_human_qa.json")
    # make dir if it does not exist
    
    os.makedirs(f'./data/spacelocqa/{seq_id}/questions', exist_ok=True)

    with open(f'./data/spacelocqa/{seq_id}/questions/position_human_qa.json', 'w') as f:
        # to_save = json.dumps(out_json, indent=4)
        json.dump(out_json, f, indent=4)
