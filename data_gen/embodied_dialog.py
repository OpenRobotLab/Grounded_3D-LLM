import random
import json
from tqdm import tqdm
from data_utils import generate_chat_completion,parallel_helper
import argparse
from copy import deepcopy
import os

global save_root,grounded_scene_caption,num_threads,num_of_samples

prompt = '''Given a local scene caption with object IDs, you are tasked with creating a dialogue with 3-5 rounds between a user and a robot assistant. The dialogue can include the following contents:
- {}
- {}
- {}
Rules:
1. You MUST refer to the given objects using the OBJ-ID format: "[<the/a/number> <object name (or descriptive phrase of the object)> <object id(s)>]". 
2. You MUST start each part of the dialogue with either "USER:" or "ASSISTANT:". Ensure there is a line break after each role exchange. 
3. Do not add any objects or details not mentioned in the scene caption.
4. The question and answer should be concise and natural. You can omit appearance details in the descriptive phrase and use only the object name in the OBJ-ID format.
'''

contents = ['User asks about the appearance(spatial relation, color, shape, material) or design purpose of objects.',
            'User engages in discussions regarding the layout of the local scene.',
            'User wants assistant to handle simple tasks related to scene objects and associated human activities.'
            ]

def process(raw_data, rank):
    counter = 0
    out_json = []
    print(len(raw_data))

    for item in tqdm(raw_data):
        collect_data_one_scene = {}
        collect_data_one_scene["scene"] = item['scene_id']
        collect_data_one_scene["grounded_dialog"] = []
        if not item["refined_caption"]:
            continue
        
        random.shuffle(contents)
        formated_prompt = deepcopy(prompt).format(contents[0],contents[1],contents[2])
        messages=[{"role": "system", "content":formated_prompt},
                  {"role":"user", "content":f"Grounded Caption: {item['refined_caption']}"}]

        chat = None
        for _ in range(0,3):
            try:
                chat = generate_chat_completion(messages=messages).replace("**","")
                dialogue_list = [line for line in chat .split('\n') if line.startswith("USER:") or line.startswith("ASSISTANT:")]
                collect_data_one_scene["grounded_dialog"].append(dialogue_list)
                collect_data_one_scene["caption"] = item['refined_caption']
                collect_data_one_scene["raw"] = chat
                break
            except Exception as e:
                print(e)
                pass
        print(item['refined_caption'])
        print('-------------------')
        print(chat)
        out_json.append(collect_data_one_scene)
        counter+=1
        if counter % 50 == 0 or counter == 1:
            with open(save_root+f"/{rank}_{counter}.json",'w') as f:
                json.dump(out_json,f,indent=4)
    with open(save_root+f"/{rank}_{counter}_complete.json",'w') as f:
        json.dump(out_json,f,indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script for embodied dialogue")
    parser.add_argument("--save_root", type=str, default="./data_gen/embodied_dialog", help="Path to save root directory")
    parser.add_argument("--grounded_scene_caption", type=str, default="./data_gen/step2_captions_by_scene_v2.json", help="Path to grounded scene caption(by scene) file")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads to call api")
    parser.add_argument("--num_of_samples", type=int, default=10, help="Number of samples to generate")

    args = parser.parse_args()

    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    
    grounded_scene_caption = args.grounded_scene_caption
    num_threads = args.num_threads
    num_of_samples = args.num_of_samples

    with open(grounded_scene_caption,'r') as f:
        data_list = json.load(f)

    all_data = []
    for k,v in data_list.items():
        all_data+=v
    
    data_list = all_data

    print(f"total captions {len(data_list)}")
    data_list = random.sample(data_list,min(num_of_samples,len(data_list)))
    print(f"sampled captions {len(data_list)}")

    parallel_helper(num_threads=num_threads,
                    data_list=data_list,
                    func=process)

