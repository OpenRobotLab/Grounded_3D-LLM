import random
import json
from data_utils import generate_chat_completion,parallel_helper
import argparse
import os

global save_root,grounded_scene_caption,num_threads,num_of_samples


prompt = '''Given a local scene caption with object IDs, you are tasked with creating a JSON dictionary containing 2-4 high-level tasks for the robot assistant. Each task should include a clear, concise step-by-step plan using the objects identified in the scene. 
Rules:
1. Format your response as a JSON dictionary, where each key is a high-level task name and its value is a list of steps in the format of
```json
{"name of task1 ": ["step1", "step2", ...], ...}
```
2. Each step should be simple and under 10 words.
3. Each given object has its own unique object ID. You MUST refer to the given objects using the OBJ-ID format: "[<the/a/number> <object name (or descriptive phrase of the object)> <object ID(s)>]".
4. Do not add any objects or details not mentioned in the scene caption.
'''

max_num_of_class = 20
from tqdm import tqdm
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
        messages=[{"role": "system", "content": prompt },
                  {"role":"user", "content":f"Scene caption: {item['refined_caption']}"}]
        planning = None
        chat = None
        for _ in range(0,3):
            try:
                chat = generate_chat_completion(messages=messages)
                planning = json.loads(chat.strip("```json").strip("```"))
                list_planning = []
                for k, v in planning.items():
                    list_planning.append(dict(k=v))
                collect_data_one_scene["grounded_dialog"].append(planning)
                collect_data_one_scene["caption"] = item['refined_caption']
                break
            except Exception as e:
                print(chat)
                print(e)
                pass
        print(collect_data_one_scene["scene"])
        print(planning)
        out_json.append(collect_data_one_scene)
        counter+=1
        if counter % 50 == 0 or counter == 1:
            with open(save_root+f"/{rank}_{counter}.json",'w') as f:
                json.dump(out_json,f,indent=4)
    with open(save_root+f"/{rank}_{counter}_complete.json",'w') as f:
        json.dump(out_json,f,indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script for embodied planning")
    parser.add_argument("--save_root", type=str, default="./plan", help="Path to save save_root directory")
    parser.add_argument("--grounded_scene_caption", type=str, default="step2_captions_by_scene_v2.json", help="Path to grounded scene caption(by scene) file")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads to call api")
    parser.add_argument("--num_of_samples", type=int, default=10, help="Number of samples to generate")

    args = parser.parse_args()

    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    grounded_scene_caption = args.grounded_scene_caption
    num_threads = args.num_threads
    num_of_samples = args.num_of_samples

    with open("step2_captions_by_scene_v2.json",'r') as f:
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


