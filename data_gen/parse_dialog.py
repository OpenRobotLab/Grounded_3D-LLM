from data_utils import parse_phrase_ids
from tqdm import tqdm
import json

path = ["dialog/0_10_complete.json"]

with open("all_objects_by_scene.json",'r') as f:
    all_objects_by_scene = json.load(f)

data = []
for p in path:
    with open(p,'r') as f:
        data+=json.load(f)

# ================ collect statistics =================
        
print(f"collect {len(data)}")
all_scene = {}

for line in data:
    if line["scene"] not in all_scene:
        all_scene[line["scene"]]=1
    else:
        all_scene[line["scene"]]+=1

print(f"number of scenes used: {len(all_scene)}")
count = []
for k,v in all_scene.items():
    count.append(v)
print(f"dialogues pre-scene: {sum(count)/len(count)}")
print(f"total dialogues: {sum(count)}")

# ======================================================

out_json = []
unannotated = 0
from copy import deepcopy
for line in data:
    instance ={}
    dialog = ""
    instance["scene_id"] = line["scene"]
    instance["anwser"] = None
    if "[" not in " ".join(line["grounded_dialog"][0]) or len(line["grounded_dialog"][0])==1:
        unannotated+=1
        continue
    for idx,r in enumerate(line["grounded_dialog"][0]):
        if idx == 0:
            if r.startswith("ASSISTANT"): continue
            dialog+=r.replace("USER: ","")
            continue
        if "ASSISTANT:" in r:
            instance_to_add = deepcopy(instance)
            instance_to_add["anwser"] = r.replace("ASSISTANT: ","")
            instance_to_add["history_with_question"] = dialog
            out_json.append(instance_to_add)
        dialog+="\n"+r
            
print(f"filtered {unannotated}")

mark = []
for dia in tqdm(out_json):
    try:
        collected_objects = []
        for all_object_i in all_objects_by_scene[dia["scene_id"]]['object_list']:
            if all(obj not in all_object_i['object_name'] for obj in ["wall", "floor", "ceiling", "object"]):
                collected_objects.append(all_object_i['object_id'])
        flag = True
        
        question,pos_in_question = parse_phrase_ids(dia['history_with_question'])
        for obj in pos_in_question:
            if int(obj) not in collected_objects:
                flag = False
                continue
        answer,pos_in_answer = parse_phrase_ids(dia["anwser"])
        for obj in pos_in_answer:
            if int(obj) not in collected_objects:
                flag = False
                continue
        dia["anwser"] = answer
        dia["answer_positive"] = pos_in_answer
        dia['history_with_question'] = question
        dia['history_with_question_positive'] = pos_in_question
        if flag:
            mark.append(dia)
    except Exception as e:
        print(e)
        print(dia)

with open("embodied_dialog.json",'w') as f:
    json.dump(mark,f,indent=4)

