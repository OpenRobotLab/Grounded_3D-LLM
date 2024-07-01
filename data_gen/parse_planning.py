from data_utils import parse_phrase_ids
from tqdm import tqdm
import json

path = ["./data_gen/embodied_planning/0_10_complete.json"]

with open("./data_gen/all_objects_by_scene.json",'r') as f:
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
        all_scene[line["scene"]]=len(list(line["grounded_dialog"][0].keys()))
    else:
        all_scene[line["scene"]]+=len(list(line["grounded_dialog"][0].keys()))
print(len(all_scene))
count = []
for k,v in all_scene.items():
    count.append(v)
print(sum(count)/len(count))
print(sum(count))

# ======================================================

out_json = {}
for scene in tqdm(data):
    grounded_plan = {}
    collected_objects = []
    for all_object_i in all_objects_by_scene[scene['scene']]['object_list']:
        if all(obj not in all_object_i['object_name'] for obj in ["wall", "floor", "ceiling", "object"]):
            collected_objects.append(all_object_i['object_id'])
    for p in scene["grounded_dialog"][0]:
        grounded_plan[p] = []
        if "[" in p:
            continue
        combined_lan = ""
        flag = True
        for idx,step in enumerate(scene["grounded_dialog"][0][p]):
            try:
                lan,pos = parse_phrase_ids(step)
            except Exception as e:
                print(e)
                continue
            for obj in pos:
                if obj<0:
                    print(scene,pos,collected_objects)
                if int(obj) not in collected_objects:
                    flag = False
                    continue
            if not flag: continue
            if len(lan)<5: continue
            combined_lan+=f"step {idx+1}. {step}\n"
        if flag and len(combined_lan)>5:
            lan,pos = parse_phrase_ids(combined_lan)
            grounded_plan[p].append({
                "desc":lan,
                "pos":pos
            })
    if scene['scene'] not in out_json:
        if len(list(grounded_plan.keys())):
            out_json[scene['scene']] = {}
            out_json[scene['scene']]["plan"] = grounded_plan
            out_json[scene['scene']]["instances"] = all_objects_by_scene[scene['scene']]['object_list']

print(len(list(out_json.keys())))

count = 0
for k in out_json:
    count+=len(list(out_json[k]["plan"].keys()))

print(f"number of plans: {count}")

with open("./data_gen/embodied_planning.json",'w') as f:
    data = json.dump(out_json,f,indent=4)