import json
import argparse
import os
import numpy as np
import shutil
from plyfile import PlyData, PlyElement
from copy import deepcopy
from glob import glob
from utils import colorize_point_cloud, save_colored_ply, generate_colored_caption

parser = argparse.ArgumentParser(description='Visualize the grounded scene caption data.')
parser.add_argument('--datapath', default='../data/processed/scannet200')
parser.add_argument('--langpath', default='../data/langdata/groundedscenecaption_format.json', help='the json path of grounded scene caption')
parser.add_argument('--count', default=10, type=int, help='numbers of captions for visualizations')
parser.add_argument('--scene_id', default='scene0000_00', type=str, help='scene id for visualization')
args = parser.parse_args()

datapath = args.datapath
demo = json.load(open(args.langpath))
captions = []
caption_positives = []

count = args.count
for i in demo:
    if i["scene_id"] == args.scene_id and count > 0:
        captions.append(i["description"])
        id_pos = {}
        for interval,ids in zip(i["all_phrases_positions"],i["object_ids"]):
            for id in ids:
                if id not in id_pos:
                    id_pos[id]=[interval]
                else:
                    id_pos[id].append(interval)
        caption_positives.append(id_pos)
        count-=1

# =============================================================

point_cloud_paths = glob(f'{datapath}/*/*.npy')
for path in point_cloud_paths:
    if args.scene_id.replace("scene","") in path:
        points = np.load(path)
        break

coordinates, color, normals, segments, labels = (
              points[:, :3],
              points[:, 3:6],
              points[:, 6:9],
              points[:, 9],
              points[:, 10:12],
          )

# save raw point cloud color
save_colored_ply(np.concatenate((coordinates, color/255),axis = -1),'./visualizer/assets/scene_to_load/instance_raw.ply')

# save instance color
colored_vertices, instance_colors = colorize_point_cloud(coordinates, np.int32(labels[:,-1]))
save_colored_ply(colored_vertices, './visualizer/assets/scene_to_load/instance_color.ply')

combined_cap = ""
for idx,(caption,caption_positive) in enumerate(zip(captions,caption_positives)):
    html_caption = generate_colored_caption(deepcopy(caption), deepcopy(caption_positive), deepcopy(instance_colors))
    combined_cap+=f"caption: {idx+1} <br> "+html_caption+"<br>"

with open("./visualizer/assets/info.json",'w')as f:
    json.dump({
        "text":combined_cap
    },f,indent=4)

print('''
--------------------------- Visualization command --------------------------
cd visualizer; python -m http.server 7890
----------------------------------------------------------------
''')




