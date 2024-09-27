## Grounded Scene Caption Data Visualization and Generation

#### Step 1: Setup the minial environment for visualization.
```
pip install -r requirements_vis.txt
# Requires Open3D library to process the ScanNet dataset (e.g. open3d==0.9.0)
```

#### Step 2: Data preparation

You can download processed ScanNet(~7G) from [our Huggingface repository]( https://huggingface.co/datasets/ShuaiYang03/Grounded_3D_LLM_with_Referent_Tokens_dataset) or prepare it by yourself.

1. Download the ScanNet dataset from the official [ScanNet website](http://www.scan-net.org/).
```bash
ln -s ScanNet_dataset ./data/rawscannet
```

2. Get ScanNet official repo for pre-processing.
```bash
mkdir third_party
cd third_party
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make
```

3. Pre-process the scannet dataset as [Mask3D](https://github.com/JonasSchult/Mask3D).
```bash
python -m datasets.preprocessing.scannet_preprocessing preprocess \
--data_dir="./data/rawscannet" \
--save_dir="data/processed/scannet200" \
--git_repo="third_party/ScanNet" \
--scannet200=true
```

After preprocessing, please download the [grounded scene caption data](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155113995_link_cuhk_edu_hk/EpGS4c90LVVMvzio0UXgHfoB1u78-WpYaZfTuJj8qCbC4g?e=B2sufx) and put it into the data folder as:
```bash
|-- 
|  |-- data
|     |-- rawscannet
|     |-- processed
|        |-- scannet200
|  |-- langdata
|     |-- groundedscenecaption_format.json
```

#### Step 3: Data visualization.
1. Run the visualization script to generate colorful point clouds. 
```bash
cd data_visualization
python visualize_grounded_text.py --datapath ../data/processed/scannet200 --langpath ../data/langdata/groundedscenecaption_format.json --count 10 --scene_id scene0000_00
```
This command accepts raw point cloud data and language annotations, displaying 10 captions in the scene `scene0000_00`.

2. Visualize the grounded scene caption and the respective scene point clouds in the http server.
```bash
cd visualizer
python -m http.server 7890
```

## Instruction data generation for Embodied Dialogue and Planning
Please follow the README instructions in the data_gen/ folder to obtain embodied dialogue and planning data with grounding annotation.
