# Instructions for Visualizing Grounded Caption

1. **Customize Data Path**: Update the root directory path in `visualize_grounded_text.py` to point to your ScanNet raw data location and path to captions:
   ```sh
   mkdir data
   ln -s RAW_SCANNET/ data/rawscannet
   ```
   
2. **Run the Visualization Script**:
   ```sh
   python visualize_grounded_text.py --datapath ../data/rawscannet/ --langpath ../data/langdata/groundedscenecaption_format.json --count 10 --scene_id scene0000_00
   ```

3. **Launch Visualizer Server**:
   ```sh
   cd visualizer; python -m http.server 7890
   ```
   
Access the visualized point cloud via your web browser at `http://localhost:7890`.

