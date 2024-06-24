# Instructions for Visualizing Grounded Caption

1. **Download Processed ScanNet Data and Language Annotation**: 

   https://huggingface.co/datasets/ShuaiYang03/Grounded_3D_LLM_with_Referent_Tokens_dataset

2. **Run the Visualization Script**:
   ```sh
   python visualize_grounded_text.py --datapath ../data/processed/scannet200 --langpath ../data/langdata/groundedscenecaption_format.json --count 10 --scene_id scene0000_00
   ```

3. **Launch Visualizer Server**:
   ```sh
   cd visualizer; python -m http.server 7890
   ```
   
Access the visualized point cloud via your web browser at `http://localhost:7890`.

