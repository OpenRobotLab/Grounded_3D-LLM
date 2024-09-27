## Environment Setups and Dataset Preparation

#### Step 1: Install the requirements in the terminal.
```
conda create -n grounded3dllm python=3.10.9
conda activate grounded3dllm

conda install openblas-devel -c anaconda
conda install openjdk=11

pip install -r requirements.txt

export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/mpc-0.8.1/lib:/mnt/petrelfs/share/gcc/mpfr-2.4.2/lib:/mnt/petrelfs/share/gcc/gmp-4.3.2/lib:/mnt/petrelfs/share/gcc/gcc-9.4.0/lib64:$LD_LIBRARY_PATH

pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install peft==0.8.2 --no-deps # ignore the pytorch version error 

cd third_party
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

cd ../pointnet2
python setup.py install
```

#### Step 2: Prepare the ScanNet dataset.

1. Download the ScanNet repository from the official ScanNet website.
ln -s ScanNet_dataset ./data/rawscannet

2. Get ScanNet official repo for pre-processing.
```
mkdir third_party
cd third_party
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make
```

3. Pre-process the scannet dataset.
```
# Requires Open3D library
python -m datasets.preprocessing.scannet_preprocessing preprocess \
--data_dir="./data/rawscannet" \
--save_dir="data/processed/scannet200" \
--git_repo="third_party/ScanNet" \
--scannet200=true
```

After preprocessing, the folder is organized as:
```
|-- 
|  |-- data
|     |-- rawscannet
|     |-- processed
|        |-- scannet200
```


