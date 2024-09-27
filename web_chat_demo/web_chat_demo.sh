export OMP_NUM_THREADS=12
export PYTHONPATH="./"

python ./web_chat_demo/web_chat_demo.py\
    --model ./saved/step3_mask3d_lang_4GPUS
