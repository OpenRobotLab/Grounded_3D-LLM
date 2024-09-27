# Grounded 3D-LLM with Referent Tokens

This repository will release the official implementation of "Grounded 3D-LLM with Referent Token".

> [[`Paper`]](https://arxiv.org/pdf/2405.10370) [[`Arxiv`]](https://arxiv.org/abs/2405.10370) [[`Website`]](https://groundedscenellm.github.io/grounded_3d-llm.github.io/) [[`Data`]](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata)

## Abstract

Prior studies on 3D scene comprehension have primarily developed specialized models for specific tasks or required task-specific fine-tuning. In this study, we propose *Grounded 3D-LLM*, which explores the potential of 3D large multi-modal models (LMMs) to consolidate various 3D visual tasks within a unified generative framework. The model utilizes a series of ``referent tokens'' to reference 3D scenes, enabling the handling of sequences that interleave 3D and textual data arbitrarily. 3D vision tasks are naturally transformed into language formats through task-specific prompts. To effectively associate the scene with text, we curate the grounded language datasets either from human-annotated sources or by bootstrapping existing object labels at the phrase level. We then employ Contrastive Language-Scene Pre-training (CLASP) to bridge the divide between 3D vision and language models, thus facilitating the use of referent tokens in subsequent language modeling tasks. Our comprehensive evaluation covers open-ended tasks like 3D visual question answering and dense captioning, as well as close-ended tasks such as object detection and language grounding. 

![image-20240515195822834](./README.assets/image-20240515195822834.png)

## Grounded Scene Caption Data Visualization and Generation

Please refer to the [data visualization page](./doc/data_vis.md) for detailed instructions on the minimal setup for visualizing the grounded scene caption dataset.

## Model Training

#### Step 1: Environment setups and dataset preparation.
Grounded 3D-LLM is trained using 4 or 8 NVIDIA Tesla A100 GPUs. Please refer to the [installation page](./doc/install.md) for detailed installation scripts for model training.

Please download all the scene-language datasets the from [HuggingFace](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/tree/main). The datasets are listed as follows:

|  Dataset | # for Train | # for Eval |
| :------: | :----: | :----: | 
| [ScanRefer](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata/scanrefer_format.json) |36639 | 9503 | 
| [Scan2Cap](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata/scanrefer_format.json) | 36639 | 9503 |
| [ScanQA](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata/scanqa_format.json) | 26516 | 9402 | 
| [Object-Description](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata/objectdescription_format.json) |  28197 | 7912 |
| [GroundedSceneCaption](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata/groundedscenecaption_format.json) |  84301 | -- |
| [EmbodiedPlanning](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata/embodiedplan_format.json) |  3500 | -- |
| [EmbodiedDialogue](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata/embodieddialog_format.json) |  129799 | -- |
| [GlobalSceneCaption](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata/global_scene_cap_format.json) | 4065 | -- |
| [3D-LLM](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata/3dllm_format.json) | 27627 | -- |
| [Alpaca](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/blob/main/langdata/alpaca_data.json) | 51865 | -- |

Please download the pretrained weights from [HuggingFace](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/tree/main/pretrained) and place them in `$ROOT_PATH/pretrained/`.

Please download the pretrained LLM weights ([Tiny-Vicuna-1B](https://huggingface.co/Jiayi-Pan/Tiny-Vicuna-1B)) and store them in `$ROOT_PATH/pretrained/llm_weight/Tiny-Vicuna-1B/`

If you would like to utilize our pretrained model checkpoints, they can be obtained from [HuggingFace](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/tree/main/saved/). Please save these in the checkpoint directory located at `$ROOT_PATH/saved`.

|  Steps  | Model Checkpoints  |
| :-------: | :------: |
| 1  |  [Mask3D-CLIP](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/tree/main/saved/step1_mask3d_clip_4GPUS)  | 
| 2  |  [Mask3D-CLASP](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/tree/main/saved/step2_mask3d_lang_4GPUS) | 
| 3  | [Grounded 3D-LLM](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data/tree/main/saved/step3_mask3d_lang_4GPUS) |  

After completing the downloads, the root folder should be organized as follows:

```
ROOT_PATH
├── data                            # data
│   ├── langdata
│   │   │── groundedscenecaption_format.json
│   │   │── scanrefer_format.json
│   │   │── scanqa_format.json
│   │   │── ...
│   ├── processed
│   │── rawscannet
│   │   │── scans
│   │   │── scannetv2-labels.combined.tsv
│── pretrained                      # pretrained weights for model training
│   │── bert-base-uncased           # bert pretrained weights
│   │── label_clip_features.pth     # clip's text features for scannet-200 class names
│   │── llm_weight
│   │   │── Tiny-Vicuna-1B          # pretrained weights from https://huggingface.co/Jiayi-Pan/Tiny-Vicuna-1B
│── saved                           # model checkpoints saved path
│   │── step1_mask3d_clip_4GPUS
│   │── step2_mask3d_lang_4GPUS
│   │── step3_mask3d_lang_4GPUS
```

#### Step 2: Pre-train the Mask3D detector:
```
bash final_scripts/step1_pretrain_detector.sh
```

#### Step 3:  After training the detector, pre-train the detector using Contrastive Language-Scene Pre-training:
```
bash final_scripts/step2_pretrain_3d-clasp.sh
```

#### Step 3: After contrastive pre-training, train the entire Grounded 3D-LLM:
```
bash final_scripts/step3_train_grounded3dllm.sh
```

The model checkpoints will be saved in `saved/step3_mask3d_lang_4GPUS/last-checkpoint.pth`, and the inference results will be stored in `saved/step3_mask3d_lang_4GPUS/${TIMESTAMP}/`.

## Model Evaluation

To evaluate all the respective results, run the following command:
```
bash final_scripts/test_llm.sh ./saved/step3_mask3d_lang_4GPUS/${TIMESTAMP}/
```

## Demo

To interact with Grounded 3D-LLM via the demo chat, first run the model inference and ensure that the `scene_features` are saved in `saved/step3_mask3d_lang_4GPUS/scene_features`. After that, launch the gradio demo chat by running the following command:
```
bash web_chat_demo/web_chat_demo.sh 
```
Please note that the visualization of the related segmentation masks is not yet supported in the Gradio demo.

## ToDo List

- [x] Release Grouded Scene Caption data (ScanNet).
- [x] Release data visualizer.
- [x] Release data generation code. 
- [x] Release pre-trained checkpoints.
- [x] Release Grounded 3D-LLM training and evaluation.
- [ ] Demo supports mask visualization.

## Acknowledgement
Many thanks to the following open-source projects:
* [Mask3D](https://github.com/JonasSchult/Mask3D)
* [OpenScene](https://github.com/pengsongyou/openscene)
* [LISA](https://github.com/dvlab-research/LISA)
* [LAVIS](https://github.com/salesforce/LAVIS/tree/main)
* [Vicuna](https://github.com/lm-sys/FastChat/tree/main)

