# Instruction Data Generation for Embodied Dialogue and Planning

## 1. Prepare raw Grounded Scene Caption dataset.

Download the necessary raw processed data files and place them in the data_gen/data_gen folder:

- [all_objects_by_scene.json](https://huggingface.co/datasets/ShuaiYang03/Grounded_3D_LLM_with_Referent_Tokens_Dataset/blob/main/raw_langdata/all_objects_by_scene.json): All object information (with object-level caption) in each scene.
- [step2_captions_by_scene_v2.json](https://huggingface.co/datasets/ShuaiYang03/Grounded_3D_LLM_with_Referent_Tokens_Dataset/blob/main/raw_langdata/step2_captions_by_scene_v2.json): The raw step-by-step generated scene captions for groundedscenecaption_format.json.

## 2. Configure your OpenAI API key and model.

Install the required NLP library *spacy*:
```sh
python3 -m spacy download en_core_web_sm 
```

In `data_utils.py`, fill in your OpenAI API key:

```python
# ===================== Your API Key ===========================
API_KEY = 'your_key'
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
# ==============================================================
```

The default model is `gpt-4o-2024-05-13`. If you wish to change the model, be aware that we cannot guarantee the prompts will work properly with other models.

```python
def generate_chat_completion(messages, model="gpt-4o-2024-05-13", temperature=1, max_tokens=None):
```

## 3. Instruction data generation

You can specify the number of threads for API calls using `--num_threads 10` and the number of samples you want to generate using `--num_of_samples 10`.

To generate 10 samples for each and parse the output, run the following commands:

```sh
python embodied_dialog.py
python parse_dialog.py
python embodied_planning.py
python parse_planning.py
```

After parsing the generated data, all positives are organized in the following format:

```json
"answer_positive": {
    "6": [
        [0, 39]
    ],
    "12": [
        [46, 54]
    ]
}
```

This format represents object IDs with their respective start and end char indices.
