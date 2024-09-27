import os
from threading import Thread
from typing import Iterator
import pickle

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,LlamaTokenizer, AutoConfig
from models.LLM.LLama3d import LLama3dForCausalLM
from peft import LoraConfig, get_peft_model
import argparse
import glob

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_path' ,type=str, help="path to model dir")
args = parser.parse_args()

llm_config = os.path.join(args.model_path, 'config.json')
pretrained_weight = os.path.join(args.model_path, 'last-epoch.ckpt')

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 512
# MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

DESCRIPTION = """\
# Grounded 3D LLM

TODO: 

1. add streamer to generation

2. add scene to GUI

"""


assert torch.cuda.is_available(), "This demo does not work on CPU."

print('*****************************************************************')
print(f'Using config: {llm_config}')
llama_config = AutoConfig.from_pretrained(llm_config)
assert llama_config.vicuna_version == llama_config.vicuna_version, "conflict model"
print('*****************************************************************')

# init tokenizer and add special tokens
from models.LLM.LLama3d import load_llama_model_and_tokenizer
model, tokenizer = load_llama_model_and_tokenizer(llama_config)

with open(f"{pretrained_weight}",'rb')  as f:
    ckpt = torch.load(f)
state_dict = ckpt["state_dict"]
llama_weight = {key.split('.', 1)[1]: value for key, value in state_dict.items() if key.startswith('llama_model')}

print(model.load_state_dict(llama_weight,strict=False))
model.cuda()
model.eval()

def generate(
    message: str,
    chat_history: list,
    max_new_tokens: int = 1024,
    top_p: float = 0.9,
    scene_id: int = 1,
    repetition_penalty: float = 4.,
) -> Iterator[str]:

    eval_types = 'chat'

    # TODO: load necessary data here
    scene_feature_path_list = sorted(glob.glob(f'{args.model_path}/scene_features/*.bin'))
    feature = torch.load(scene_feature_path_list[int(scene_id-1)])
    instance_queries_hidden_state = feature["instance_queries_hidden_state"]
    instance_queries_normalized_embed = feature["instance_queries_normalized_embed"]
    print(chat_history)

    # streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    
    output = model.evaluate(
        input_text_list = [message],
        batch_instance_queries_hidden_state = [instance_queries_hidden_state],
        batch_instance_queries_normalized_embed = [instance_queries_normalized_embed],
        use_mini_batch=False,
        batch_eval_types =[eval_types],
        output_logits = True,
        text_only_output = True,
        max_new_tokens = max_new_tokens,
        top_p = top_p,
        repetition_penalty = float(repetition_penalty)
    ).replace("<","[").replace(">","]")

    return output

chat_interface = gr.ChatInterface(
    fn=generate,
    analytics_enabled=False,
    additional_inputs=[
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="scene id",
            minimum=1,
            maximum=6,
            step=1,
            value=1,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=5.0,
            step=0.05,
            value=4.,
        ),
    ],
    stop_btn=None,
    examples=[
        ["Describe this scene."],
        ["Can you fine all chairs in the scene?"],
        ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ],
)


with gr.Blocks(css="style.css", analytics_enabled=False) as demo:
    gr.Markdown(DESCRIPTION)
    # gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    chat_interface.render()

if __name__ == "__main__":
    demo.queue().launch(server_name='0.0.0.0')

