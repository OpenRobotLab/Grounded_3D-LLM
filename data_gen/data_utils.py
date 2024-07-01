import spacy
import json
import requests
import json
from time import sleep
import threading

# ===================== your api key ===========================
API_KEY = 'your key'
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
# ==============================================================

nlp = spacy.load("en_core_web_sm")
def find_nearest_phrase(text, index):
    doc = nlp(text[:index])
    nearest_phrase = ""
    for chunk in doc.noun_chunks:
        nearest_phrase = chunk.text
    if not nearest_phrase:
        for token in doc:
            if token.pos_ in ['VERB', 'NOUN']:
                nearest_phrase = token.text
    return nearest_phrase

def parse_phrase_ids(text):
    stack = []
    id_positions = {} 
    clean_text = ""
    i = 0
    is_isolate = False
    for index, char in enumerate(text):
        if char == "[":
            stack.append((index, i))
        elif char == "]":
            start, clean_start = stack.pop()
            phrase_and_ids = text[start + 1:index]
            if phrase_and_ids.isdigit():
                nearest_phrase = find_nearest_phrase(clean_text, clean_start)
                ids = phrase_and_ids
                phrase = nearest_phrase
                is_isolate = True
            else:
                for last_char_index in range(len(phrase_and_ids) - 1, -1, -1):
                    if phrase_and_ids[last_char_index].isalpha():
                        break
                is_isolate = False
                phrase = phrase_and_ids[:last_char_index + 1]
                ids = phrase_and_ids[last_char_index + 2:]
            if not is_isolate:
                phrase_start = i
                clean_text += phrase  
                i += len(phrase)
                phrase_end = i
            else: # do not modify original text
                phrase_end = i-1
                phrase_start = i-len(phrase)-1
            for id in ids.split(","):
                id = id.strip()
                if id:
                    id = [char for char in id if char.isdigit()]
                    id = int(''.join(id)) if id else -1
                    if id not in id_positions:
                        id_positions[id] = []
                        
                    id_positions[id].append([phrase_start, phrase_end])
        else:
            if not stack:
                clean_text += char
                i += 1
    return clean_text, id_positions

def generate_chat_completion(messages, model="gpt-4o-2024-05-13", temperature=1, max_tokens=None):# gpt-3.5-turbo-0125 gpt-4-1106-preview
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    try:
        response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
    except:
        sleep(20)
        return generate_chat_completion(messages, 
                                        model=model, 
                                        temperature=temperature, 
                                        max_tokens=max_tokens)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(response.status_code)
        sleep(20)
        return generate_chat_completion(messages, 
                                        model=model, 
                                        temperature=temperature, 
                                        max_tokens=max_tokens)
    
def parallel_helper(num_threads,data_list,func):
    segment_size = len(data_list) // num_threads
    threads = []
    for i in range(num_threads):
        start = i * segment_size
        end = start + segment_size if i < num_threads - 1 else len(data_list)
        thread = threading.Thread(target=func, args=(data_list[start:end], i,))
        threads.append(thread)
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()

