import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForSeq2SeqLM.from_pretrained('./finetuned_flan_t5-small').to(device)
tokenizer = AutoTokenizer.from_pretrained('./finetuned_flan_t5-small')

model.eval()

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)
    
test_data = read_jsonl('./input_data/attribute_test.data')

def preprocess_data(data):

    data['input_text'] = data.apply(lambda row: f"title: {row['title']} store: {row['store']} details_Manufacturer: {row['details_Manufacturer']}", axis=1)
    
    return data[['input_text']]
    
test_processed = preprocess_data(test_data)

test_dataset = Dataset.from_pandas(test_processed)

test_data = test_dataset['input_text']

def generate_text(inputs):
    inputs = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True, truncation=True, max_length=352) # 352)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts

def extract_details(text):
    pattern = r'details_Brand: (.*?) L0_category: (.*?) L1_category: (.*?) L2_category: (.*?) L3_category: (.*?) L4_category: (.*)'
    match = re.match(pattern, text)
    if match:
        return tuple(item if item is not None else 'na' for item in match.groups())
    return 'na', 'na', 'na', 'na', 'na', 'na'

def clean_repeated_patterns(text):
    cleaned_data = text.split(' L4_category')[0] 
    return cleaned_data


batch_size = 512
generated_details = []

for i in tqdm(range(0, len(test_data), batch_size), desc="Processing test data"):
    batch_inputs = test_data[i:i+batch_size]
    
    generated_texts = generate_text(batch_inputs)
    
    for generated_text in generated_texts:
        generated_details.append(extract_details(generated_text))

print('Generated info extracted.............')
    
    
categories = ['details_Brand', 'L0_category', 'L1_category', 'L2_category', 'L3_category', 'L4_category']

with open('./infer_results/attribute_test_finetuned_flan_t5-small.predict', 'w') as file:

    for indoml_id, details in enumerate(generated_details):
        result = {"indoml_id": indoml_id}
        for category, value in zip(categories, details):
            result[category] = value
        
        file.write(json.dumps(result) + '\n')
        
print('Results saved to ./infer_results/attribute_test_finetuned_flan_t5-small.predict')