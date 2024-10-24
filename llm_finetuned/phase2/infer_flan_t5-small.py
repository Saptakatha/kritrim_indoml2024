import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
import json

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

model = AutoModelForSeq2SeqLM.from_pretrained('./fine_tuned_flan_t5-small_60_es10_stratified_va_es10').to(device)
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_flan_t5-small_60_es10_stratified_va_es10')

model.eval()

def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)
    
# test_data = read_jsonl('./input_data/attribute_test.data')
# test_data = read_jsonl('./input_data/phase_2_test_set1.features')
test_data = read_jsonl('./input_data/final_test_data.features')

def preprocess_data(data):

    data['input_text'] = data.apply(lambda row: f"description: {row['description']} retailer: {row['retailer']}  price: {row['price']}", axis=1) # 
    
    return data[['input_text']]
    
test_processed = preprocess_data(test_data)

test_dataset = Dataset.from_pandas(test_processed)

test_data = test_dataset['input_text']

def generate_text(inputs):
    inputs = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True, truncation=True, max_length=64) # 352)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64)
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts

def extract_details(text):
    pattern = r'supergroup: (.*?) group: (.*?) module: (.*?) brand: (.*)'
    match = re.match(pattern, text)
    if match:
        return tuple(item if item is not None else 'na' for item in match.groups())
    return 'na', 'na', 'na', 'na'

# def clean_repeated_patterns(text):
#     cleaned_data = text.split(' brand')[0] 
#     return cleaned_data


batch_size = 1024
generated_details = []

for i in tqdm(range(0, len(test_data), batch_size), desc="Processing test data"):
    batch_inputs = test_data[i:i+batch_size]
    
    generated_texts = generate_text(batch_inputs)
    
    for generated_text in generated_texts:
        generated_details.append(extract_details(generated_text))

print('Generated info extracted.............')
    
    
categories = ['supergroup', 'group', 'module', 'brand']

with open('./infer_results_final/test_fine_tuned_flan_t5-small_60_es10_stratified_va_es10.predict', 'w') as file:

    for indoml_id, details in enumerate(generated_details):
        result = {"indoml_id": indoml_id}
        for category, value in zip(categories, details):
            result[category] = value
        
        file.write(json.dumps(result) + '\n')
        
        
# import zipfile

# file_to_zip = 'attribute_test_finetuned_flan_t5-small.predict'
# zip_file_name = './submissions/sapta_flan_t5-small_sub003.zip'

# with zipfile.ZipFile(zip_file_name, 'w') as zipf:
#     zipf.write(file_to_zip, arcname=file_to_zip)