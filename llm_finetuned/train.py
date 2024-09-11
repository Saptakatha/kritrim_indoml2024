import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset, DatasetDict
import numpy as np


def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)


train_data = read_jsonl('./input_data/attribute_train.data')
train_solution = read_jsonl('./input_data/attribute_train.solution')
val_data = read_jsonl('./input_data/attribute_val.data')
val_solution = read_jsonl('./input_data/attribute_val.solution')


def preprocess_data(data, solution):
    merged = pd.merge(data, solution, on='indoml_id')

    merged['input_text'] = merged.apply(lambda row: f"title: {row['title']} store: {row['store']} details_Manufacturer: {row['details_Manufacturer']}", axis=1)
    merged['target_text'] = merged.apply(lambda row: f"details_Brand: {row['details_Brand']} L0_category: {row['L0_category']} L1_category: {row['L1_category']} L2_category: {row['L2_category']} L3_category: {row['L3_category']} L4_category: {row['L4_category']}", axis=1)
    
    return merged[['input_text', 'target_text']]


train_processed = preprocess_data(train_data, train_solution)
val_processed = preprocess_data(val_data, val_solution)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_processed)
val_dataset = Dataset.from_pandas(val_processed)


dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=352, padding='max_length', truncation=True)
    labels = tokenizer(targets, max_length=128, padding='max_length', truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)


tokenized_datasets.save_to_disk('./flan_t5-small-tokenized_dataset')


training_args = TrainingArguments(
    output_dir='./results_flan_t5-small',
    evaluation_strategy='epoch',
    learning_rate=2e-3,
    per_device_train_batch_size=96,
    per_device_eval_batch_size=96,
    num_train_epochs=60,
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir='./logs_flan_t5-small',
    logging_steps=100,
    report_to='none'
)


class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step: {state.global_step}")
            for key, value in logs.items():
                print(f"{key}: {value}")
            print("\n")
            
            
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    callbacks=[CustomCallback()]
)

trainer.train()


val_results = trainer.evaluate(eval_dataset=tokenized_datasets['validation'])
print(f"Validation Loss: {val_results['eval_loss']}")

# test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
# print(f"Test Loss: {test_results['eval_loss']}")


model.save_pretrained('./finetuned_flan_t5-small')
tokenizer.save_pretrained('./finetuned_flan_t5-small')