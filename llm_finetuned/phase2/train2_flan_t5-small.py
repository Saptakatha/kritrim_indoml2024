import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
import numpy as np
import logging 
import os


def read_jsonl(file_path, nrows=None):
    return pd.read_json(file_path, lines=True, nrows=nrows)


train_data = read_jsonl('./input_data/attribute_train_stratified.data')
train_solution = read_jsonl('./input_data/attribute_train_stratified.solution')
# test_data = read_jsonl('./data/attribute_test.data')
# test_solution = read_jsonl('./data/attribute_test.solution')
val_data = read_jsonl('./input_data/attribute_val_stratified.data')
val_solution = read_jsonl('./input_data/attribute_val_stratified.solution')


def preprocess_data(data, solution):
    merged = pd.merge(data, solution, on='indoml_id')

    merged['input_text'] = merged.apply(lambda row: f"description: {row['description']} retailer: {row['retailer']} price: {row['price']}", axis=1) #  
    merged['target_text'] = merged.apply(lambda row: f"supergroup: {row['supergroup']} group: {row['group']} module: {row['module']} brand: {row['brand']}", axis=1)
    
    return merged[['input_text', 'target_text']]


train_processed = preprocess_data(train_data, train_solution)
# test_processed = preprocess_data(test_data, test_solution)
val_processed = preprocess_data(val_data, val_solution)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_processed)
# test_dataset = Dataset.from_pandas(test_processed)
val_dataset = Dataset.from_pandas(val_processed)


dataset_dict = DatasetDict({
    'train': train_dataset,
    # 'test': test_dataset,
    'validation': val_dataset
})


# tokenizer = AutoTokenizer.from_pretrained("fine_tuned_flan_t5-small_60")
# model = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_flan_t5-small_60")

tokenizer = AutoTokenizer.from_pretrained("flan_t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("flan_t5-small")


def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=64, padding='max_length', truncation=True)
    labels = tokenizer(targets, max_length=64, padding='max_length', truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)


tokenized_datasets.save_to_disk('./finetuned_flan_t5-small-tokenized_wo_price_stratified_dataset')


# Ensure the logging directory exists
os.makedirs('./logs_flan_t5-small_60_es10_wo_price_stratified', exist_ok=True)

# Set up logging configuration
logging.basicConfig(
    filename='./logs_flan_t5-small_60_es10_wo_price_stratified/training.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info(f"Number of training examples: {len(dataset_dict['train'])} \n Number of validation examples: {len(dataset_dict['validation'])}")


training_args = TrainingArguments(
    output_dir='./results_flan_t5-small_60_es10_wo_price_stratified',
    evaluation_strategy='epoch',
    save_strategy='epoch',  # Ensure save strategy matches evaluation strategy
    learning_rate=2e-3,
    # learning_rate=9.370139240269111e-09,
    per_device_train_batch_size=384,
    per_device_eval_batch_size=384,
    num_train_epochs=60,
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir='./logs_flan_t5-small_60_es10_wo_price_stratified',
    logging_steps=100,
    report_to='none',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False
)


# Log the training arguments
logging.info(f"Training Arguments: {training_args}")


class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
        #     print(f"Step: {state.global_step}")
        #     for key, value in logs.items():
        #         print(f"{key}: {value}")
        #     print("\n")

            log_message = f"Step: {state.global_step}\n"
            for key, value in logs.items():
                log_message += f"{key}: {value}\n"
            log_message += "\n"
            print(log_message)
            logging.info(log_message)
            
            
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    callbacks=[CustomCallback(), EarlyStoppingCallback(early_stopping_patience=10)]
)

trainer.train()


val_results = trainer.evaluate(eval_dataset=tokenized_datasets['validation'])
print(f"Validation Loss: {val_results['eval_loss']}")
logging.info(f"Validation Loss: {val_results['eval_loss']}")

# test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
# print(f"Test Loss: {test_results['eval_loss']}")


model.save_pretrained('./fine_tuned_flan_t5-small_60_es10_wo_price_stratified')
tokenizer.save_pretrained('./fine_tuned_flan_t5-small_60_es10_wo_price_stratified')