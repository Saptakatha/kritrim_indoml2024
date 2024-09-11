# indoml_2024

### Environment Setup <a name="env-setup"></a>
The dependency pakages can be installed using the command.
```python
pip install -r requirements.txt
```

### Finetune LLM model <a name="finetune"></a>
The flan-t5-small model is finetuned on training dataset using the following command.
```python 
python train.py
```

### Infer using finetuned LLM model <a name="infer"></a>
Run the following script to perform inference on test samples using the finetuned llm model.
```python
python infer.py
```
