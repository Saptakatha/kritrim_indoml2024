# indoml_2024

### Environment Setup <a name="env-setup"></a>
The dependency pakages can be installed using the command.
```python
pip install -r requirements.txt
```

### Start LLM server  <a name="llm-server-setup"></a>
The llama-3-8b-instruct-awq model is hosted as a server using the following command.
```python 
python llm_server.py
```

### Few-shot inferencing <a name=""></a>
Run the following script to perform few-shot inference on test samples using the llm model.
```python
python fewshot_infer.py
```
