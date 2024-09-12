## LLAMA-Factory based Instruction Tuning
Clone the github repository [link](https://github.com/hiyouga/LLaMA-Factory.git)
Use conda environment with python==3.9
```
cd LLaMA-Factory
```
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip uninstall -y jax
pip install -e .[bitsandbytes,liger-kernel]
```
Then run the following command
```
llamafactory-cli train train_llama3_20k.json
```
## vLLM based Inferencing
Create a different conda followed by its activation
```
pip install vllm==0.4.1
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.0
```
Run the follwing command
```
python3 lora_inference.py
