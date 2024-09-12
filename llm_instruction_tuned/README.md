## LLAMA-Factory
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
