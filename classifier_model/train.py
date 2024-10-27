import torch
from src.trainer import Trainer

# note that this code IS NOT going to run on multiple GPUs,
# please see the README / trainer.py for that. If you run
# on colab, it will utilize the GPU there. Please raise an
# issue if you don't see that happen.

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

trainers = {}
for col in ['module']: # 'supergroup', 'group', 'module', 'brand'
    print("*"*88)
    print(f"Training for {col}")
    print("*"*88)
    trainers[col] = Trainer(data_dir="data/", device=device, output=col, trim=True)
    trainers[col].train()
    print(f"Training finished for {col}")
    print("*"*88)
    # print(f"Testing on {col}")
    # print("*"*88)
    # trainer[col].test()
    # print(f"Testing finished for {col}")
    # print("*"*88)