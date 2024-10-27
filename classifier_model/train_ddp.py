import torch
from src.trainer import Trainer, DDP_wrapper
import torch.multiprocessing as mp

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

trainers = {}
for col in ['group']:  # 'supergroup', # , 'module', 'brand'
    world_size = torch.cuda.device_count()
    # DDP check.
    if world_size > 1:
        print(f"Running on {world_size} GPUs.")
        mp.spawn(DDP_wrapper, args=(world_size, col, False), nprocs=world_size)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {device}")
        trainer = Trainer(data_dir="data/", device=device, output=col, trim=False)
        
        trainer.train()
        # trainer.test()