import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class ReviewsDataset(Dataset):

    def __init__(self, data_dir, split="test", output="brand", trim=False):

        self.data_dir = data_dir
        self.split = split
        self.output = output

        start = time()
        # Load the data
        self.data = self._load_data(data_dir, split, output)

        # Load the maps
        self.out2idx, self.idx2out = self._load_map(output)

        # Categorize the data using the map
        if "na" in self.out2idx and split != "test":
            print("Found 'na' as the label, removing examples with 'na' label.")
            print(f"Original data has {len(self.data)} examples.")
            self.data = self.data[self.data[output] != "na"] # remove examples with 'na' label
            # remove 'na' from the map by reducing everything by 1 if it is greater than the index of 'na'
            for key in self.out2idx.keys():
                if self.out2idx[key] > self.out2idx["na"]:
                    self.out2idx[key] -= 1
            del self.out2idx["na"]
            self.idx2out = {v: k for k, v in self.out2idx.items()}
            
        if self.split != "test":
            self.data[output] = self.data[output].map(self.out2idx)

        # TODO: CAN CHANGE THIS
        # Create input
        # Convert all relevant columns to string
        self.data[["description", "retailer", "price"]] = self.data[
            ["description", "retailer", "price"]
        ].astype(str)

        # Create 'input' column
        self.data["input"] = "Product Name: " + self.data["description"]

        # Add 'retailer' and 'price' information if they are not 'None'
        self.data.loc[self.data["retailer"] != None, "input"] += (
            ", Sold by retailer: " + self.data.loc[self.data["retailer"] != None, "retailer"]
        )
        self.data.loc[self.data["price"] != None, "input"] += (
            ", Price: "
            + self.data.loc[
                self.data["price"] != None, "price"
            ]
        )

        # # trim the dataset for debugging
        # if trim:
        #     if split == "train":
        #         idx = np.random.choice(len(self.data), 1000, replace=False)
        #         self.data = self.data.iloc[idx]
        #     elif split == "val":
        #         idx = np.random.choice(len(self.data), 200, replace=False)
        #         self.data = self.data.iloc[idx]
        #     else: # don't touch the test set
        #         idx = np.random.choice(len(self.data), 200, replace=False)
        #         self.data = self.data.iloc[idx]
        end = time()

        print(f"Data loaded in {(end-start)/60:.3f} minutes.")
        print(f"{split} data has {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        example = self.data.iloc[idx]
        if self.split != "test":
            return {"input": example["input"], "output": example[self.output], "id": example["indoml_id"]}
        else:
            return {"input": example["input"], "id": example["indoml_id"]}

    def _load_data(self, data_dir, split, output) -> pd.DataFrame:
        '''
        Creates a unified dataFrame from the json files consisting of the data and labels.
        '''
        
        data = json.load(open(f"{data_dir}/{split}.json", "r"))
        labels = json.load(open(f"{data_dir}/{split}_sol.json", "r")) if split != "test" else [{} for _ in range(len(data))]
        
        keys = list(data[0].keys()) + [key for key in labels[0].keys() if key == output]
        
        df = {key: [] for key in keys}
        
        for d, l in tqdm(zip(data, labels), desc=f"Loading {split} data", total=len(data)):
            for key in keys:
                val = d.get(key, l.get(key, None))
                df[key].append(val)
                
        df = pd.DataFrame(df)
        
        return df
    
    def _load_map(self, column) -> Tuple[dict, dict]:
        '''
        Aux function to load the map from column to index.
        Inputs the col_name to load the map from.
        '''

        with open(f"{self.data_dir}/maps/{column}2idx.pkl", "rb") as f:
            map = pickle.load(f)
        
        with open(f"{self.data_dir}/maps/idx2{column}.pkl", "rb") as f:
            map_ = pickle.load(f)

        return map, map_


def reviewCollate(batch):

    inputs = [example["input"] for example in batch]
    not_test = "output" in batch[0].keys()
    # outputs = torch.LongTensor([example["output"] for example in batch]) if not_test else torch.LongTensor([[-1] for _ in batch])
    # try:
    #     outputs = torch.LongTensor([example["output"] for example in batch]) if not_test else torch.LongTensor([[-1] for _ in batch])
    # except Exception as e:
    #     print(f"Error converting outputs to LongTensor: {e}")
    #     for example in batch:
    #         print(f"Example output: {example['output']}")
    #     raise e

    # max_value = 2**63 - 1  # Maximum value for int64
    # outputs = torch.LongTensor([min(int(example["output"]), max_value) for example in batch]) if not_test else torch.LongTensor([[-1] for _ in batch])
    outputs = torch.LongTensor([int(0 if example["output"] != example["output"] else example["output"]) for example in batch]) if not_test else torch.LongTensor([[-1] for _ in batch])

    # outputs = torch.Tensor([example["output"] for example in batch]).to(dtype=torch.int64) if not_test else torch.Tensor([[-1] for _ in batch])
    ids = torch.Tensor([example["id"] for example in batch]).to(dtype=torch.int32)

    return {"input": inputs, "output": outputs, "ids": ids}


class ReviewsDataLoader(DataLoader):
    '''
    Utility class as a DataLoader.
    '''

    def __init__(self, *args, **kwargs):
        assert "collate_fn" not in kwargs, "collate_fn is not allowed, we already have a custom one in src/dataset.py." 
        kwargs["collate_fn"] = reviewCollate
        super(ReviewsDataLoader, self).__init__(*args, **kwargs)
        

if __name__ == '__main__':
    
    dataset = ReviewsDataset(data_dir="data/", split="test", output="supergroup")
    print(len(dataset))
    dataloader = ReviewsDataLoader(dataset, batch_size=32, shuffle=True)
    
    # for batch in dataloader:
    #     if any(batch['output'] >= len(dataset.out2idx)) or any(batch['output'] < 0):
    #         print("Found an index out of bounds.")
    #         print(len(dataset.out2idx))
    #         print(batch['output'])
    #         break