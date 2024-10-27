import json
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from tqdm import tqdm

from src.dataset import ReviewsDataset, ReviewsDataLoader

class Inference:

    def __init__(self, model_dir, data_file, output_file, device="cuda"):
        self.device = device
        self.model_dir = model_dir
        self.data_file = data_file
        self.output_file = output_file

        # Load the tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained("./bert-tokenizer/")
        
        # Load the models
        self.lm = BertForSequenceClassification.from_pretrained("./bert-model/").to(device)
        self.head = torch.nn.Linear(self.lm.config.hidden_size, len(self.load_idx2out())).to(device)
        
        # Load the best model checkpoint
        checkpoint = torch.load(f"{model_dir}/best_model_brand_epoch_5.pt")
        self.lm.load_state_dict(checkpoint[0])
        self.head.load_state_dict(checkpoint[1])
        
        self.lm.eval()
        self.head.eval()

    def load_idx2out(self):
        # Load the idx2out mapping from the training dataset
        train_dataset = ReviewsDataset("data/", split="train", output="brand", trim=False)
        return train_dataset.idx2out

    def load_test_data(self):
        # Load the test data
        test_dataset = ReviewsDataset(self.data_file, split="test", output="brand", trim=False)
        test_dataloader = ReviewsDataLoader(test_dataset, batch_size=512, shuffle=False)
        return test_dataloader

    @torch.no_grad()
    def infer(self):
        test_dataloader = self.load_test_data()
        idx2out = self.load_idx2out()

        test_outputs = []
        test_ids = []

        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            # Tokenize the input
            inputs = self.tokenizer(
                batch["input"],
                truncation=True,
                return_tensors="pt",
                padding="max_length",
                max_length= 64 # 512,
            )
            inputs.to(self.device)
            targets = batch["output"].to(self.device)

            # Forward pass
            lm_outputs = self.lm(**inputs, output_hidden_states=True)
            cls_output = lm_outputs.hidden_states[-1][:, 0, :]
            out = self.head(cls_output)

            outputs = torch.argmax(out, dim=1)
            test_outputs.append(outputs)
            test_ids.append(batch["ids"].to(self.device))

        # Concatenate the outputs and ids
        test_outputs = torch.cat(test_outputs).cpu().detach().numpy().reshape(-1)
        test_ids = torch.cat(test_ids).cpu().detach().numpy().reshape(-1)

        # Format the results
        results = []
        for test_id, test_out in zip(test_ids, test_outputs):
            cur_out = {
                "indoml_id": int(test_id),
                "brand": str(idx2out[test_out])
            }
            results.append(cur_out)

        # Save the results to the output file
        with open(self.output_file, "w") as outfile:
            json.dump(results, outfile, indent=4)

        print(f"Results saved to {self.output_file}")

if __name__ == "__main__":
    model_dir = "model"
    data_file = "attribute_test.data"
    output_file = "attribute_test.solution"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inference = Inference(model_dir, data_file, output_file, device)
    inference.infer()