from transformers import AutoTokenizer, BertModel
# import certifi
import os

def download():
    # os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = BertModel.from_pretrained("google-bert/bert-base-uncased")

    tokenizer.save_pretrained("./bert-tokenizer/")
    model.save_pretrained("./bert-model/")
    
    return

if __name__ == '__main__':
    download()