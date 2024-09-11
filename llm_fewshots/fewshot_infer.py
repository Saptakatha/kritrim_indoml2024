import json
from multiprocessing.pool import Pool
from abc import ABC, abstractmethod
import pickle
import pandas as pd
from tqdm import tqdm
import time

# Global constants
NUM_THREADS = 1
TEST_FILE = "./data/attribute_test.data"
LLM_TYPE = "api"
OUTPUT_FILE = "./data/attribute_test.solution"
PROMPT_TEMPLATE_FILE = "./prompt.txt"
SIMILARITY_MATRIX = "./data/faiss_search_results_full.pkl"
TRAINING_DATA = "./data/train_data_label.csv"



class LLM_RESOURCE(ABC):

    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def generate_output(self, prompt):
        raise NotImplementedError

class REST_LLM_RESOURCE(LLM_RESOURCE):
    def __init__(self):
        self.endpoint = "http://localhost:8080"
        
    
    def generate_output(self, prompt):
        import requests
        response = requests.post("{}/infer".format(self.endpoint), json={"prompt": prompt})
        if response.status_code != 200:
            print(response.status_code)
            raise Exception("LLM Down")
        return response.text


# factory function for LLM resource
def get_relevant_llm_resource(llm_type):
    return {
        "api": REST_LLM_RESOURCE()
    }[llm_type]


class Inferencer:
    def __init__(self, llm_resource, num_shots=5):
        self.llm = llm_resource
        self.num_shots_for_learning = num_shots

    def read_prompt(self, learning_input, test_input):
        with open(PROMPT_TEMPLATE_FILE) as f:
            data = f.read()
        data = data.strip()
        return data.format(
            learning_input=learning_input,
            test_input=test_input
        )

    def get_similar_text_for_id(self, indoml_id):
        with open(SIMILARITY_MATRIX, "rb")  as f:
            data = pickle.load(f)
        ids = data[indoml_id]
        # Read the data for the ids
        train_df = pd.read_csv(TRAINING_DATA)
        
        query = 'indoml_id in {}'.format(ids)
        
        result = train_df.query(query)
        return result
    
    def generate_input_dict_for_training_text(self, training_text):
        inputs = ["title", "store", "details_Manufacturer"]
        return {key: training_text[key] for key in inputs}
        
    def generate_output_dict_for_training_text(self, training_text):
        
        labels = ["details_Brand","L0_category","L1_category","L2_category","L3_category","L4_category"]
        return {key: training_text[key] for key in labels}

    def generate_learning_inputs(self, indoml_id):
        # Get the similar texts from training data corresponding to this indoml id
        similar_texts = self.get_similar_text_for_id(indoml_id)
        
        # outputs = [self.generate_output_dict_for_training_text(t) for t in similar_texts.iterrows()]
        outputs = similar_texts.apply(self.generate_output_dict_for_training_text, axis=1)
        outputs = list(outputs)
        inputs = similar_texts.apply(self.generate_input_dict_for_training_text, axis=1)
        inputs = list(inputs)
        
        learning_materials = ['''
            input: {}
            output:{}

        '''.format(
            inputs[indx], 
            outputs[indx]) 
        for indx in range(len(inputs))]
        return "\n".join(learning_materials)


    def infer(self, indoml_id, input_text):
        learning_input = self.generate_learning_inputs(
            indoml_id
        )
        prompt = self.read_prompt(
            learning_input,
            input_text
        )
        output = self.llm.generate_output(
            prompt
        )
        return self.post_processing(output)
        # return output


    def post_processing(self, text):
        text = text.split("\\n")[1]
        start = text.find('{')
        end = text.find('}')
        json_txt = text[start:end+1]
        txt = "".join(json_txt.split("\n"))
        txt = txt.replace("\\", "")
        return txt


def main(data):
    indoml_id = data.get("indoml_id")
    text = {"title": data.get('title'), "store": data.get('store'), "details_Manufacturer": data.get('details_Manufacturer')}
    text = json.dumps(text)

    # Get the llm resource we want  to use
    llm_resource = get_relevant_llm_resource(LLM_TYPE)
    inferencer = Inferencer(llm_resource, num_shots=5)
    output = inferencer.infer(indoml_id, text)
    output = json.loads(output)
    output["indoml_id"] = indoml_id
    
    # print(" \n\n ouput :: {}".format(output))
    # persist the data in the output report
    with open(OUTPUT_FILE, "a") as fop:
        fop.write(json.dumps(output))
        fop.write("\n")

        

if __name__ == '__main__':
    # read the test inputs
    start_time = time.time()
    with open (TEST_FILE) as f:
        data = [json.loads(x.strip()) for x in f.readlines()]
    # print(data)
    with Pool(NUM_THREADS) as pool:
        pool.map(main, data)

    end_time = time.time()
    print("Execution time in seconds: ", end_time - start_time)