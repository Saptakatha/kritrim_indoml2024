from flask import Flask, make_response, request
from vllm import LLM, SamplingParams 
from threading import Thread
import torch, os


# Set the CUDA device to GPU ID 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Now cuda:0 refers to the second GPU


# Config
config = {
    "model": "casperhansen/llama-3-8b-instruct-awq",
    "sampling_parameters": SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=100
    )
}



app = Flask(__name__)
llm = LLM(model=config["model"], quantization="awq", device=device)


def infer_from_llm(prompt, response_list):
    # print("Start of infer_from_llm")
    response = llm.generate(prompt, config['sampling_parameters'])[0].outputs[0].text
    response_list.append(response)
    # print("Output of infer_from_llm",response)

def process_inference_request(prompt):
    # print("Start of process_inference_request")
    response_list = []
    thread = Thread(target=infer_from_llm, args=(prompt,response_list))
    thread.start()
    thread.join()
    result = response_list[0]
    # print("Output of process_inference_request",result)
    return result

# Healthcheck route
@app.route("/status", methods=['GET'])
def healthcheck():
    return make_response({"status": True})
    
# inference Route
@app.route('/infer', methods=['POST'])
def hello_world():
    req = request.json
    # print("hell_world_request",req)
    prompt = req.get('prompt', None)
    res=process_inference_request(prompt)
    # print("end of hell_world_request",res)
    return {'result': res}

if __name__ == "__main__":
    PORT = 8080
    HOST = "0.0.0.0"
    app.run(port=PORT, host=HOST)