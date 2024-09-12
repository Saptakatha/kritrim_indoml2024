import gc
import json
import torch
from tqdm import tqdm
from typing import List, Tuple, Optional
from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
import pickle

# Function to read prompt data from file and create prompts dynamically
def create_custom_prompts_from_file(file_path: str, lora_path: str) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Dynamically create prompts from an input JSON file."""
    
    # Load the file containing prompt data
    with open(file_path, 'r') as f:
        test_lines = f.readlines()

    test_entries = [json.loads(line) for line in test_lines]

    # Create the prompt list dynamically
    prompts = []
    for entry in test_entries:
        # Extract data from the dictionary
        text = entry
        temperature = 0.0
        max_tokens = 100
        lora_request_name = "lora-request"

        # Create SamplingParams object
        sampling_params = SamplingParams(
            temperature=temperature,  
            top_p=0.95,
            max_tokens=max_tokens
        )

        # Create LoRARequest object if specified
        lora_request = None
        if lora_request_name:
            lora_request = LoRARequest(lora_request_name, 1, lora_path)

        # Add to the prompt list
        prompts.append((text, sampling_params, lora_request))

    return prompts

# Process requests and save outputs with a progress bar and real-time JSON updates
def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams, Optional[LoRARequest]]],
                     output_json_file: str):
    """Process a list of prompts, handle the outputs, and save to a JSON file in real-time."""
    request_id = 0
    results = []  # To store the results

    # Progress bar for the request processing
    total_prompts = len(test_prompts)
    with tqdm(total=total_prompts, desc="Processing Prompts") as pbar:
        while test_prompts or engine.has_unfinished_requests():
            if test_prompts:
                prompt, sampling_params, lora_request = test_prompts.pop(0)
                instn = '''As an expert E-commerce label predictor, your task is to generate the appropriate labels for a given input text. You will receive an input text and a list of labels, and your output should predict the most relevant labels for the input text. Your response should accurately identify the appropriate labels for the new input text, considering the provided list of labels. Please note that your response should be flexible enough to accommodate various input texts and label lists, allowing for accurate and creative label predictions based on the content of the input. For a new input and a list of labels, generate the output in json format only. Dont add any other information apart from the output. input: '''
                prompt_str = instn + str(prompt)

                # Add request to the engine
                engine.add_request(str(request_id),
                                   prompt_str,
                                   sampling_params,
                                   lora_request=lora_request)
                request_id += 1

            request_outputs: List[RequestOutput] = engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    result = {
                        "Prompt": request_output.prompt,
                        "Output": request_output.outputs[0].text
                    }
                    results.append(result)
                    
                    # Save results in real-time to the JSON file
                    with open(output_json_file, 'a') as json_file:
                        json.dump(result, json_file)
                        json_file.write('\n')  # To ensure each JSON object is on a new line

                    pbar.update(1)  # Update progress bar after each request

    print(f"Results saved to {output_json_file}")

# Initialize the engine with model and quantization settings
def initialize_engine(model: str, quantization: str, lora_repo: Optional[str]) -> LLMEngine:
    """Initialize the LLMEngine."""
    engine_args = EngineArgs(
        model=model,
        quantization=quantization,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True
    )
    return LLMEngine.from_engine_args(engine_args)

# Main function that integrates all components
def main():
    """Main function that sets up and runs the prompt processing."""
    
    test_configs = [{
        "name": "Custom_inference_with_lora_example",
        'model': 'casperhansen/llama-3-8b-instruct-awq',  # Replace with your model
        'quantization': "awq",  # Replace with your quantization method
        'lora_repo': '/notebooks/indoml/LLaMA-Factory/llama3_lora_20k/checkpoint-312'  # Replace with your LoRA repo
    }]

    for test_config in test_configs:
        print(f"~~~~~~~~~~~~~~~~ Running: {test_config['name']} ~~~~~~~~~~~~~~~~")

        # Initialize the engine with the provided model and LoRA
        engine = initialize_engine(test_config['model'],
                                   test_config['quantization'],
                                   test_config['lora_repo'])

        # Download the LoRA model if needed
        lora_path = "/notebooks/indoml/LLaMA-Factory/llama3_lora_20k/checkpoint-312"

        # Load the custom prompts from the JSON file
        test_prompts = create_custom_prompts_from_file('/notebooks/indoml/attribute_test.data', lora_path)

        # JSON file for real-time saving
        output_json_file = "/notebooks/indoml/output_results_20k.json"

        # Process the prompts and save the results to the JSON file
        process_requests(engine, test_prompts, output_json_file)

        # Clean up the GPU memory for the next test
        del engine
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
