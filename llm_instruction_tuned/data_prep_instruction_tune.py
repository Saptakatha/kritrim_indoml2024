import json
from tqdm import tqdm
#instrcn_txt='''As an expert E-commerce label predictor, your task is to generate the appropriate labels for a given input text. You will receive an input text and your output should predict the most relevant labels for the input text. Please note that your response should be flexible enough to accommodate various input texts and label lists, allowing for accurate and creative label predictions based on the content of the input. For a new input, generate the output in JSON format only.'''

instrcn_txt ='''You are a world-class algorithm for extracting attribute-value pair information and label prediction in a structured format. Extract the attribute values and predict the most relevant labels for the input text in a JSON format. The valid output attributes are indoml_id, supergroup, group, module and brand.'''

fl_train_in=open("/rec-data/sapta/misc/LLaMA-Factory/input_data/train.features")
fl_train_out=open("/rec-data/sapta/misc/LLaMA-Factory/input_data/train.labels")

fl_train_in_ls=fl_train_in.readlines()
fl_train_out_ls=fl_train_out.readlines()
#fl_test=open('/notebooks/indoml/data_test.txt','w')

# for i,j in enumerate(tqdm(fl_train_in_ls)):
#     # prompt=""+j
#     fl_test.write(j)
#     #fl_test.write("\n")

# fl_test.close()

dataset=[]
for i,j in enumerate(tqdm(fl_train_in_ls)):
  tmp_dict={}
  tmp_dict['instruction']=instrcn_txt
  tmp_dict['input']=j
  tmp_dict['output']=fl_train_out_ls[i]
  dataset.append(tmp_dict)


with open('/rec-data/sapta/misc/LLaMA-Factory/data/phase2_data_train.json', 'w') as json_file:
    json.dump(dataset, json_file, indent=4)

# with open('/notebooks/indoml/data_test.json', 'w') as json_file:
#     json.dump(dataset, json_file, indent=4)