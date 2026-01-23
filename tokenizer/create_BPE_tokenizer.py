import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing



DATA_PARAM_PATH = "dataset/data_parameters.json"
TOKENIZER_PARAM_PATH = "tokenizer/tokenizer_param.json"
with open(DATA_PARAM_PATH, 'r') as f:
    dic_data_param = json.load(f)
with open(TOKENIZER_PARAM_PATH, 'r') as f:
    dic_token_param = json.load(f)

