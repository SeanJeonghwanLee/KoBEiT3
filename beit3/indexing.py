from datasets import CustomDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("../models/smp.model")


CustomDataset.make_dataset_index(
    data_path="../",
    tokenizer=tokenizer,
    label_path="../labels",
    non_existing_file="/home/seanlee/class/SpeechVQAPipeline/test_non_existing.txt"
)