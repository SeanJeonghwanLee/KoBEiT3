import sentencepiece as spm
import pandas as pd
import urllib.request
import csv

spm.SentencePieceTrainer.Train('--input=./merge_text.txt --model_prefix=smp --vocab_size=10000 --character_coverage=0.9995') # --max_sentence_length=9999')
