#tagging.py

import json

import transformers
import spacy
from datasets import load_dataset

dataset = load_dataset('eraser-benchmark/movie_rationales', trust_remote_code=True)

tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

nlp = spacy.load("en_core_web_sm")

sents = []
for line in dataset['test']['review']:
  tokens_list = [{'token': '[CLS]', 'word': '[CLS]', 'pos': None, 'position': 0, 'synt_tag': None}]
  tags = nlp(line.replace('\n', ''))

  separate_tokens = [tokenizer.decode(token) for token in tokenizer.encode(line)]

  cur_token = ''
  counter = 0
  for tok in separate_tokens[1:-1]:
    cur_token += tok.strip('#')
    tokens_list.append({'token': tok, 'word': tags[counter].text, 'pos': tags[counter].pos_, 'position': counter+1, 'synt_tag': tags[counter].dep_})
    if tags[counter].text == cur_token:
      counter += 1
      cur_token = ''
  tokens_list.append({'token': '[SEP]', 'word': '[SEP]', 'pos': None, 'position': -1, 'synt_tag': None})

  sents.append(tokens_list)

  with open('sents.json', 'w') as file:
    file.write(json.dumps(sents))