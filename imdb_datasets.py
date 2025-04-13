#datasets.py

import torch
from torch.utils.data import DataLoader, Dataset

class ImdbDataset(Dataset):
    def __init__(self, notes, targets, tokenizer, max_len):
        self.notes = notes
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return (len(self.notes))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        note = str(self.notes[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
          note,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=True,
          truncation=True,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
            #'text': note,
            'label': torch.tensor(target, dtype=torch.long),
            'input_ids': (encoding['input_ids']).flatten(),
            'attention_mask': (encoding['attention_mask']).flatten(),
            'token_type_ids': (encoding['token_type_ids']).flatten()
        }