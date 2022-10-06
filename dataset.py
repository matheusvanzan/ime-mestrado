import torch
import pandas as pd

from torch.utils.data import Dataset

import settings


class NetflixDataset(Dataset):

    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        for txt in txt_list:
            # Encode the descriptions using the GPT tokenizer
            encodings_dict = tokenizer('<|startoftext|>' 
                                        + txt +    
                                        '<|endoftext|>',
                                        truncation=True,
                                        max_length=max_length, 
                                        padding='max_length')
            input_ids = torch.tensor(encodings_dict['input_ids'])    
            self.input_ids.append(input_ids)
            mask = torch.tensor(encodings_dict['attention_mask'])
            self.attn_masks.append(mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def get_dataset(tokenizer):
    descriptions = pd.read_csv(settings.PATH_PROJECT + 'netflix_titles.csv')['description'][:100] # limit 1000

    max_length = max([len(tokenizer.encode(description)) for description in descriptions])

    print('max_length', max_length)
    dataset = NetflixDataset(descriptions, tokenizer, max_length)
    return dataset