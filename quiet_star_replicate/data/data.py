import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset
import os
import random
import tqdm
from typing import Any
from collections import Counter, defaultdict
class CustomTokenizer:
    def __init__(self, data_str, min_freq=0):
        temp_vocab = Counter(data_str)
        temp_vocab = {k for k, v in temp_vocab.items() if v > min_freq}
        self.vocab = sorted(list(temp_vocab))
        self.bos_token = "<bos>"
        self.vocab.insert(0, self.bos_token)
        self.bos_token_id = 0
        if min_freq > 0:
            self.unk_token = "<unk>"
            self.vocab.insert(1, self.unk_token)
            self.unk_token_id = 1

        self.index_to_char_list = self.vocab
        self.char_to_index_dict = dict((c, i) for i, c in enumerate(self.index_to_char_list))
        
        if min_freq > 0:
            self.char_to_index_dict = defaultdict(self.get_unk_token_id, self.char_to_index_dict)
    def get_unk_token_id(self):
        return self.unk_token_id
    def add_token(self, token_str_identifier):
        if token_str_identifier in self.char_to_index_dict:
            print('token already present')
            return self.char_to_index_dict[token_str_identifier]
        token_id = len(self.vocab)
        self.vocab.append(token_str_identifier)
        self.char_to_index_dict[token_str_identifier] = token_id
        return token_id
    def encode(self, char):
        return self.char_to_index_dict[char]
    def detokenize(self, ids):
        if isinstance(ids, torch.Tensor):
            assert len(ids.shape) == 1, "the size of a detokenized ids tensor can only be one dimensional"
            ids = ids.tolist()
        return "".join([self.index_to_char_list[i] for i in ids])
    def batch_detokenize(self, batch_ids):
        return [self.detokenize(ids) for ids in batch_ids]
    def tokenize(self, string: str):
        return [self.char_to_index_dict[c] for c in string]
    @staticmethod
    def load(folder, type_name):
        return torch.load(os.path.join(folder, CustomTokenizer.get_file_name(type_name)), weights_only=False)
    @staticmethod
    def get_file_name(type_name):
        return f"CustomTokenizer_{type_name}.pkl"
    def save(self, folder, type_name):
        torch.save(self, os.path.join(folder, CustomTokenizer.get_file_name(type_name)))

class ShakespeareDataset(Dataset):
    def __init__(self, data_str, seq_len):
        self.data = [data_str[i * seq_len: (i+1) * seq_len] for i in range(len(data_str) // seq_len)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def get_train_val_test_datasets(data_str, seq_len) -> tuple[Dataset, Dataset, Dataset]:
    # data_str = open('./tiny_shakespeare.txt', 'r').read()
    # data_str = open(data_file_path, 'r').read()

    full_dataset_shakespeare = ShakespeareDataset(data_str, seq_len)
    train_dataset_shakespeare, train_reward_model_dataset_shakespeare, eval_dataset_shakespeare = random_split(full_dataset_shakespeare, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    return train_dataset_shakespeare, train_reward_model_dataset_shakespeare, eval_dataset_shakespeare
def get_shakespeare_collate_fn(tokenizer):
    def shakespeare_collate_fn(examples):
        input_ids = []
        for example in examples:
            input_ids.append([tokenizer.bos_token_id] + tokenizer.tokenize(example))
        return torch.tensor(input_ids)
    return shakespeare_collate_fn


class FWDataset(Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        data = load_dataset("HuggingFaceFW/fineweb", "sample-10BT")
        # preprocess the data so that I take a random sample of seq_len from every element in the dataset
        self.data = []
        for data_element in tqdm.tqdm(data['train'], desc="loading data"):
            data_element_len = len(data_element['text'])
            if data_element_len < seq_len:
                continue
            start_index = random.randint(0, data_element_len - seq_len)
            self.data += [data_element['text'][start_index: start_index + seq_len]]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]
    def save(self, dataset_path):
        '''saves the current config in a standard naming convention'''
        torch.save(self, os.path.join(dataset_path, FWDataset.get_file_name(self.seq_len)))
    @staticmethod
    def load(dataset_path, seq_len):
        '''loads dataset if cached, else computes the dataset and caches it.'''
        dataset_name = FWDataset.get_file_name(seq_len)
        if dataset_name in os.listdir(dataset_path):
            return torch.load(os.path.join(dataset_path, FWDataset.get_file_name(seq_len)), weights_only=False)
        else:
            print(f"couldn't find cached dataset for: {dataset_name}")
            new_dataset = FWDataset(seq_len)
            new_dataset.save(dataset_path)
            return new_dataset
    @staticmethod
    def get_file_name(seq_len, dataset_type="one_per"):
        return f"FWDataset_sl={seq_len}_{dataset_type}.pkl"

