import torch
from torch.utils.data import Dataset, random_split
import os

class CustomTokenizer:
    def __init__(self, data_str):
        self.vocab = set(data_str)
        self.vocab = sorted(list(self.vocab))
        self.bos_token = "<bos>"
        self.vocab.insert(0, self.bos_token)
        self.bos_token_id = 0
        self.index_to_char_list = self.vocab
        self.char_to_index_dict = dict((c, i) for i, c in enumerate(self.index_to_char_list))
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



