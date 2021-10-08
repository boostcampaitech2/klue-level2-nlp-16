import pandas as pd
import torch
from ast import literal_eval
from tokenizer import *

class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for subj, obj in zip(dataset['subject_entity'], dataset['object_entity']):
        subj = literal_eval(subj)['word']
        obj = literal_eval(obj)['word']
        
        subject_entity.append(subj)
        object_entity.append(obj)

    out_dataset = pd.DataFrame({
        'id':dataset['id'],
        'sentence':dataset['sentence'],
        'subject_entity':subject_entity,
        'object_entity':object_entity,
        'label':dataset['label'],
        })
    return out_dataset

def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)
  
    return dataset

def load_test_dataset(dataset_dir, tokenizer, args):
    """
        test dataset을 불러온 후,
        tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int,test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, args)

    return test_dataset['id'], tokenized_test, test_label
