import torch
from tqdm import tqdm
from transformers import AutoTokenizer

## Tokenizer를 로드하기 위해 Model_Type과 Model_Name 필요
def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.model_name)

def tokenized_dataset(dataset, tokenizer, args):
    """ input_style에 따라 sentence를 tokenizing 합니다."""
    
    if args.input_style == 'baseline':
        concat_entity = []
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
            temp = ''
            temp = e01 + '[SEP]' + e02
            concat_entity.append(temp)
        max_length=256
    
    # [CLS] subject [RELATION] object [SEP] sentence [SEP]
    elif args.input_style == 'relation_token':
        concat_entity = []
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
            temp = ''
            temp = e01 + '[RELATION]' + e02
            concat_entity.append(temp)
        special_tokens_dict = {'additional_special_tokens': ['[RELATION]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        max_length=256
    
    # [CLS] 다음 문장에서 subject 와 object 의 관계를 알수있다. [SEP] sentence [SEP]
    elif args.input_style == 'daum':
        concat_entity = []
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
            temp = ''
            temp = '다음 문장에서 ' + e01 + ' 와 ' + e02 + ' 의 관계를 알수있다.'
            concat_entity.append(temp)
        max_length=256


    # 모델에 따라 token_type_ids return 여부 결정
    if args.model_name.startswith('klue/roberta'):
        return_token_type_ids=False
    else:
        return_token_type_ids=True

    
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_token_type_ids=return_token_type_ids,
        )
        

    if args.stopwords==True:
        # load StopWords
        stopwords = []
        f = open(args.stopwords_dir)
        while True:
            line = f.readline()
            stopwords.append(line.strip())
            if not line:
                break 
        f.close()
        stopwords_ids = tokenizer.convert_tokens_to_ids(stopwords)
        stopwords_ids = set(stopwords_ids)

        print("Start removing stopwords")
        stopwords_sentence = dict(input_ids = [], attention_mask = [])
        input_ids, attention_mask = [], []

        max_len = 256
        for input_id in tqdm(tokenized_sentences.input_ids):
            none_stopwords = [int(id) for id in input_id if id not in stopwords_ids]

            # add "input_ids" 
            id = none_stopwords
            if len(none_stopwords) < max_len:
                id.extend([1] * (max_len-len(none_stopwords)))
            input_ids.append(id)

            # add "attention_mask"
            noone = len([id for id in none_stopwords if id > 1])
            attention = [0] * (max_len)
            attention[:noone] = [1] * noone
            attention_mask.append(attention)

        stopwords_sentence['input_ids'] = torch.as_tensor(input_ids)
        stopwords_sentence['attention_mask'] = torch.as_tensor(attention_mask)

        return stopwords_sentence



    return tokenized_sentences


