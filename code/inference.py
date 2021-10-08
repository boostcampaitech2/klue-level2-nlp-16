import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *
from model import *
from load_data import *
from tokenizer import *

def main(args):

    tokenizer=load_tokenizer(args)
    test_id, test_dataset, test_label = load_test_dataset(args.testdataset_dir, tokenizer, args)
    Re_test_dataset=RE_Dataset(test_dataset,test_label)
    
    dataloader = DataLoader(Re_test_dataset, batch_size=16, shuffle=False)
    
    pred_prob=[]

    # KFold Soft-Voting
    if args.kfold:
        for fold_index in range(1,6):

            model=load_model(args,f'./result/kfold/{args.model_name}/{fold_index}-fold')
            model.resize_token_embeddings(len(tokenizer))
            model.to(args.device)
            model.eval()
            
            output_prob = []

            for i, data in enumerate(tqdm(dataloader)):
                with torch.no_grad():
                    if args.model_name.startswith('klue/roberta'):
                        outputs=model(
                            input_ids=data['input_ids'].to(args.device),
                            attention_mask=data['attention_mask'].to(args.device),
                        )
                    else:
                        outputs=model(
                            input_ids=data['input_ids'].to(args.device),
                            attention_mask=data['attention_mask'].to(args.device),
                            token_type_ids=data['token_type_ids'].to(args.device)
                        )

                    logits=outputs[0]
                    prob=F.softmax(logits,dim=-1).detach().cpu().numpy()
                    output_prob.append(prob)

            output_prob=np.concatenate(output_prob, axis=0).tolist()
            pred_prob.append(output_prob)       

        pred_prob=np.sum(pred_prob, axis=0)/5
        pred_answer=np.argmax(pred_prob, axis=-1)
        pred_answer=num_to_label(pred_answer)
        
        output=pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':pred_prob.tolist()})
        output.to_csv(f'./result/kfold/{args.model_name}/submission.csv',index=False)
        print('---- Finish! ----')

    # non KFold
    else:
        ## load my model
        model=load_model(args,f'./result/non_kfold/{args.model_name}')
        model.resize_token_embeddings(len(tokenizer))
        model.to(args.device)
        model.eval()

        ## predict answer
        output_prob = []
        output_pred = []

        for i,data in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                if args.model_name.startswith('klue/roberta'):
                    outputs=model(
                        input_ids=data['input_ids'].to(args.device),
                        attention_mask=data['attention_mask'].to(args.device),
                    )
                else:
                    outputs=model(
                        input_ids=data['input_ids'].to(args.device),
                        attention_mask=data['attention_mask'].to(args.device),
                        token_type_ids=data['token_type_ids'].to(args.device)
                    )

            logits=outputs[0]
            prob=F.softmax(logits,dim=-1).detach().cpu().numpy()
            logits=logits.detach().cpu().numpy()
            result=np.argmax(logits,axis=-1)

            output_pred.append(result)
            output_prob.append(prob)

        pred_answer=np.concatenate(output_pred).tolist()
        pred_answer=num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
        output_prob=np.concatenate(output_prob, axis=0).tolist()
    
        output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
        output.to_csv(f'./result/non_kfold/{args.model_name}/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
        print('---- Finish! ----')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', type=bool, default=True)    
    parser.add_argument('--testdataset_dir',type=str,default="../dataset/test/test_data.csv")
    parser.add_argument('--device',type=str,default='cuda:0')
    
    parser.add_argument('--kfold', type=bool, default=False)
    parser.add_argument('--stopwords', type=bool, default=False)
    parser.add_argument('--stopwords_dir',type=str,default='../dataset/stopwords/StopWords_klue_bert-base_5.txt')
    
    ## require argumnet
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--input_style',type=str, required=True ,help='sentence input_style [baseline, relation_token, daum] ')
    
    # parser.add_argument('--Tokenizer_Model_Name',type=str,default="monologg/koelectra-base-v3-discriminator")
    args=parser.parse_args()
    print(args)

    main(args)
