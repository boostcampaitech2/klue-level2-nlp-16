import os
import argparse
import pandas as pd
import wandb
from model import load_model
from sklearn.model_selection import StratifiedKFold, train_test_split
from load_data import *
from tokenizer import *
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments
from utils import *
from trainer import *
from augmentation import *

def train(args):
    # Dataset
    total_dataset = pd.read_csv(args.load_data_dir)
    total_label = total_dataset['label'].values

    # Tokenizer
    tokenizer = load_tokenizer(args)
    special_tokens_dict = {'additional_special_tokens': ['[RELATION]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    if args.k_fold:
        print(f'-----------------Kfold ---------------------')
        fold_idx=1
        kf=StratifiedKFold(n_splits=5,shuffle=True) 
        for train_index, valid_index in kf.split(total_dataset,total_label):

            wandb.init(
                entity="nlprime",
                project=args.project_name,
                name=args.run_name+f'_{fold_idx}-fold',
                tags=[args.model_name]
            )
            wandb.config.update(args)


            print(f'-----------------Kfold {fold_idx} start---------------------')
            
            os.makedirs(f'./result/kfold/{args.model_name}/{fold_idx}-fold', exist_ok=True)
            train_dataset, valid_dataset = total_dataset.iloc[train_index], total_dataset.iloc[valid_index]
            train_dataset, valid_dataset = run_augmentation(args, train_dataset, valid_dataset, total_dataset)
            
            train_label, valid_label = train_dataset['label'].values, valid_dataset['label'].values
            train_label, valid_label = label_to_num(train_label), label_to_num(valid_label)
            
            train_tokenized = tokenized_dataset(train_dataset,tokenizer,args)
            valid_tokenized = tokenized_dataset(valid_dataset,tokenizer,args)
        
            RE_train_dataset = RE_Dataset(train_tokenized, train_label)
            RE_valid_dataset = RE_Dataset(valid_tokenized, valid_label)

            training_args = TrainingArguments(
                output_dir=f'./result/kfold/{args.model_name}/{fold_idx}-fold',        
                save_total_limit=args.save_total_limit,             
                save_steps=args.save_steps,               
                num_train_epochs=args.epochs,            
                learning_rate=args.learning_rate,              
                per_device_train_batch_size=args.per_device_train_batch_size,  
                per_device_eval_batch_size=args.per_device_eval_batch_size,  
                warmup_steps=args.warmup_steps,               
                weight_decay=args.weight_decay,               
                logging_dir=args.logging_dir,           
                logging_steps=args.logging_steps,             
                evaluation_strategy=args.evaluation_strategy,
                eval_steps = args.eval_steps,
                load_best_model_at_end = args.load_best_model_at_end, 
                metric_for_best_model='eval_micro f1 score'           
             )
            
            model=load_model(args)
            model.to(args.device)
            model.resize_token_embeddings(len(tokenizer))
            print(f'./result/kfold/{args.model_name}/{fold_idx}-fold')


            trainer = CustomTrainer(
                loss_name=args.loss_name,
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=RE_train_dataset,         # training dataset
                eval_dataset=RE_valid_dataset,             # evaluation dataset
                compute_metrics=compute_metrics,        # define metrics function
                callbacks=[EarlyStoppingCallback(args.early_stopping_patience)]
            )

            trainer.train()
            model.save_pretrained(f'./result/kfold/{args.model_name}/{fold_idx}-fold')
            wandb.finish()
            fold_idx += 1
    
    else:
        os.makedirs(f'./result/non_kfold/{args.model_name}', exist_ok=True)
        # load dataset
        train_dataset, valid_dataset = train_test_split(total_dataset, test_size=0.2, stratify=total_dataset['label'], random_state=16)
        train_dataset, valid_dataset = run_augmentation(args, train_dataset, valid_dataset, total_dataset)

        train_label = label_to_num(train_dataset['label'].values)
        valid_label = label_to_num(valid_dataset['label'].values)

        # tokenizing dataset
        train_tokenized = tokenized_dataset(train_dataset, tokenizer,args)
        valid_tokenized = tokenized_dataset(valid_dataset, tokenizer,args)

        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(train_tokenized, train_label)
        RE_valid_dataset = RE_Dataset(valid_tokenized, valid_label)

        model = load_model(args)
        model.resize_token_embeddings(len(tokenizer))
        model.parameters
        model.to(args.device)

        wandb.init(
            entity="nlprime",
            project=args.project_name,
            name=args.run_name,
            tags=[args.model_name]
        )
        wandb.config.update(args)
    
        training_args = TrainingArguments(
                output_dir = f'./result/non_kfold/{args.model_name}',
                save_total_limit=args.save_total_limit,             
                save_steps=args.save_steps,               
                num_train_epochs=args.epochs,            
                learning_rate=args.learning_rate,              
                per_device_train_batch_size=args.per_device_train_batch_size,  
                per_device_eval_batch_size=args.per_device_eval_batch_size,  
                warmup_steps=args.warmup_steps,               
                weight_decay=args.weight_decay,               
                logging_dir=args.logging_dir,           
                logging_steps=args.logging_steps,             
                evaluation_strategy=args.evaluation_strategy,
                eval_steps = args.eval_steps,
                load_best_model_at_end = args.load_best_model_at_end, 
                metric_for_best_model='eval_micro f1 score'  
            )
        
        trainer = CustomTrainer(
                loss_name=args.loss_name,
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=RE_train_dataset,         # training dataset
                eval_dataset=RE_valid_dataset,             # evaluation dataset
                compute_metrics=compute_metrics,        # define metrics function
                callbacks=[EarlyStoppingCallback(args.early_stopping_patience)]
            )

        # train model
        trainer.train()
        model.save_pretrained(f'./result/non_kfold/{args.model_name}')

if __name__ =='__main__':
    parser=argparse.ArgumentParser()
    
    ## seedseting
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    
    ## load_data dir
    parser.add_argument('--load_data_dir', type=str, default="../dataset/train/train.csv", help='Enter the location of the train file')
    parser.add_argument('--device',type=str,default='cuda:0')
    
    ## TrainingArguments
    parser.add_argument('--k_fold',type=bool,default=False)
    parser.add_argument('--save_total_limit',type=int,default='5',help='number of total save model(default:5')
    parser.add_argument('--save_steps',type=int,default='500',help='model saving setp(default:500')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate',type=float,default=5e-5,help='learning_rate(defalute=5e-5) ')
    parser.add_argument('--per_device_train_batch_size',type=int,default=16,help='batch size per device during training')
    parser.add_argument('--per_device_eval_batch_size',type=int,default=16,help='batch size for evaluation')
    parser.add_argument('--warmup_steps',type=int,default=500,help='number of warmup steps for learning rate scheduler')
    parser.add_argument('--weight_decay',type=int,default=0.01,help='strength of weight decay')
    parser.add_argument('--logging_dir',type=str,default='./logs',help='directory for storing logs')
    parser.add_argument('--logging_steps',type=int,default=100,help='lpog saving ste.')
    parser.add_argument('--evaluation_strategy',type=str,default='steps',help='evaluation strategy to adopt during training')
    parser.add_argument('--eval_steps',type=int,default=500,help='evaluation step.')
    parser.add_argument('--load_best_model_at_end',type=bool,default=True)
    parser.add_argument('--metric_for_best_model',type=str,default='eval_micro f1 score',help='metric for saving best model and earlystopping')
    parser.add_argument('--early_stopping_patience',type=int,default=3,help='earlystopping patience')
    parser.add_argument('--loss_name',type=str,default='default',help='loss function type(default: default) [cross_entropy, focal, f1, label_smoothing, dice]')
    
    ## Stopwords Argument
    parser.add_argument('--stopwords',type=bool,default=False)
    parser.add_argument('--stopwords_dir',type=str,default='../dataset/stopwords/StopWords_klue_bert-base_5.txt')
    
    ## Require Argument
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--input_style',type=str, required=True ,help='sentence input_style [baseline, relation_token, daum]')
    
    ## Inference procedure or not   
    parser.add_argument('--inference', type=bool, default=False)    
    
    ## Data Augmentation
    parser.add_argument('--augmentation', type=bool, default=False, help='enable augmentation or not')
    parser.add_argument('--max_aug', type=int, default=1500, help='Decide max augmentation data per label')
    parser.add_argument('--new_sentence', type=bool, default=True, help='Start augmentation with synonyms')
    parser.add_argument('--random_delete', type=bool, default=True, help='Start random delete augmentation')
    parser.add_argument('--random_switch', type=bool, default=True, help='Start random switch augmentation')
    parser.add_argument('--aeda', type=bool, default=True, help='Start AEDA augmentation')
    parser.add_argument('--switch', type=bool, default=True, help='Start Switch augmentation')
    parser.add_argument('--backtranslation', type=bool, default = True, help='Enable augmentation using backtranslation')
    parser.add_argument('--bt_data_dir', type=str, default="../dataset/backtranslation/bt_under_1000_dataset.csv", help='Enter the location of the backtranslation file')

    args=parser.parse_args()

    set_seed(args.seed)
    print(args)
    train(args)