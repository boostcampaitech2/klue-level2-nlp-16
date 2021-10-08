import random
import pandas as pd
from ast import literal_eval

from koeda import AEDA
from load_data import *


# Random Delete
def preprocessing_dataset_rd(dataset):
    """ Augment train data with Random Delete Process """
    fixed_sentence = []
    new_subj = []
    new_obj = []
    new_label = []
    stop_word = ['은', '는', '이', '가', '을', '를']
    for subj, obj, sentence, label in zip(dataset['subject_entity'], dataset['object_entity'], 
        dataset['sentence'], dataset['label']):
        
        subj = literal_eval(subj)['word']
        obj = literal_eval(obj)['word']
        sentence = sentence.split(" ")
                
        # randomly select index and delete selected index word
        random_index = random.randint(0, len(sentence)-1)
        remove = sentence[random_index]
        if remove[-1] in stop_word:
            remove = remove[:-1]
        while remove in subj or subj in remove or remove in obj or obj in remove:
            random_index = random.randint(0, len(sentence)-1)
            remove = sentence[random_index]
            if remove[-1] in stop_word:
                remove = remove[:-1]
        sentence.pop(random_index)
        sentence = " ".join(sentence)
        fixed_sentence.append(sentence)
        new_subj.append(subj)
        new_obj.append(obj)
        new_label.append(label)

    output = pd.DataFrame({
        'sentence':fixed_sentence,
        'subject_entity':new_subj,
        'object_entity':new_obj,
        'label':new_label
        })
    return output

# Random Swap
def preprocessing_dataset_rs(dataset):
    """ Augment train data with Random Swap Process """
    fixed_sentence = []
    new_subject = []
    new_object = []
    new_label = []
    for sentence, subj, obj, label in zip(dataset['sentence'], dataset['subject_entity'], 
        dataset['object_entity'],dataset['label']):
        
        sentence = sentence.split(" ")
                
        # randomly select two indicies and switch
        random_index_1 = random.randint(0, len(sentence)-1)
        random_index_2 = random.randint(0, len(sentence)-1)
        while random_index_1 == random_index_2:
            random_index_2 = random.randint(0, len(sentence)-1)
                
        temp = sentence[random_index_1]
        sentence[random_index_1] = sentence[random_index_2]
        sentence[random_index_2] = temp
        sentence = " ".join(sentence)

        subj = literal_eval(subj)['word']
        obj = literal_eval(obj)['word']
                
        fixed_sentence.append(sentence)
        new_subject.append(subj)
        new_object.append(obj)
        new_label.append(label)

    output = pd.DataFrame({
        'sentence':fixed_sentence,
        'subject_entity':new_subject,
        'object_entity':new_object,
        'label':new_label
    })
    return output

# Subject, Object Dictionary
def create_entitiy_dic(dataset):
    """ Create subject and object type dictionary to use in create new sentence process """
    entity_dic = {}
    for subj, obj in zip(dataset['subject_entity'], dataset['object_entity']):
        subj_type = literal_eval(subj)['type']
        subj_word = literal_eval(subj)['word']

        if subj_type not in entity_dic:
            entity_dic[subj_type] = [subj_word]
        else:
            entity_dic[subj_type].append(subj_word)
        
        obj_type = literal_eval(obj)['type']
        obj_word = literal_eval(obj)['word']

        if obj_type not in entity_dic:
            entity_dic[obj_type] = [obj_word]
        else:
            entity_dic[obj_type].append(obj_word)
            
    return entity_dic 

def create_new_sentence(dataset, dictionary):
    """ Create new sentence that changed subject and object with synonyms in dictionary """
    literal_subject = []
    literal_object = []
    new_sentence = []
    new_label = []
    for subj, obj, sentence, label in zip(dataset['subject_entity'], dataset['object_entity'], 
        dataset['sentence'], dataset['label']):

        subj_word = literal_eval(subj)['word']
        subj_type = literal_eval(subj)['type']
        obj_word = literal_eval(obj)['word']
        obj_type = literal_eval(obj)['type']

        subj_len = len(dictionary[subj_type]) -1
        subj_index = random.randint(0, subj_len)
        subj_new_word = dictionary[subj_type][subj_index] 
        sentence = sentence.replace(subj_word, subj_new_word)

        obj_len = len(dictionary[obj_type]) - 1
        obj_index = random.randint(0, obj_len)
        obj_new_word = dictionary[obj_type][obj_index]
        sentence = sentence.replace(obj_word, obj_new_word)

        literal_subject.append(subj_new_word)
        literal_object.append(obj_new_word)
        new_sentence.append(sentence)
        new_label.append(label)
        
    output_time = pd.DataFrame({
        'sentence':new_sentence,
        'subject_entity':literal_subject,
        'object_entity':literal_object,
        'label':new_label,      
        })
    return output_time

# AEDA
SPACE_TOKEN = "\u241F"

def replace_space(text):
    return text.replace(" ", SPACE_TOKEN)

def revert_space(text):
    clean = " ".join("".join(text).replace(SPACE_TOKEN, " ").split()).strip()
    return clean

class myAEDA(AEDA):
    def aeda(self, data, p):
        if p is None:
            p = self.ratio

        split_words = self.morpheme_analyzer.morphs(replace_space(data))
        words = self.morpheme_analyzer.morphs(data)

        new_words = []
        q = random.randint(1, int(p * len(words) + 1))
        qs_list = [
            index
            for index in range(len(split_words))
            if split_words[index] != SPACE_TOKEN
        ]
        qs = random.sample(qs_list, q)

        for j, word in enumerate(split_words):
            if j in qs:
                new_words.append(SPACE_TOKEN)
                new_words.append(
                    self.punctuations[random.randint(0, len(self.punctuations) - 1)]
                )
                new_words.append(SPACE_TOKEN)
                new_words.append(word)
            else:
                new_words.append(word)

        augmented_sentences = revert_space(new_words)
        return augmented_sentences

# Augmentation with AEDA process
def preprocessing_dataset_aeda(dataset):
    """ Augment data with AEDA process """
    literal_subject = []
    literal_object = []
    new_sentence = []
    new_label = []
    aeda = myAEDA(morpheme_analyzer="Okt", punc_ratio=0.3, punctuations=[".", ",", "!", "?", ";", ":"])
    for subj, obj, sentence, label in zip(dataset['subject_entity'], dataset['object_entity'], 
        dataset['sentence'], dataset['label']):
        if label != "no_relation":
            sentence = aeda(sentence, p = 0.3)
            literal_subject.append(literal_eval(subj)['word'])
            literal_object.append(literal_eval(obj)['word'])
            new_sentence.append(sentence)
            new_label.append(label)
        
    output_time = pd.DataFrame({
        'sentence':new_sentence,
        'subject_entity':literal_subject,
        'object_entity':literal_object,
        'label':new_label,      
        })
    return output_time

# Subject, Object Switch
def preprocessing_dataset_switch(dataset):
    """ Augment data with switching subject and object """
    literal_subject = []
    literal_object = []
    new_sentence_list = []
    new_label = []

    from_label = ['org:alternate_names', 'per:alternate_name', 'per:other_family', 'per:colleagues', 
        'per:siblings', 'per:spouse', 'org:members', 'org:member_of', 'per:parents', 'per:children',
        'org:top_members/employees']

    to_label = ['org:alternate_names', 'per:alternate_name', 'per:other_family', 'per:colleagues', 
        'per:siblings', 'per:spouse', 'org:member_of', 'org:members', 'per:children', 'per:parents',
        'per:employee_of']

    for i in range(len(from_label)):
        change_data = dataset[dataset['label'] == from_label[i]]
        for subj, obj, sentence in zip(change_data['subject_entity'], change_data['object_entity'], 
            change_data['sentence']):
            new_sentence = ''
            to_object = literal_eval(subj)['word']
            to_subject = literal_eval(obj)['word']

            sub_index = (literal_eval(subj)['start_idx'], literal_eval(subj)['end_idx']+1)
            obj_index = (literal_eval(obj)['start_idx'], literal_eval(obj)['end_idx']+1)

            if sub_index[0] < obj_index[0]:
                new_sentence = sentence[:sub_index[0]] + to_subject + sentence[sub_index[1] : obj_index[0]] + to_object + sentence[obj_index[1]:]
            else:
                new_sentence = sentence[:obj_index[0]] + to_object + sentence[obj_index[1] : sub_index[0]] + to_subject + sentence[sub_index[1]:]
            
            literal_subject.append(to_subject)
            literal_object.append(to_object)
            new_sentence_list.append(new_sentence)
            new_label.append(to_label[i])
    
    output_time = pd.DataFrame({
        'sentence':new_sentence_list,
        'subject_entity':literal_subject,
        'object_entity':literal_object,
        'label':new_label,      
        })
    
    return output_time


def total_aug(dataset, dictionary, back_translation_dataset, max_aug, new_sentence = True, rd = True, 
    rs = True, aeda = True, switch = True, backtranslation = True):
    """ Augment data with argument specifiy whether run augmentation or not """

    aug_list = ["org:top_members/employees", "per:employee_of", "per:title", "org:member_of", "org:alternate_names", 
        "per:origin", "org:place_of_headquarters", "per:date_of_birth", "per:alternate_names", "per:spouse", "per:colleagues",
        "per:parents", "org:founded", "org:members", "per:date_of_death", "org:product", "per:children", "per:place_of_residence",
        "per:other_family", "per:place_of_birth", "org:founded_by", "per:product", "per:siblings", "org:political/religious_affiliation",
        "per:religion", "per:schools_attended", "org:dissolved", "org:number_of_employees/members", "per:place_of_death"]
    
    output = preprocessing_dataset(dataset)
    output = output.drop('id', axis=1)

    if switch:
        aug_data5 = preprocessing_dataset_switch(dataset)
        output = pd.concat([output, aug_data5], axis = 0)

    if backtranslation:
        back_translation_dataset = preprocessing_dataset(back_translation_dataset)
        back_translation_dataset = back_translation_dataset[back_translation_dataset['sentence'].notnull()]
        output = pd.concat([output, back_translation_dataset], axis=0)
    
    while_loop_aug = new_sentence or rs or rd or aeda
    for aug in aug_list:
        candidate = dataset[dataset['label'] == aug]

        while len(output[output['label'] == aug]) < max_aug and while_loop_aug:
            if len(output[output['label'] == aug]) < max_aug and new_sentence:
                aug_data1 = create_new_sentence(candidate, dictionary)
                output = pd.concat([output, aug_data1])
             
            if len(output[output['label'] == aug]) < max_aug and rs:
                aug_data3 = preprocessing_dataset_rs(candidate)
                output = pd.concat([output, aug_data3])
            
            if len(output[output['label'] == aug]) < max_aug and rd:
                aug_data2 = preprocessing_dataset_rd(candidate)
                output = pd.concat([output, aug_data2])
            
            if len(output[output['label'] == aug]) < max_aug and aeda:
                aug_data4 = preprocessing_dataset_aeda(candidate)
                output = pd.concat([output, aug_data4])

        output = output.sample(frac=1).reset_index(drop=True)
        output['id'] = output.index

    return output

def run_augmentation(args, train_dataset, valid_dataset, total_dataset):
    """ By argument, run augmentation or preprocess dataset """
    if args.augmentation:
        entity_dic = create_entitiy_dic(total_dataset)
        bt_dataset = pd.read_csv(args.bt_data_dir)
        
        train_dataset = total_aug(train_dataset, entity_dic, bt_dataset, max_aug=args.max_aug,
            new_sentence=args.new_sentence, rd=args.random_delete, rs=args.random_switch, 
            aeda=args.aeda, switch=args.switch, backtranslation=args.backtranslation)
        
    else:
        train_dataset = preprocessing_dataset(train_dataset)
        
    valid_dataset = preprocessing_dataset(valid_dataset)
        
    return train_dataset, valid_dataset