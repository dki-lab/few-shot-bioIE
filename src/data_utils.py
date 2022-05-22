import pandas as pd
import numpy as np
import ipdb
import pickle
import re
import os
import yaml
from transformers import GPT2Tokenizer
gpt_2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import copy
import wandb
from torch.utils.data import TensorDataset

def load_yaml(path):
    with open(path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict

def aggregate_to_sequence(df, token_col_name, seq_col_name):
    # Adding Sentence to a BIO dataframe with only tokens and ner_tags as columns.

    sents = []
    sent = []

    for i, row in df.iterrows():

        if pd.isna(row[token_col_name]):
            sents.append(' '.join(sent))
            sent = []
        else:
            sent.append(row[token_col_name])

    sent_col = []
    i = 0

    for j, row in df.iterrows():

        try:
            sent_col.append(sents[i])
        except:
            import ipdb
            ipdb.set_trace()

        if pd.isna(row[token_col_name]):
            i += 1

    df[seq_col_name] = sent_col

    return df

def process_conll_datasets(df):
    df = aggregate_to_sequence(df, 'tokens', 'sents')
    df = aggregate_to_sequence(df, 'ner_tags', 'ner_seq')
    assert np.sum([len(sent.split()) == len(ner.split()) for sent, ner in zip(df.sents, df.ner_seq)]) == len(df), print('Something Wrong With CoNLL Format dataset.')
    df = df[['sents', 'ner_seq']].drop_duplicates()
    df = df.reset_index(drop=True)

    return df

def write_conll_format_file(filename, df, sent_label, tag_label):
    df = df.reset_index()

    with open(filename, 'w') as f:

        f.write('tokens\tner_tags\n')

        for i, row in df.iterrows():

            sent = row[sent_label].split()
            ner_seq = row[tag_label].split()

            for w, t in zip(sent, ner_seq):
                f.write('{}\t{}\n'.format(w, t))

            if i < len(df) - 1:
                f.write('\n')

    return df

def load_ner_dataset(data_path, subset_num):
    """
    Load train, dev and test datasets in CoNLL format and save them in a standard "one example per line" format.

    :param data_path: Filename for Dataset (loads filename + {"train","dev","test"}
    :return: (Train, Dev, Test) Tuple with Pandas DataFrames
        NER DataFrame Columns: ['sents', 'ner_seq']
    """
    if subset_num is None:
        train_path = data_path + '/train.csv'
    else:
        train_path = data_path + '/training_subsets/{}/subset.csv'.format(subset_num)

    if os.path.exists(train_path):
        train = pd.read_csv(train_path, sep='\t', quoting=3)
        dev = pd.read_csv(data_path+'/dev.csv', sep='\t', quoting=3)
        test = pd.read_csv(data_path+'/test.csv', sep='\t', quoting=3)
    else:
        train = pd.read_csv(data_path+'/train.conll.csv', sep='\t', skip_blank_lines=False, quoting=3,  keep_default_na=False, na_values=[''])
        dev = pd.read_csv(data_path+'/dev.conll.csv', sep='\t', skip_blank_lines=False, quoting=3,  keep_default_na=False, na_values=[''])
        test = pd.read_csv(data_path+'/test.conll.csv', sep='\t', skip_blank_lines=False, quoting=3, keep_default_na=False, na_values=[''])

        train = process_conll_datasets(train)
        dev = process_conll_datasets(dev)
        test = process_conll_datasets(test)

        train.to_csv(data_path+'/train.csv', sep='\t')
        dev.to_csv(data_path+'/dev.csv', sep='\t')
        test.to_csv(data_path+'/test.csv', sep='\t')

    return train, dev, test

def load_re_dataset(data_path, subset_num):
    if subset_num is None:
        train_path = data_path + '/train.tsv'
    else:
        train_path = data_path + '/training_subsets/{}/subset.csv'.format(subset_num)

    assert os.path.exists(train_path), 'No training file.'

    if subset_num is None:
        train = pd.read_csv(train_path, sep='\t', quoting=3)
    else:
        train = pd.read_csv(train_path, sep='\t', quoting=3, index_col=0)

    dev = pd.read_csv(data_path + '/dev.tsv', sep='\t', quoting=3)
    test = pd.read_csv(data_path + '/test.tsv', sep='\t', quoting=3)

    train.columns = ['id','sents','ent1','ent2','masked_sents','label']
    dev.columns = ['id','sents','ent1','ent2','masked_sents','label']
    test.columns = ['id','sents','ent1','ent2','masked_sents','label']
    return train, dev, test

def get_re_dataset(train_df, tokenizer, label_to_id):
    labels = [label_to_id[str(l)] for l in train_df['label']]
    input_ids, attention_mask, labels = tokenize(train_df['masked_sents'], labels, tokenizer)
    dataset = TensorDataset(input_ids, attention_mask, labels)

    return dataset

# Tokenize all texts and align the labels with them.
def tokenize(sentences, labels, tokenizer):
    input_ids_dev = []
    attention_masks_dev = []

    # For every sentence...
    for sent in sentences:
        encoded_dict_dev = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=wandb.config.max_seq_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids_dev.append(encoded_dict_dev['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks_dev.append(encoded_dict_dev['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids_dev, dim=0)
    attention_masks = torch.cat(attention_masks_dev, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

def reverse_token_preprocessing(sent):
    # Removing spaces around special chars
    sent = re.sub(" (?=[\'\)\]\,\-\.\%\:\;\/])", "", sent)
    sent = re.sub("(?<=[\'\-\(\[\.\/]) ", "", sent)

    return sent


def token_preprocessing(sent):
    #Adding spaces around special chars
    sent = re.sub("(?=[\'\)\]\,\-\.\%\:\;\/])", " ", sent)
    sent = re.sub("(?<=[\'\-\(\[\.\/])", " ", sent).strip()

    #Removing double spaces
    sent = re.sub("\s+",' ',sent)
    return sent

def augment_bio_dataframe(df, prompt_config):
    sep = prompt_config['sep']
    df = df[['sents', 'ner_seq']].drop_duplicates()
    # df = df.reset_index(drop=True)

    df = extract_entities(df, sep)
    df['orig_tok_sent'] = df.sents
    df['sents'] = [reverse_token_preprocessing(s) for s in df.sents]
    df = add_ner_prompts(df, prompt_config, sep)

    return df

def augment_re_dataframe(df, prompt_config):
    prompts = []
    empty_prompts = []
    labels = []

    prompt_sample_structure = prompt_config['sent_intro'] + ' {}\n' + prompt_config['retrieval_message'] + ' {}'
    empty_prompt_sample_structure = prompt_config['sent_intro'] + ' {}\n' + prompt_config['retrieval_message']

    for i, row in df.iterrows():
        sent = row['sents']
        entity1 = row['ent1']
        entity2 = row['ent2']
        label = row['label']
        label = prompt_config['label_verbalizer'][str(label)]

        prompt = prompt_sample_structure.format(sent, entity1, entity2, label)
        empty_prompt = empty_prompt_sample_structure.format(sent, entity1, entity2)

        prompts.append(prompt)
        empty_prompts.append(empty_prompt)
        labels.append(label)

    df['prompts'] = prompts
    df['empty_prompts'] = empty_prompts
    df['verbalized_label'] = labels
    unique_labels = set(labels)
    df['unique_labels'] = [unique_labels for _ in empty_prompts]

    return df

def add_ner_prompts(df, prompt_config, sep):
    """
    Combining sentences and entities to create prompts in Dataframe with 'sents' and 'entities' columns.
    Adds 'prompts' and 'empty_prompts' (prompt without answer) columns to DataFrame
    """

    prompts = []
    empty_prompts = []

    prompt_sample_structure = prompt_config['sent_intro'] + ' {}\n' + prompt_config['retrieval_message'] + ' {}'
    empty_prompt_sample_structure = prompt_config['sent_intro'] + ' {}\n' + prompt_config['retrieval_message']

    for i, row in df.iterrows():
        sent = row['sents']
        entities = sep.join(row['entities'])

        prompt = prompt_sample_structure.format(sent, entities)
        empty_prompt = empty_prompt_sample_structure.format(sent)

        prompts.append(prompt)
        empty_prompts.append(empty_prompt)

    df['prompts'] = prompts
    df['empty_prompts'] = empty_prompts

    return df

def extract_entities(df, sep='/'):
    # Extracting entities based on BIO Tags with columns 'sents','ner_seq'
    entities = []

    for i, row in df.iterrows():
        sent = row['sents'].split()
        bio_tags = row['ner_seq'].split()

        sent_entities = []
        current_ent = []

        num_bi_tags = 0

        for token, bio in zip(sent, bio_tags):
            if bio.startswith('B') or bio.startswith('O'):
                if len(current_ent) > 0:
                    #                     current_ent = ' '.join(current_ent).lower()
                    current_ent = ' '.join(current_ent)
                    if sep in current_ent:
                        print('woah, separator found in example {}'.format(current_ent))
                    sent_entities.append(current_ent)

                    current_ent = []

                if bio.startswith('B'):
                    current_ent.append(token)
                    num_bi_tags += 1

            elif bio.startswith('I'):
                current_ent.append(token)
                num_bi_tags += 1

        # Add Entity at end of sentence
        if len(current_ent) > 0:
            #             current_ent = ' '.join(current_ent).lower()
            current_ent = ' '.join(current_ent)
            if sep in current_ent:
                print('woah, separator found in example {}'.format(current_ent))
            sent_entities.append(current_ent)

        assert num_bi_tags == len(' '.join(sent_entities).split()), ipdb.set_trace()

        unique_sent_entities = []
        for ent in sent_entities:
            if ent not in unique_sent_entities:
                unique_sent_entities.append(reverse_token_preprocessing(ent))

        entities.append(unique_sent_entities)

    df['entities'] = entities
    df['num_entities'] = [len(e) for e in entities]
    df['num_tokens'] = [len(gpt_2_tokenizer.encode(sep.join(set(ents)))) for ents in df.entities]

    return df

def get_embeddings(sents, model, tokenizer, mode='cls'):
    embeddings = []

    with torch.no_grad():
        for sent in tqdm(sents):
            embedding = get_embedding(sent, model, tokenizer, mode=mode)
            embeddings.append(embedding)

    embeddings = np.array(embeddings)
    norm_embeddings = embeddings.T / np.linalg.norm(embeddings, axis=1)

    return norm_embeddings.T

def get_embedding(sent, model, tokenizer, mode='cls'):

    input_dict = tokenizer(sent, return_tensors='pt').to('cuda')
    embedding = model(**input_dict)['last_hidden_state'].cpu().numpy()[0]

    if mode == 'cls':
        embedding = embedding[0]
    elif mode == 'avg':
        embedding = np.mean(embedding, axis=0)

    return embedding

def get_prompts_from_df(train, dev, prompt_size, sampling_strategy, random_seed=42,
                        cross_val=False):
    if sampling_strategy == 'random':
        dev = get_random_prompts(train, dev, prompt_size, cross_val, random_seed)
    else:
        dev = get_bert_knn_prompts(train, dev, prompt_size, sampling_strategy, cross_val)

    return dev

def get_random_prompts(train, dev, prompt_size, cross_val, random_seed=42):
    if cross_val:
        dev = train.copy()
        dev = dev.reset_index()

    random_prompts = []

    for i, row in dev.iterrows():

        if cross_val:
            curr_train = train[train.index != i]
        else:
            curr_train = train

        prompt_samples = curr_train.sample(prompt_size, random_state=np.random.RandomState(random_seed+i))
        prompt_samples = prompt_samples.prompts.values

        empty_prompt = row['empty_prompts']

        random_prompts.append('\n\n'.join(prompt_samples) + '\n\n' + empty_prompt)

    dev['test_ready_prompts'] = random_prompts
    dev['prompt_samples'] = 'random'

    return dev

def get_bert_knn_prompts(train, dev, prompt_size, sampling_strategy, cross_val=False):
    if cross_val:
        dev = train.copy()
        dev = dev.reset_index()

    bert_model = sampling_strategy
    mode = 'cls'

    model = AutoModel.from_pretrained(bert_model).to('cuda')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    train_embeddings = get_embeddings(train.sents.values, model, tokenizer, mode)

    knn_prompt_samples = []
    knn_prompts = []

    with torch.no_grad():
        for i, row in dev.iterrows():
            test_sent = row['sents']

            sent_emb = get_embedding(test_sent, model, tokenizer, mode=mode)
            sent_emb = sent_emb / np.linalg.norm(sent_emb)

            if cross_val:
                if i == 0:
                    cross_val_train_embeddings = train_embeddings[1:]
                elif i == len(dev):
                    cross_val_train_embeddings = train_embeddings[-1:]
                else:
                    cross_val_train_embeddings = np.vstack((train_embeddings[:i], train_embeddings[i + 1:]))

                assert len(cross_val_train_embeddings) == len(dev) - 1, ipdb.set_trace()
                sims = cross_val_train_embeddings.dot(sent_emb)
                indices = np.concatenate([range(i), range(i+1,len(train_embeddings))])
            else:
                sims = train_embeddings.dot(sent_emb)
                indices = np.arange(len(train_embeddings))

            real_indices = indices.astype('int')
            sorted_indices = np.argsort(sims, kind='stable')[::-1]
            assert sims[sorted_indices[0]] > sims[sorted_indices[-1]], ipdb.set_trace()
            selected_sims = sims[sorted_indices][:prompt_size]

            real_sorted_indices = real_indices[sorted_indices]
            selected_real_indices = real_sorted_indices[:prompt_size]
            selected_prompts = train.prompts.values[selected_real_indices]

            empty_prompt = row['empty_prompts']

            knn_prompt_samples.append((selected_prompts, selected_sims))
            knn_prompts.append('\n\n'.join(selected_prompts) + '\n\n' + empty_prompt)

        dev['prompt_samples'] = knn_prompt_samples
        dev['test_ready_prompts'] = knn_prompts

    return dev
