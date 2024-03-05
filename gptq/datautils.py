import os
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import hashlib
import random

def hash_string(input_string, length=10):
    # Create a SHA-256 hash object
    sha256 = hashlib.sha256()

    # Update the hash object with the input string encoded as bytes
    sha256.update(input_string.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hex_hash = sha256.hexdigest()

    # Truncate the hash to the desired length
    truncated_hash = hex_hash[:length]

    return truncated_hash


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_xlsum(nsamples, seed, seqlen, model, language):
        
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    # Load train and test datasets
    testdata = load_dataset('csebuetnlp/xlsum', language=language)

    # Encode datasets
    testenc = tokenizer("\n\n".join(testdata['validation']['text']), return_tensors='pt')
    
    return testenc, testenc

def get_wikitext2(nsamples, seed, seqlen, model):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)

    return testenc, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_cc100(nsamples, seed, seqlen, tokenizer, language):
    # Load train and validation datasets
    traindata = load_dataset('cc100', lang=language)
    print(f"One Calibration Sample: {traindata['train'][0]['text']}")
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for n in range(nsamples):
        i = random.randint(0, len(traindata['train']) - 1)
        trainenc = tokenizer(traindata['train'][i]['text'], return_tensors='pt')
        input_ids = trainenc.input_ids
        while True:
            i = random.randint(0, len(traindata['train']) - 1)
            trainenc = tokenizer(traindata['train'][i]['text'], return_tensors='pt')

            input_ids = torch.cat([input_ids, trainenc.input_ids], 1)
            if input_ids.shape[1] > seqlen:
                break
        inp = input_ids[:, 0:seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, trainloader

def get_cc100(nsamples, seed, seqlen, model):

    nsamples=[87,47,37,31,14,13,7,4,3,3,1,1,1,1,1,1,1,1,1,1]
    languages=['en','zh-Hans','fr','es','pt','ar','vi','hi','id','bn','ta','te','ur','ne','mr','gu','zh-Hant','sw','yo','ig']


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    print("Not exist. Loading from scratch...")
    trainloader = []
    for i in range(len(languages)):
        trainloader += get_cc100(nsamples[i], seed, seqlen, tokenizer, languages[i])[0]
    assert len(trainloader)==sum(nsamples), f"Length of dataloader : {len(trainloader)}, sample sum: {sum(nsamples)}"


    return trainloader, trainloader


def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=256, seed=0, seqlen=2048, model='', language = ""
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    # if 'ptb' in name:
    #     if 'new' in name:
    #         return get_ptb_new(nsamples, seed, seqlen, model)
    #     return get_ptb(nsamples, seed, seqlen, model)
    if 'cc100' in name:
        return get_cc100(nsamples, seed, seqlen, model)
    if "xlsum" in name:
        return get_xlsum(nsamples, seed, seqlen, model, language)
