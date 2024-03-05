# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import os
import numpy as np
import random
import torch
from datasets import load_dataset
import hashlib

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


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
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

def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('c4', split='train')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for n in range(nsamples):
        i = random.randint(0, len(traindata) - 1)
        trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
        input_ids = trainenc.input_ids
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            input_ids = torch.cat([input_ids, trainenc.input_ids], 1)
            if input_ids.shape[1] > seqlen:
                break
        inp = input_ids[:, 0:seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, trainloader


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

def mix_cc100(nsamples, seed, seqlen, tokenizer, languages):
    trainloader = []
    for i in range(len(languages)):
        trainloader += get_cc100(nsamples[i], seed, seqlen, tokenizer, languages[i])[0]
    assert len(trainloader)==sum(nsamples), f"Length of dataloader : {len(trainloader)}, sample sum: {sum(nsamples)}"
    return trainloader, trainloader


def get_xlsum(nsamples, seed, seqlen, tokenizer, language):
    # Load train and test datasets
    traindata = load_dataset('csebuetnlp/xlsum', language=language)
    testdata = load_dataset('csebuetnlp/xlsum', language=language)

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['train']['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['validation']['text']), return_tensors='pt')

    # Generate samples from training set
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


def mix_bibles(nsamples, seed, seqlen, tokenizer, languages):
    trainloader = []
    for i in range(len(languages)):
        trainloader += get_bibles(nsamples[i], seed, seqlen, tokenizer, languages[i])[0]
    assert len(trainloader)==sum(nsamples), f"Length of dataloader : {len(trainloader)}, sample sum: {sum(nsamples)}"
    return trainloader, trainloader
    


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, language=None):
    if name=='wikitext2':
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name=="c4":
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if name=="cc100":
        return mix_cc100(nsamples, seed, seqlen, tokenizer, language)
    if name=="xlsum":
        return get_xlsum(nsamples, seed, seqlen, tokenizer, language)
    if name=="mix_bibles":
        return mix_bibles(nsamples, seed, seqlen, tokenizer, language)
    
