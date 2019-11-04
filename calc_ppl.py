#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 MPL Ribeiro
# 
# This file is part of a final year undergraduate project for
# generating discrete text sequences using generative adversarial
# networks (GANs)
# 
# GNU GPL-3.0-or-later

import os
import re
import sys
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from IPython import display

# PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# My modules
from dataset_utils import save_tokenized_dataset, get_batches
from training_utils import save_gan_training_params, load_gan_training_params
from training_utils import save_example_generation, save_training_log

gen_model_choices = ['default']
dataset_choices = ['EZ','J2015','SJ2015','SJ2015S','OF']

parser = argparse.ArgumentParser(description='Help description')
parser.add_argument('--load-checkpoint', default=None, help='File containing pre-trained generator among other things')
parser.add_argument('--load-gen', default=None, help='File containing pre-trained generator state dict')
parser.add_argument('--dataset', choices=dataset_choices, required=True)
parser.add_argument('--seq-len', type=int, default=32, help='Sequence length to generate for each sentence')
parser.add_argument('--top-k', type=int, default=5, help='top k choice when selecting next word token')
parser.add_argument('--num-sentences-portion', type=float, default=0.5, help='Number representing the percentage of training text sentence count, to generate. (1.0 means generate the same number of sentences that training text contains)')
parser.add_argument('--training-text-portion', type=float, default=1.0, help='Percentage of the original text to keep (value between 0 and 1)')

# enter the params of the gen that will be loaded
parser.add_argument('--gen-model', choices=gen_model_choices, default="default", help='generator model')
parser.add_argument('--embed-dim', type=int, default=32, help='embed dim (default=32)')
parser.add_argument('--lstm-size', type=int, default=32, help='lstm size (default=32)')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--force-reload', action='store_true', help='Force reload the datasets, or just retrieve preread version if available (default=False)')

parser.add_argument('--begin-calculation', action='store_true', help='Begin calculation immediately')
parser.add_argument('--delete-duplicates', action='store_true', help='Delete duplicate sentences')
parser.add_argument('--delete-generated-text', action='store_true', help='Delete generated sentences immediately after done calculating perplexity')
parser.add_argument('--use-test-set', action='store_true', help='Whether to use test set or dev set to calculate perplexity')
parser.add_argument('--suppress-print', action='store_true', help='Suppress print commands')
parser.add_argument('--calc-original-ppl', action='store_true', help='Calculate the ppl using the original train set and dev/test set')
parser.add_argument('--save-dir', default="./", help='Directory to save the ppl value')

args = parser.parse_args()

if not args.suppress_print:
    print("\n")
use_cuda = args.cuda

if torch.cuda.is_available():
    if not use_cuda:
        if not args.suppress_print:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if use_cuda:
        raise Exception("CUDA device not found")

device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    if not args.suppress_print:
        print("Computation: GPU")
else:
    if not args.suppress_print:
        print("Computation: CPU")

if not args.suppress_print:
    print("-" * 80)

np.random.seed(args.seed); # Fix seed
torch.manual_seed(args.seed); # Fix seed

dataset=None
if args.dataset == 'EZ':
    dataset = save_tokenized_dataset('./dataset/preread/ez_cs/',
                                        '../dataset/ez_cs_dataset/train.txt',
                                        '../dataset/ez_cs_dataset/dev.txt',
                                        '../dataset/ez_cs_dataset/test.txt',
                                        # lambda sentence: sentence.replace("_zul","").replace("_eng",""),
                                        lambda x: x,
                                        "<s>", "</s>", force_reload=args.force_reload, skip_count_check=True)
elif args.dataset == 'J2015':
    dataset = save_tokenized_dataset('./dataset/preread/johnnic_2015_cleaned_shuffled/',
                                        './dataset/johnnic_2015_cleaned_shuffled/train.txt',
                                        './dataset/johnnic_2015_cleaned_shuffled/dev.txt',
                                        './dataset/johnnic_2015_cleaned_shuffled/test.txt',
                                        lambda x : x,
                                        None, None, force_reload=args.force_reload, skip_count_check=True)
elif args.dataset == 'SJ2015':
    dataset = save_tokenized_dataset('./dataset/preread/smaller_johnnic/',
                                        './dataset/smaller_johnnic/train.txt',
                                        './dataset/smaller_johnnic/dev.txt',
                                        './dataset/smaller_johnnic/test.txt',
                                        lambda x : x,
                                        None, None, force_reload=args.force_reload, skip_count_check=True)
elif args.dataset == 'SJ2015S':
    dataset = save_tokenized_dataset('./dataset/preread/smaller_johnnic_shuffled/',
                                        './dataset/smaller_johnnic_shuffled/train.txt',
                                        './dataset/smaller_johnnic_shuffled/dev.txt',
                                        './dataset/smaller_johnnic_shuffled/test.txt',
                                        lambda x : x,
                                        None, None, force_reload=args.force_reload, skip_count_check=True)
elif args.dataset == 'OF':
    dataset = save_tokenized_dataset('./dataset/preread/overfit/',
                                        './dataset/overfit/train.txt',
                                        './dataset/overfit/dev.txt',
                                        './dataset/overfit/test.txt',
                                        lambda x : x,
                                        None, None, force_reload=args.force_reload, skip_count_check=True)
else:
    raise Exception("Invalid dataset. Specify --dataset and the name")


dataset_name = args.dataset
if not args.suppress_print:
    print("Dataset:", dataset_name)
    print("Vocab size:",len(dataset.dictionary.idx2word))


if args.gen_model == 'default':
    from models.generators.default.generator import Generator_model as Generator
else:
    raise Exception("Invalid generator model. Specify --gen-model and the name")

current_run_desc = args.gen_model
embedding_size = args.embed_dim
lstm_size = args.lstm_size

# calculate number of lines in training text and num lines to use from it
try:
    proc_sen_count = subprocess.Popen(['wc', '-l', dataset.train_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o, e = proc_sen_count.communicate()
    num_lines_training_text = int(o.decode('ascii').split(' ',maxsplit=1)[0]) # split the text at whitespace, stop after one occurrence
    num_lines_to_use = int(args.training_text_portion*num_lines_training_text)
except:
    print("Failed to calculate perplexity")
    with open(os.path.join(args.save_dir,"cur_ppl.txt"),"w") as p:
        p.write("-1")
        p.close()
    sys.exit(-1)

seq_len = args.seq_len
top_k = args.top_k
num_sentences = int(num_lines_training_text*args.num_sentences_portion)

from math import ceil
batch_size = max(min(100, num_sentences), ceil(0.1*num_sentences)) # batch size is 1/10th of sentences to generate, at least 100 (or num of sentences)
num_batches = ceil(num_sentences/batch_size)

if not args.suppress_print:
    print("Num sentences in training text:", num_lines_training_text)
    print("Num sentences from training text to keep for calculation:", num_lines_to_use)
    print("Num sentences to generate:", num_sentences)
    print("-" * 80)
    print("Generator model:",args.gen_model)
    if args.load_checkpoint is not None:
        print("\tLoading checkpoint:",args.load_checkpoint)
    if args.load_gen is not None:
        print("\tLoading generator:",args.load_gen)
    print("-" * 80)
    print("Word embedding dimensions:",embedding_size)
    print("LSTM hidden state dimensions:",lstm_size)
    print("-" * 80)
    
    print("Batch size:",batch_size)
    print("Sequence length:", seq_len)
    print("Num batches:", num_batches)
    print("Top k:", top_k)
    if args.delete_duplicates:
        print("Delete duplicate sentences")
    if args.delete_generated_text:
        print("Delete generated text after calculation")
    print("-" * 80)
    print()

if not args.begin_calculation:
    if input("Begin calculation? y/n\n") != "y":
        print("-" * 80)
        print("Exiting...")
        sys.exit()

###################
# Start calculating
###################

vocab_size = len(dataset.dictionary.idx2word)
int_to_vocab, vocab_to_int = dataset.dictionary.idx2word, dataset.dictionary.word2idx
current_run_time = datetime.now().strftime('%Y-%m-%d_%H-%M') # time for saving filename appropriately

if not args.calc_original_ppl:
    gen = Generator(use_cuda, vocab_size, batch_size, seq_len, embedding_size, lstm_size)

if not args.calc_original_ppl:
    if use_cuda:
        gen.cuda()

# load pre-trained models
if not args.calc_original_ppl:
    if args.load_checkpoint is not None:
        load_gan_training_params(args.load_checkpoint, gen_model = gen)
        # Get current directory of saved file
        save_dir, filename = os.path.split(args.load_checkpoint)
        save_dir += '/' # add final forward slash, as split command above removes it
        generation_filename = filename + '-generation.txt'
        generation_filename_path = os.path.join(save_dir, generation_filename)

if not args.calc_original_ppl:        
    if args.load_gen is not None:
        state = torch.load(args.load_gen)
        gen.load_state_dict(state['gen_model'])
        # Get current directory of saved file
        save_dir, filename = os.path.split(args.load_gen)
        save_dir += '/' # add final forward slash, as split command above removes it
        generation_filename = filename + '-generation.txt'
        generation_filename_path = os.path.join(save_dir, generation_filename)
    

if not args.suppress_print:
    print()
    print('#' * 80)
    print()
    print("Started:", current_run_time)
    print('\n')
_rows_to_erase = 0 # for updating the training parameters

last_5_times = [10,10,10,10,10] # for printing out the average time per sample
iter_time = time.time()

# remove file, if it already exists

if not args.calc_original_ppl:
    try:
        os.remove(generation_filename_path)
    except OSError:
        pass

if not args.calc_original_ppl:
    print()
    try:
        gen.eval() # change generator into evaluation mode (disables dropout/batch-normalisation)

        for n_batch in range(num_batches):
            # hidden = gen.zero_state(batch_size)
            hidden = (torch.rand(1, batch_size, gen.hidden_state_dim), torch.rand(1, batch_size, gen.hidden_state_dim))

            if use_cuda: hidden = (hidden[0].cuda(), hidden[1].cuda())

            # random_SOS = torch.cat((torch.tensor([[vocab_to_int["<s>"]]]*batch_size),torch.randint(vocab_size,(batch_size,1))),dim=1)
            random_SOS = torch.tensor([[vocab_to_int["<s>"]]]*batch_size) # random start of sentence
            if use_cuda: random_SOS = random_SOS.cuda()
            
            sentences = random_SOS

            initial_length = random_SOS.size(1)
            for word_idx in range(initial_length):
                probs, hidden = gen(random_SOS[:,word_idx].unsqueeze(1), hidden)
            
            num_needed_words = seq_len - initial_length
            for i in range(num_needed_words):
                _, next_possible_tokens = torch.topk(probs, k=top_k)
                # next_possible_tokens size: (batch_size * top_k)
            
                choice_idx = torch.randint(top_k, (batch_size,1))
                if use_cuda: choice_idx = choice_idx.cuda()
                next_token = torch.gather(next_possible_tokens, 1, choice_idx) # get next word token from the chosen index
            
                sentences = torch.cat((sentences, next_token), dim=1) # concatenate the chosen word token to the end of out

                probs, hidden = gen(next_token, hidden)
            
            with open(generation_filename_path,'a') as f:
                for n_sentence in range(batch_size):
                    sentence = [int_to_vocab[word.item()] for word in sentences[n_sentence,:]]
                    sentence = " ".join(sentence)
                    # sentence = sentence.replace(" <s>","\n<s>").replace("</s> ","</s>\n")

                    # sentence = re.sub(r"^(?=[^<])([A-Z\s]*)", r"<s> \1", sentence) # add BOS token
                    # sentence = re.sub(r"^(<s> [A-Z\s]*)$", r"\1 </s>", sentence) # add EOS token

                    f.write(sentence+"\n")

                f.close()

            sys.stdout.write("\033[F\033[K")
            print("Calculating PPL - Batch:", str(n_batch+1) + "/" + str(num_batches))

        gen.train()


    except KeyboardInterrupt:
        print()
        print('-' * 80)
        print('Exiting from generation early')
        f.close()

    # delete line
    sys.stdout.write("\033[F\033[K")

if not args.calc_original_ppl:
    if not args.suppress_print:
        print("Done generating")
try:
    # if using generated text
    if not args.calc_original_ppl:
        if args.delete_duplicates:
            # specify input and output file as same file
            subprocess.call(['sort', '-o',generation_filename_path,'-u',generation_filename_path])

        # calc ppl
        lm_filename_path = os.path.join(save_dir, filename+'-lm.arpa')

        portion_filename_path = os.path.join(save_dir, filename+'-portion.txt')
        portion_file = open(portion_filename_path, 'w')
        
        subprocess.Popen(['shuf', '-n', str(num_lines_to_use), dataset.train_path], stdout=portion_file, stderr=subprocess.PIPE)
        portion_file.close()

        concat_filename_path = os.path.join(save_dir, filename+'-concat.txt')
        concat_file = open(concat_filename_path, 'w')
        # concat training and generated text

        subprocess.Popen(['cat', generation_filename_path, portion_filename_path], stdout=concat_file, stderr=subprocess.PIPE)
        concat_file.close()
        
    # if using original text only
    else:
        concat_filename_path = dataset.train_path
        lm_filename_path = os.path.join(os.path.split(dataset.train_path)[0], 'lm-train.arpa')
            
    # subprocess.call(['ngram-count', '-text', generation_filename_path, '-lm', lm_filename_path])
    _,_ = subprocess.Popen(['ngram-count', '-text', concat_filename_path, '-lm', lm_filename_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    
    if not args.calc_original_ppl:
        os.remove(concat_filename_path)
        os.remove(portion_filename_path)

    if args.use_test_set:
        proc1 = subprocess.Popen(['ngram', '-lm', lm_filename_path, '-ppl', dataset.test_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        proc1 = subprocess.Popen(['ngram', '-lm', lm_filename_path, '-ppl', dataset.valid_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    o, e = proc1.communicate()
    o = o.decode('ascii')

    perplexity = re.search(" ppl= ([0-9]*[\.]*[0-9]*) ", o).group(1)

    if not args.suppress_print:
        print('Perplexity: ' + str(perplexity))
        # print(o)
        # print(time.time()-_start)

    if not args.calc_original_ppl:
        if args.delete_generated_text:
            os.remove(generation_filename_path)
            os.remove(lm_filename_path)

except KeyboardInterrupt:
    print("Failed to calculate perplexity")
    with open(os.path.join(args.save_dir,"cur_ppl.txt"),"w") as p:
        p.write("-1")
        p.close()
    sys.exit(-1)

with open(os.path.join(args.save_dir,"cur_ppl.txt"),"w") as p:
    p.write(perplexity)
    p.close()



