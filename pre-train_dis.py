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
from dataset_utils import save_tokenized_dataset, get_batches, Corpus
from training_utils import save_gan_training_params, load_gan_training_params
from training_utils import save_example_generation, save_training_log

dis_model_choices = ['basic_dense_ff','rnn','conv_net','relgan']
pretrain_proc_choices = ['default','other']
dataset_choices = ['EZ','J2015','SJ2015','SJ2015S','OF']
criterion_choices = ['CE','BCE','BCEwL']

parser = argparse.ArgumentParser(description='Help description')
parser.add_argument('--dis-model', choices=dis_model_choices, required=True, help='discriminator model')
parser.add_argument('--pretrain-proc', choices=pretrain_proc_choices, required=True, help='training procedure')
parser.add_argument('--dataset', choices=dataset_choices, required=True)
parser.add_argument('--criterion', choices=criterion_choices, required=True, help='Criterion for calculating losses')
parser.add_argument('--num-epochs', type=int, default=5, help='num epochs (default=5)')
parser.add_argument('--embed-dim', type=int, default=32, help='embed dim (default=32)')
parser.add_argument('--lstm-size', type=int, default=32, help='lstm size (default=32)')
parser.add_argument('--batch-size', type=int, default=16, help='batch size (default=16)')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--force-reload', action='store_true', help='Force reload the datasets, or just retrieve preread version if available (default=False)')
parser.add_argument('--seq-len', type=int, default=16, help='Sequence length during training')
parser.add_argument('--dis-lr', type=float, default=0.0004, help='Discriminator learning rate')
parser.add_argument('--N-save-model', type=int, default=20, help='How many models to save during training, besides the final model')
parser.add_argument('--begin-training', action='store_true', help='Begin training immediately')
parser.add_argument('--save-graph', action='store_true', help='Save graph immediately')
parser.add_argument('--show-graph', action='store_true', help='Show graph after generating it')
parser.add_argument('--load-dis', default=None, help='File containing pre-trained discriminator state dict')
parser.add_argument('--load-checkpoint', default=None, help='File containing pre-trained discriminator and other objects')
parser.add_argument('--scramble-text', action='store_true', help='Whether to scramble the fake data for each batch')
parser.add_argument('--plot-name', default=None, help='Name for the output image training plot')

args = parser.parse_args()

print("\n")
use_cuda = args.cuda

if torch.cuda.is_available():
    if not use_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if use_cuda:
        raise Exception("CUDA device not found")

device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    print("Computation: GPU")
else:
    print("Computation: CPU")

print("-" * 80)

np.random.seed(args.seed); # Fix seed
torch.manual_seed(args.seed); # Fix seed

# Load the scrambled fake sentences corpus as well
dataset=None
if args.dataset == 'EZ':
    dataset = save_tokenized_dataset('./dataset/preread/ez_cs/',
                                        '../dataset/ez_cs_dataset/train.txt',
                                        '../dataset/ez_cs_dataset/dev.txt',
                                        '../dataset/ez_cs_dataset/test.txt',
                                        # lambda sentence: sentence.replace("_zul","").replace("_eng",""),
                                        lambda x: x,
                                        "<s>", "</s>", force_reload=args.force_reload, skip_count_check=True)
    scrambled_dataset = Corpus('../dataset/ez_cs_dataset/scrambled-fixed.txt')
    scrambled_dataset.clean_tokenize(lambda sentence: sentence.replace("_zul","").replace("_eng",""), 
                                    start_identifier="<s>", end_identifier="</s>", remove_words_with_count_less_than=0, skip_count_check=True)
elif args.dataset == 'J2015':
    dataset = save_tokenized_dataset('./dataset/preread/johnnic_2015_cleaned_shuffled/',
                                        './dataset/johnnic_2015_cleaned_shuffled/train.txt',
                                        './dataset/johnnic_2015_cleaned_shuffled/dev.txt',
                                        './dataset/johnnic_2015_cleaned_shuffled/test.txt',
                                        lambda x : x,
                                        None, None, force_reload=args.force_reload, skip_count_check=True)
    scrambled_dataset = Corpus('./dataset/johnnic_2015_cleaned_shuffled/scrambled-fixed.txt')
    scrambled_dataset.clean_tokenize(lambda x : x, skip_count_check=True)
elif args.dataset == 'SJ2015':
    dataset = save_tokenized_dataset('./dataset/preread/smaller_johnnic/',
                                        './dataset/smaller_johnnic/train.txt',
                                        './dataset/smaller_johnnic/dev.txt',
                                        './dataset/smaller_johnnic/test.txt',
                                        lambda x : x,
                                        None, None, force_reload=args.force_reload, skip_count_check=True)
    scrambled_dataset = Corpus('./dataset/smaller_johnnic/scrambled-fixed.txt')
    scrambled_dataset.clean_tokenize(lambda x : x, skip_count_check=True)
elif args.dataset == 'SJ2015S':
    dataset = save_tokenized_dataset('./dataset/preread/smaller_johnnic_shuffled/',
                                        './dataset/smaller_johnnic_shuffled/train.txt',
                                        './dataset/smaller_johnnic_shuffled/dev.txt',
                                        './dataset/smaller_johnnic_shuffled/test.txt',
                                        lambda x : x,
                                        None, None, force_reload=args.force_reload, skip_count_check=True)
    scrambled_dataset = Corpus('./dataset/smaller_johnnic_shuffled/scrambled-fixed.txt')
    scrambled_dataset.clean_tokenize(lambda x : x, skip_count_check=True)
elif args.dataset == 'OF':
    dataset = save_tokenized_dataset('./dataset/preread/overfit/',
                                        './dataset/overfit/train.txt',
                                        './dataset/overfit/dev.txt',
                                        './dataset/overfit/test.txt',
                                        lambda x : x,
                                        None, None, force_reload=args.force_reload, skip_count_check=True)
    scrambled_dataset = Corpus('./dataset/overfit/scrambled-fixed.txt')
    scrambled_dataset.clean_tokenize(lambda x : x, skip_count_check=True)
else:
    raise Exception("Invalid dataset. Specify --dataset and the name")


print("Pre-train-Dis\n")
dataset_name = args.dataset
print("Dataset:", dataset_name)
print("Vocab size:",len(dataset.dictionary.idx2word))
print("-" * 80)

if args.dis_model == 'basic_dense_ff':
    from models.discriminators.basic_dense_ff.discriminator import Discriminator_model_dense_ff as Discriminator
elif args.dis_model == 'rnn':
    from models.discriminators.rnn.discriminator import Discriminator_model_rnn as Discriminator
elif args.dis_model == 'conv_net':
    raise Exception("Not yet implemented")
elif args.dis_model == 'relgan':
    from models.discriminators.relgan.discriminator import Discriminator_model_relgan as Discriminator
else:
    raise Exception("Invalid discriminator model. Specify --dis-model and the name")

if args.pretrain_proc == 'default':
    from models.pre_training_procedures.default.dis_default import pretrain_dis as pretrain_dis
elif args.pretrain_proc == 'other':
    raise Exception("Not yet implemented")
else:
    raise Exception("Invalid training procedure. Specify --train-proc and the name")

if args.criterion == 'CE':
    criterion = nn.CrossEntropyLoss()
elif args.criterion == 'BCE':
    criterion = nn.BCELoss()
elif args.criterion == 'BCEwL':
    criterion = nn.BCEWithLogitsLoss()
else:
    raise Exception("Invalid criterion. Specify --criterion and the name")

# print remaining training parameters
current_run_desc = "pre-train_dis-" + args.dis_model + "-" + args.pretrain_proc
embedding_size = args.embed_dim
lstm_size = args.lstm_size
batch_size = args.batch_size
dis_lr = args.dis_lr

num_epochs =  args.num_epochs
seq_len = args.seq_len

# since fake data is from derived from the real data it will have the same num_batches
num_batches = len(dataset.training_dataset) // (batch_size*seq_len)

print("Discriminator model:",args.dis_model)
if args.load_checkpoint is not None:
    print("\tLoading checkpoint:",args.load_checkpoint)
if args.load_dis is not None:
    print("\tLoading discriminator:",args.load_dis)
print("Pre-training procedure:",args.pretrain_proc)
print("Criterion:",args.criterion)
print("-" * 80)
print("Word embedding dimensions:",embedding_size)
print("LSTM hidden state dimensions:",lstm_size)
print("-" * 80)
print("Batch size:",batch_size)
print("Sequence length:", seq_len)
print("Num epochs:",num_epochs)
print("Num mini-batches:", num_batches)
print(args.N_save_model+1,"models saved during training at iterations:")
print([num_epochs*num_batches//args.N_save_model*i+1 for i in range(args.N_save_model)]+[(num_epochs)*(num_batches)])
print("-" * 80)
print("Discriminator learning rate:", dis_lr)
if args.scramble_text: print("Scrambling text each epoch")
print()

# if save 0 models, then set to -1, to avoid division by zero, but still prevent saving additional models
if args.N_save_model < 1:
    args.N_save_model = -1

if not args.begin_training:
    if input("Begin training? y/n\n") != "y":
        print("-" * 80)
        print("Exiting...")
        sys.exit()

################
# Start training
################
vocab_size = len(dataset.dictionary.idx2word)
int_to_vocab, vocab_to_int = dataset.dictionary.idx2word, dataset.dictionary.word2idx
current_run_time = datetime.now().strftime('%Y-%m-%d_%H-%M') # time for saving filename appropriately

dis = Discriminator(use_cuda, vocab_size, batch_size, seq_len, embedding_size, lstm_size)

if use_cuda:
    dis.cuda()

dis_optimiser = torch.optim.Adam(dis.parameters(), lr=dis_lr)

# load pre-trained models
if args.load_checkpoint is not None:
    load_gan_training_params(args.load_checkpoint, dis_model=dis, dis_optimiser=dis_optimiser)

if args.load_dis is not None:
    state = torch.load(args.load_dis)
    dis.load_state_dict(state['dis_model'])
    dis_optimiser.load_state_dict(state['dis_optimiser'])

# save training params to file
save_dir = 'pre-trained_save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+'/'
os.makedirs(save_dir, exist_ok=True)
with open(save_dir+'training_params.txt', "w") as tp:
    tp.write("Computation: GPU\n" if use_cuda else "Computation: CPU\n")
    tp.write("Dataset: "+ dataset_name+'\n')
    tp.write("Vocab size: "+ str(vocab_size)+'\n')
    tp.write("Discriminator model: " + args.dis_model + '\n')
    if args.load_checkpoint is not None:
        tp.write("Loading checkpoint: " + args.load_checkpoint + '\n')
    if args.load_dis is not None:
        tp.write("Loading discriminator: " + args.load_dis + '\n')
    tp.write("Criterion: "+str(args.criterion)+'\n')
    tp.write("Pretraining procedure: " + args.pretrain_proc + '\n')
    tp.write("Word embedding dimensions: " + str(embedding_size) + '\n')
    tp.write("LSTM hidden state dimensions: " +str(lstm_size) +'\n')
    tp.write("Batch size: "+ str(batch_size)+'\n')
    tp.write("Sequence length: "+ str(seq_len)+'\n')
    tp.write("Num epochs: "+ str(num_epochs)+'\n')
    tp.write("Num mini-batches: "+ str(num_batches)+'\n')
    tp.write(str(args.N_save_model) + "models saved during training at iterations:\n")
    tp.write(str([num_epochs*num_batches//args.N_save_model*i for i in range(args.N_save_model)]+[(num_epochs)*(num_batches)-1])+'\n')
    tp.write("Discriminator learning rate:" + str(dis_lr) +'\n')
    tp.write('\n')
    tp.write('\n--load-dis ' + save_dir + '\n\n')
    tp.write("python pre-train_dis.py")
    tp.write(" --dis-model " + args.dis_model +  " --pretrain-proc " + args.pretrain_proc +  " --dataset " + args.dataset + " --criterion " + args.criterion)
    tp.write(" --load-checkpoint " + str(args.load_checkpoint) + " --load-dis " + str(args.load_dis))
    tp.write(" --num-epochs " + str(args.num_epochs) + " --embed-dim " + str(args.embed_dim) +  " --lstm-size " + str(args.lstm_size) +  " --batch-size " + str(args.batch_size))
    tp.write(" --seed " + str(args.seed) + " --seq-len " + str(args.seq_len))
    tp.write(" --dis-lr " + str(args.dis_lr) + " --N-save-model " + str(args.N_save_model))
    if args.plot_name is not None:
        tp.write(" --plot-name " + str(args.plot_name))
    if args.scramble_text: tp.write(" --scramble-text")
    if args.force_reload: tp.write(" --force-reload")
    if use_cuda: tp.write(" --cuda")
    tp.close()

print()
print('#' * 80)
print()
print("Started:", current_run_time)
print()
_rows_to_erase = 0 # for updating the training parameters

try:
    # Put model in training mode (enables things like dropout and batchnorm)
    dis.train()
    
    for epoch in range(num_epochs):
        if args.scramble_text:
            import subprocess
            if dataset_name == "EZ": # does not contain start/stop words
                cmd ="""
                awk '
                BEGIN {srand()}
                {
                    orig = $0;
                    if (NF > 1){
                        num_dupl=0; 
                        for(i=1;i<NF;i++){
                            if($i==$(i+1)) 
                                num_dupl++;
                        }                
                        if(num_dupl<(NF-1)){ # if there are 3 repeated words, then num_dupl will =2, hence the NF-1
                            while ($0 == orig) {
                                for (i = 1; i <= NF; i++) {
                                    r = int(rand() * (NF)) + 1
                                    x = $r; $r = $i; $i = x
                                }
                            }
                        }
                    }
                    print $0
                }'

                """
            else:
                cmd = """
                awk '
                BEGIN {srand()} {
                    orig = $0;
                    if (NF > 3){
                        num_dupl=0; 
                        for(i=1;i<NF;i++){
                            if($i==$(i+1)) 
                                num_dupl++;
                        }                
                        if(num_dupl<(NF-1)){ # if there are 3 repeated words, then num_dupl will =2, hence the NF-1
    	                    while ($0 == orig) {
                                for (i = 2; i < NF; i++) {
                                    r = int(rand() * (NF-2)) + 2
                                    x = $r; $r = $i; $i = x
                                }
                            }
                        }
                    }
                    print $0
                }'
            """

            input_training_text_file = open(dataset.train_path)
            
            output_scrambled_text_file = open('pre-trained_save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+"/temp-scrambled.txt","w")

            subprocess.call(cmd,stdin=input_training_text_file,stdout=output_scrambled_text_file, shell=True)
            input_training_text_file.close()
            output_scrambled_text_file.close()

            scrambled_dataset = Corpus('pre-trained_save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+"/temp-scrambled.txt")
            scrambled_dataset.clean_tokenize(lambda x : x, skip_count_check=True)
            
            os.remove('pre-trained_save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+"/temp-scrambled.txt")

        batches_real = get_batches(dataset.training_dataset, batch_size, seq_len)
        batches_fake = get_batches(scrambled_dataset.train, batch_size, seq_len)
        
        for n_batch in range(num_batches):
            batch_real = next(batches_real)
            batch_fake = next(batches_fake)
            iterations = epoch*num_batches+n_batch
            
            # Transfer data to GPU
            if use_cuda: batch_real, batch_fake = batch_real.cuda(), batch_fake.cuda()

            # Train Discriminator on real/fake data
            dis_error, d_pred_real, d_pred_fake = pretrain_dis(dis, dis_optimiser, criterion, batch_real, batch_fake, use_cuda)
            dis_error, d_pred_real, d_pred_fake = dis_error.item(), torch.mean(d_pred_real).item(), torch.mean(d_pred_fake).item()

            save_training_log(epoch, n_batch, dis_error=dis_error, dataset_name=dataset_name, current_run_desc=current_run_desc, current_run_time=current_run_time, save_folder="pre-trained_save/")

            if iterations % ((num_batches*num_epochs)//args.N_save_model) == 0 or iterations == (num_epochs)*(num_batches)-1: # save N instances of the model, and the final model
                # Save the current training model
                save_gan_training_params(dis_model=dis, dis_optimiser=dis_optimiser, epoch=epoch, 
                                            iteration=n_batch, dis_error=dis_error, dataset_name=dataset_name, 
                                            current_run_desc=current_run_desc, current_run_time=current_run_time, save_folder="pre-trained_save/")

            # clear lines
            sys.stdout.write("\033[F\033[K"*(_rows_to_erase))

            rows, columns = os.popen('stty size', 'r').read().split() # get width (columns) of terminal
            # set the amount after, so that when the example text changes, it erases the previous text
            _rows_to_erase = 5
            
            # Print current training parameters
            print('Epoch: {}/{}'.format(epoch+1, num_epochs),
                    '\nBatch number: {}/{}'.format(n_batch+1, num_batches),
                    '\nPred real: {}'.format(d_pred_real),
                    '\nPred fake: {}'.format(d_pred_fake),
                    '\nTotal Loss: {}'.format(dis_error))
                    # maybe add real/fake loss
            
except KeyboardInterrupt:
    print()
    print('-' * 80)
    print('Exiting from training early')

print()
print('#' * 80)
print()

# save graph
import csv
filepath = 'pre-trained_save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+'/log/log.txt'

with open(filepath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first_line = True
    dis_loss = []
    
    start_time = None
    stop_time = None
    for n, row in enumerate(csv_reader):
        if first_line:
            first_line = False
            continue
        else:
            if n == 1:
                start_time = row[5].replace("-",":")
            # epoch*num_batches + iteration
            dis_loss.append(round(float(row[3]),5))
            
            stop_time = row[5]
    stop_time = stop_time.replace("-",":")
    csv_file.close()

#plot graph
plt.figure(figsize=(15,5))
plt.suptitle("'"+current_run_desc+"' on the '"+dataset_name+"' dataset. Training started at "+start_time[:-3]+" stopped at "+stop_time[:-3])

plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.plot(range(len(dis_loss)), dis_loss, color='g', label="dis_loss")
plt.legend(loc='upper center')

if not args.save_graph:
    if input("Save output training graph? y/n\n") == 'y':
        if args.plot_name is not None:
            plot_filename = str(args.plot_name) + ".png"
        else:
            plot_filename = "output.png"
        plt.savefig('pre-trained_save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+"/" + plot_filename)
else:
    if args.plot_name is not None:
        plot_filename = str(args.plot_name) + ".png"
    else:
        plot_filename = "output.png"
    plt.savefig('pre-trained_save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+"/" + plot_filename)
    
if args.show_graph:
    plt.show()
