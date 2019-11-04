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
import subprocess
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
dis_model_choices = ['basic_dense_ff','rnn','conv_net','relgan']
train_proc_choices = ['default','ACGD','modified_minimax']
dataset_choices = ['EZ','J2015','SJ2015','SJ2015S','OF']
criterion_choices = ['CE','BCE','BCEwL']

parser = argparse.ArgumentParser(description='Help description')
parser.add_argument('--gen-model', choices=gen_model_choices, required=True, help='generator model')
parser.add_argument('--dis-model', choices=dis_model_choices, required=True, help='discriminator model')
parser.add_argument('--train-proc', choices=train_proc_choices, required=True, help='training procedure')
parser.add_argument('--dataset', choices=dataset_choices, required=True)
parser.add_argument('--criterion', choices=criterion_choices, required=True, help='Criterion for calculating losses')
parser.add_argument('--num-epochs', type=int, default=5, help='num epochs (default=5)')
parser.add_argument('--gen-embed-dim', type=int, default=32, help='generator embed dim (default=32)')
parser.add_argument('--dis-embed-dim', type=int, default=32, help='discriminator embed dim (default=32)')
parser.add_argument('--gen-lstm-size', type=int, default=32, help='generator lstm size (default=32)')
parser.add_argument('--dis-lstm-size', type=int, default=32, help='discriminator lstm size (default=32)')
parser.add_argument('--batch-size', type=int, default=16, help='batch size (default=16)')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--force-reload', action='store_true', help='Force reload the datasets, or just retrieve preread version if available (default=False)')
parser.add_argument('--seq-len', type=int, default=16, help='Sequence length during training')
parser.add_argument('--top-k', type=int, default=1, help='top k choice when selecting next word token')
parser.add_argument('--init-temp', type=float, default=5.0, help='The initial temperature parameter for Gumbel-Softmax')
parser.add_argument('--N-temp', type=float, default=0.5, help='Percentage of the total iterations before temperature is annealed to 1. (value between and 1)')
parser.add_argument('--temp-control', default='lin', help='The control policy for annealing the temperature parameter for Gumbel-Softmax')
parser.add_argument('--gen-lr', type=float, default=0.0004, help='Generator learning rate')
parser.add_argument('--dis-lr', type=float, default=0.0004, help='Discriminator learning rate')
parser.add_argument('--N-save-model', type=int, default=20, help='How many models to save during training, besides the final model')
parser.add_argument('--begin-training', action='store_true', help='Begin training immediately')
parser.add_argument('--save-graph', action='store_true', help='Save graph immediately')
parser.add_argument('--show-graph', action='store_true', help='Show graph after generating it')
parser.add_argument('--load-gen', default=None, help='File containing pre-trained generator state dict')
parser.add_argument('--load-dis', default=None, help='File containing pre-trained discriminator state dict')
parser.add_argument('--load-checkpoint', default=None, help='File containing pre-trained generator and discriminator as well as their respective optimisers state dict')
parser.add_argument('--num-sentences-portion', type=float, default=0.5, help='Number representing the percentage of training text sentence count, to generate. (1.0 means generate the same number of sentences that training text contains)')
parser.add_argument('--training-text-portion', type=float, default=1.0, help='Percentage of the original text to keep (value between 0 and 1)')
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
if args.plot_name is not None:
    print(args.plot_name)
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
print("Dataset:", dataset_name)
print("Vocab size:",len(dataset.dictionary.idx2word))
print("-" * 80)

if args.gen_model == 'default':
    from models.generators.default.generator import Generator_model as Generator
else:
    raise Exception("Invalid generator model. Specify --gen-model and the name")

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

if args.train_proc == 'default':
    from models.training_procedures.default.training_gen_and_dis import train_generator as train_gen, train_discriminator as train_dis
elif args.train_proc == 'ACGD':
    from models.training_procedures.acgd.training_gen_and_dis import train_generator as train_gen, train_discriminator as train_dis
elif args.train_proc == 'modified_minimax':
    from models.training_procedures.modified_minimax_loss.training_gen_and_dis import train_generator as train_gen, train_discriminator as train_dis
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
current_run_desc = args.gen_model + "-" + args.dis_model + "-" + args.train_proc
gen_embedding_size = args.gen_embed_dim
dis_embedding_size = args.dis_embed_dim
gen_lstm_size = args.gen_lstm_size
dis_lstm_size = args.dis_lstm_size
batch_size = args.batch_size
gen_lr = args.gen_lr
dis_lr = args.dis_lr

num_epochs =  args.num_epochs
seq_len = args.seq_len
top_k = args.top_k
initial_temperature = args.init_temp
temp_control_policy = args.temp_control
num_batches = len(dataset.training_dataset) // (batch_size*seq_len)
N_temperature = int(num_batches*num_epochs*args.N_temp)

print("Generator model:",args.gen_model)
print("Discriminator model:",args.dis_model)
if args.load_checkpoint is not None:
    print("\tLoading checkpoint:",args.load_checkpoint)
if args.load_gen is not None:
    print("\tLoading generator:",args.load_gen)
if args.load_dis is not None:
    print("\tLoading discriminator:",args.load_dis)
print("Training procedure:",args.train_proc)
print("Criterion:",args.criterion)
print("-" * 80)
print("Generator Word embedding dimensions:",gen_embedding_size)
print("Discriminator Word embedding dimensions:",dis_embedding_size)
print("Generator LSTM hidden state dimensions:",gen_lstm_size)
print("Discriminator LSTM hidden state dimensions:",dis_lstm_size)
print("-" * 80)
print("Batch size:",batch_size)
print("Sequence length:", seq_len)
print("Num epochs:",num_epochs)
print("Num mini-batches:", num_batches)
print(args.N_save_model+1,"models saved during training at iterations:")
print([num_epochs*num_batches//args.N_save_model*i+1 for i in range(args.N_save_model)]+[(num_epochs)*(num_batches)])
print("-" * 80)
print("Top k:", top_k)
print("Initial temperature:", initial_temperature)
print("Temperature annealed to 1 after", N_temperature, "iterations")
print("Temperature control policy:", temp_control_policy)
print("-" * 80)
print("Generator learning rate:", gen_lr)
print("Discriminator learning rate:", dis_lr)
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

gen = Generator(use_cuda, vocab_size, batch_size, seq_len, gen_embedding_size, gen_lstm_size, initial_temperature, control_policy=temp_control_policy)
dis = Discriminator(use_cuda, vocab_size, batch_size, seq_len, dis_embedding_size, dis_lstm_size)

if use_cuda:
    gen.cuda()
    dis.cuda()

gen_optimiser = torch.optim.Adam(gen.parameters(), lr=gen_lr)
dis_optimiser = torch.optim.Adam(dis.parameters(), lr=dis_lr)

# load pre-trained models
if args.load_checkpoint is not None:
    load_gan_training_params(args.load_checkpoint, gen, dis, gen_optimiser, dis_optimiser)

if args.load_gen is not None:
    state = torch.load(args.load_gen)
    gen.load_state_dict(state['gen_model'])
    gen_optimiser.load_state_dict(state['gen_optimiser'])

if args.load_dis is not None:
    state = torch.load(args.load_dis)
    dis.load_state_dict(state['dis_model'])
    dis_optimiser.load_state_dict(state['dis_optimiser'])

# save training params to file
save_dir = 'save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+'/'

while os.path.exists(save_dir): # if it exists, then another training process is using that folder. Wait 10 seconds and try again
    time.sleep(10)
    print("Waiting 10 seconds to find available slot...")
    current_run_time = datetime.now().strftime('%Y-%m-%d_%H-%M') # time for saving filename appropriately
    save_dir = 'save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+'/'

os.makedirs(save_dir, exist_ok=True)
with open(save_dir+'training_params.txt', "w") as tp:
    tp.write("Computation: GPU\n" if use_cuda else "Computation: CPU\n")
    if args.plot_name is not None:
        tp.write("Plot: " + args.plot_name +"\n")
    tp.write("Dataset: "+ dataset_name+'\n')
    tp.write("Vocab size: "+ str(vocab_size)+'\n')
    tp.write("Generator model: " + args.gen_model + '\n')
    tp.write("Discriminator model: " + args.dis_model + '\n')
    if args.load_checkpoint is not None:
        tp.write("Loading checkpoint: " + args.load_checkpoint + '\n')
    if args.load_gen is not None:
        tp.write("Loading generator: " + args.load_gen + '\n')
    if args.load_dis is not None:
        tp.write("Loading discriminator: " + args.load_dis + '\n')
    tp.write("Training procedure: " + args.train_proc + '\n')
    tp.write("Criterion: "+str(args.criterion)+'\n')
    tp.write("Generator Word embedding dimensions: " + str(gen_embedding_size) + '\n')
    tp.write("Discriminator Word embedding dimensions: " + str(dis_embedding_size) + '\n')
    tp.write("Generator LSTM hidden state dimensions: " +str(gen_lstm_size) +'\n')
    tp.write("Discriminator LSTM hidden state dimensions: " +str(dis_lstm_size) +'\n')
    tp.write("Batch size: "+ str(batch_size)+'\n')
    tp.write("Sequence length: "+ str(seq_len)+'\n')
    tp.write("Num epochs: "+ str(num_epochs)+'\n')
    tp.write("Num mini-batches: "+ str(num_batches)+'\n')
    tp.write(str(args.N_save_model) + "models saved during training at iterations:\n")
    tp.write(str([num_epochs*num_batches//args.N_save_model*i for i in range(args.N_save_model)]+[(num_epochs)*(num_batches)-1])+'\n')
    tp.write("Top k: "+ str(top_k)+'\n')
    tp.write("Initial temperature: "+str(initial_temperature)+'\n')
    tp.write("Temperature annealed to 1 after " + str(N_temperature) + " iterations\n")
    tp.write("Temperature control policy: " + temp_control_policy + '\n')
    tp.write("Generator learning rate:" + str(gen_lr) +'\n')
    tp.write("Discriminator learning rate:" + str(dis_lr) +'\n')
    tp.write('\n')
    tp.write("python train_gan.py")
    tp.write(" --gen-model " + args.gen_model + " --dis-model " + args.dis_model +  " --train-proc " + args.train_proc +  " --dataset " + args.dataset + " --criterion " + args.criterion)
    if args.load_checkpoint is not None:
        tp.write(" --load-checkpoint " + str(args.load_checkpoint))
    if args.load_gen is not None:
        tp.write(" --load-gen " + str(args.load_gen))
    if args.load_dis is not None:
        tp.write(" --load-dis " + str(args.load_dis))
    tp.write(" --num-epochs " + str(args.num_epochs) + " --gen-embed-dim " + str(args.gen_embed_dim) + " --dis-embed-dim " + str(args.dis_embed_dim) +  " --gen-lstm-size " + str(args.gen_lstm_size) +  " --dis-lstm-size " + str(args.dis_lstm_size) +  " --batch-size " + str(args.batch_size))
    tp.write(" --seed " + str(args.seed) +  " --seq-len " + str(args.seq_len))
    tp.write(" --top-k " + str(args.top_k) + " --init-temp " + str(args.init_temp) +  " --N-temp " + str(args.N_temp) +  " --temp-control " + args.temp_control +  " --gen-lr " + str(args.gen_lr))
    tp.write(" --dis-lr " + str(args.dis_lr) + " --N-save-model " + str(args.N_save_model))
    tp.write(" --num-sentences-portion " + str(args.num_sentences_portion) + " --training-text-portion " + str(args.training_text_portion))
    if args.plot_name is not None:
        tp.write(" --plot-name " + str(args.plot_name))
    if args.force_reload: tp.write(" --force-reload")
    if use_cuda: tp.write(" --cuda")
    tp.close()

print()
print('#' * 80)
print()
print("Started:", current_run_time)
print()
_rows_to_erase = 0 # for updating the training parameters

last_5_times = [10,10,10,10,10] # for printing out the average time per sample
iter_time = time.time()
ppl_calculated=0 # count of ppl calculations for time estimation

try:
    # Put model in training mode (enables things like dropout and batchnorm)
    gen.train()
    dis.train()
    
    # calc initial value for perplexity using just train.txt
    cmd = "python calc_ppl.py --dataset " + dataset_name + " --begin-calculation --suppress-print --calc-original-ppl" + " --save-dir " + 'save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time
    if use_cuda: cmd += " --cuda"
    os.system(cmd)
    lines = [line.rstrip('\n') for line in open('save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time +"/cur_ppl.txt")]
    perplexity = lines[0]

    # calculate number of lines in training text for more accurate time remaining estimate
    proc_sen_count = subprocess.Popen(['wc', '-l', dataset.train_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o, e = proc_sen_count.communicate()
    num_lines_training_text = int(o.decode('ascii').split(' ',maxsplit=1)[0]) # split the text at whitespace, stop after one occurrence

    # time_for_ppl_calc = (num_lines_training_text*args.num_sentences_portion)/200.0 # initial estimate
    time_for_ppl_calc = 60.0

    for epoch in range(num_epochs):
        batches = get_batches(dataset.training_dataset, batch_size, seq_len)
        # batches = DataLoader(input_text, batch_size=batch_size)
        for n_batch in range(num_batches):
            batch = next(batches)
            iterations = epoch*num_batches+n_batch
            
            # Transfer data to GPU
            if use_cuda: batch = batch.cuda()

            # Train Discriminator
            initial_word = torch.tensor(vocab_to_int["<s>"]).repeat(batch_size,1)
            dis_error, d_pred_real, d_pred_fake = train_dis(gen, dis, dis_optimiser, criterion, batch, use_cuda, top_k=top_k, i_temperature=iterations, N_temperature=N_temperature, initial_word=initial_word)

            # Train Generator
            gen_error = train_gen(gen, dis, gen_optimiser, criterion, use_cuda, top_k=top_k, i_temperature=iterations, N_temperature=N_temperature, initial_word=initial_word)

            # Clear output display
            # os.system('cls' if os.name == 'nt' else 'clear')

            if iterations % ((num_batches*num_epochs)//args.N_save_model) == 0 or iterations == (num_epochs)*(num_batches)-1: # save N instances of the model, and the final model
                # Save the current training model
                save_gan_training_params(gen, dis, gen_optimiser, dis_optimiser, epoch, 
                                            n_batch, gen_error, dis_error, dataset_name, 
                                            current_run_desc, current_run_time)

                gen.eval() # change generator into evaluation mode (disables dropout/batch-normalisation)
                example_batch_size = 1
                example_seq_len = 30

                random_SOS = torch.tensor([[vocab_to_int["<s>"]]]*example_batch_size) # random start of sentence
                if use_cuda: random_SOS = random_SOS.cuda()

                example_text = gen.sample_sentence(int_to_vocab, random_SOS, example_batch_size, example_seq_len)
                example_text = ' '.join(example_text)
                example_text = '\"' + example_text + '\"'

                if iterations != 0: # don't calculate perplexity on the first iteration
                    ppl_calculated += 1
                    cmd = "python calc_ppl.py" + " --dataset " + dataset_name + " --num-sentences-portion " + str(args.num_sentences_portion) + " --training-text-portion " + str(args.training_text_portion) + " --seq-len " + str(seq_len) + " --top-k " + str(top_k) + " --gen-model " + args.gen_model + " --embed-dim " + str(gen_embedding_size) + " --lstm-size " + str(gen_lstm_size) + " --delete-duplicates --begin-calculation --delete-generated-text --suppress-print" + " --load-checkpoint " + "save/"+current_run_desc+"/"+dataset_name+"/"+current_run_time+"/saved_models/saved_training_e-"+str(epoch)+"_i-"+str(n_batch)+".pth" + " --save-dir " + 'save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time

                    if use_cuda: cmd += " --cuda"

                    _ppl_time_start = time.time()
                    os.system(cmd)
                    time_for_ppl_calc = round(time.time() - _ppl_time_start, 2)

                    lines = [line.rstrip('\n') for line in open('save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time + "/cur_ppl.txt")]
                    perplexity = lines[0]
                    
                gen.train()
                
                # Log generated text
                save_example_generation(example_text, epoch, n_batch, dis_error.item(), gen_error.item(), 
                                        perplexity, dataset_name, current_run_desc, current_run_time)


            save_training_log(epoch, n_batch, dis_error.item(), gen_error.item(), perplexity, dataset_name, current_run_desc, current_run_time)

            # clear lines
            sys.stdout.write("\033[F\033[K"*(_rows_to_erase))
            
            rows, columns = os.popen('stty size', 'r').read().split() # get width (columns) of terminal
            # set the amount after, so that when the example text changes, it erases the previous text
            _rows_to_erase = 9+len(example_text)//int(columns)+1
            
            last_5_times[iterations%5] = time.time() - iter_time
            iter_time = time.time()
            avg_time = round(sum(last_5_times)/5.0,2)
            
            time_remaining = avg_time*(num_epochs*num_batches-iterations)
            time_remaining += time_for_ppl_calc*(args.N_save_model-ppl_calculated)

            if time_remaining > 86399: 
                time_remaining = "more than " + str(round(time_remaining//86399)) + " day(s)"
            else:
                time_remaining = time.strftime('%H:%M:%S', time.gmtime(time_remaining))

            # Print current training parameters
            print('Epoch: {}/{}'.format(epoch+1, num_epochs),
                    '\nBatch number: {}/{}'.format(n_batch+1, num_batches),
                    '\nD(G(z)): {}'.format(gen_error),
                    '\n   D(x): {}'.format(dis_error),
                    '\ngentemp: {}'.format(gen.temperature),
                    '\nPerplexity: {}'.format(perplexity),
                    '\n\n'+example_text,
                    '\n\n' + str(avg_time) + 's/iter, ' + str(time_for_ppl_calc) + 's/ppl_calc, ' + time_remaining + ' remaining')
            
except KeyboardInterrupt:
    print()
    print('-' * 80)
    print('Exiting from training early')

print()
print('#' * 80)
print()

# save graph
import csv
filepath = 'save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+'/log/log.txt'

with open(filepath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first_line = True
    gen_loss = []
    dis_loss = []
    perplex = []
    
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
            gen_loss.append(round(float(row[2]),5))
            dis_loss.append(round(float(row[3]),5))
            perplex.append(round(float(row[4]),5))
            
            stop_time = row[5]
    stop_time = stop_time.replace("-",":")
    csv_file.close()

#plot graph
plt.figure(figsize=(15,5))
plt.suptitle("'"+current_run_desc+"' on the '"+dataset_name+"' dataset. Training started at "+start_time[:-3]+" stopped at "+stop_time[:-3])
plt.subplot(1,2,1)
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.plot(range(len(dis_loss)), dis_loss, color='g', label="dis_loss")
plt.plot(range(len(gen_loss)), gen_loss, color='b', label="gen_loss")
plt.legend(loc='upper center')

plt.subplot(1,2,2)
plt.xlabel("Iteration")
plt.ylabel("Perplexity")

plt.plot(range(len(perplex)), perplex, color='r', label="perplexity")
plt.legend(loc='upper center')

if not args.save_graph:
    if input("Save output training graph? y/n\n") == 'y':
        if args.plot_name is not None:
            plot_filename = str(args.plot_name) + ".png"
        else:
            plot_filename = "output.png"
        plt.savefig('save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+"/" + plot_filename)
else:
    if args.plot_name is not None:
        plot_filename = str(args.plot_name) + ".png"
    else:
        plot_filename = "output.png"
    plt.savefig('save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+"/" + plot_filename)
    
if args.show_graph:
    plt.show()
