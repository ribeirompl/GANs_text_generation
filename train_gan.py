"""Training a GAN

This file is part of a final year undergraduate project for
generating discrete text sequences using generative adversarial
networks (GANs)
 
GNU GPL-3.0-or-later
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import display
from torch.utils.data import DataLoader, Dataset

from dataset_utils import get_batches, save_tokenized_dataset
from training_utils import (load_gan_training_params, save_example_generation,
                            save_gan_training_params, save_training_log)


def check_argv():
    """Check the command line arguments."""

    GEN_MODEL_CHOICES = ['default']
    DIS_MODEL_CHOICES = ['basic_dense_ff','rnn','conv_net','relgan']
    TRAIN_PROC_CHOICES = ['default','ACGD','modified_minimax']
    DATASET_CHOICES = ['EZ','J2015','SJ2015','SJ2015S','OF']
    CRITERION_CHOICES = ['CE','BCE','BCEwL']

    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=True)

    parser.add_argument('--gen-model', choices=GEN_MODEL_CHOICES, required=True, help='generator model')
    parser.add_argument('--dis-model', choices=DIS_MODEL_CHOICES, required=True, help='discriminator model')
    parser.add_argument('--train-proc', choices=TRAIN_PROC_CHOICES, required=True, help='training procedure')
    parser.add_argument('--dataset', choices=DATASET_CHOICES, required=True)
    parser.add_argument('--criterion', choices=CRITERION_CHOICES, required=True, help='Criterion for calculating losses')
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
    parser.add_argument('--begin-training', action='store_true', help='Begin training immediately')
    parser.add_argument('--save-graph', action='store_true', help='Save graph immediately')
    parser.add_argument('--show-graph', action='store_true', help='Show graph after generating it')
    parser.add_argument('--load-gen', default=None, help='File containing pre-trained generator state dict')
    parser.add_argument('--load-dis', default=None, help='File containing pre-trained discriminator state dict')
    parser.add_argument('--load-checkpoint', default=None, help='File containing pre-trained generator and discriminator as well as their respective optimisers state dict')
    parser.add_argument('--num-sentences-portion', type=float, default=0.5, help='Number representing the percentage of training text sentence count, to generate. (1.0 means generate the same number of sentences that training text contains)')
    parser.add_argument('--training-text-portion', type=float, default=1.0, help='Percentage of the original text to keep (value between 0 and 1)')
    parser.add_argument('--plot-name', default=None, help='Name for the output image training plot')

    return parser.parse_args()


def get_dataset(dataset_name, force_reload=False):
    """
    force_reload = True specifies that the dataset should be re-processed.
        otherwise the pre-processed dataset will be loaded in.
    """
    dataset=None
    if dataset_name == 'EZ':
        dataset = save_tokenized_dataset('./dataset/preread/ez_cs/',
                                            './dataset/ez_cs_dataset/train.txt',
                                            './dataset/ez_cs_dataset/dev.txt',
                                            './dataset/ez_cs_dataset/test.txt',
                                            lambda x: x,
                                            "<s>", "</s>", force_reload=force_reload, skip_count_check=True)
    elif dataset_name == 'J2015':
        dataset = save_tokenized_dataset('./dataset/preread/johnnic_2015_cleaned_shuffled/',
                                            './dataset/johnnic_2015_cleaned_shuffled/train.txt',
                                            './dataset/johnnic_2015_cleaned_shuffled/dev.txt',
                                            './dataset/johnnic_2015_cleaned_shuffled/test.txt',
                                            lambda x: x,
                                            None, None, force_reload=force_reload, skip_count_check=True)
    elif dataset_name == 'SJ2015':
        dataset = save_tokenized_dataset('./dataset/preread/smaller_johnnic/',
                                            './dataset/smaller_johnnic/train.txt',
                                            './dataset/smaller_johnnic/dev.txt',
                                            './dataset/smaller_johnnic/test.txt',
                                            lambda x: x,
                                            None, None, force_reload=force_reload, skip_count_check=True)
    elif dataset_name == 'SJ2015S':
        dataset = save_tokenized_dataset('./dataset/preread/smaller_johnnic_shuffled/',
                                            './dataset/smaller_johnnic_shuffled/train.txt',
                                            './dataset/smaller_johnnic_shuffled/dev.txt',
                                            './dataset/smaller_johnnic_shuffled/test.txt',
                                            lambda x: x,
                                            None, None, force_reload=force_reload, skip_count_check=True)
    elif dataset_name == 'OF':
        dataset = save_tokenized_dataset('./dataset/preread/overfit/',
                                            './dataset/overfit/train.txt',
                                            './dataset/overfit/dev.txt',
                                            './dataset/overfit/test.txt',
                                            lambda x: x,
                                            None, None, force_reload=force_reload, skip_count_check=True)
    else:
        raise Exception("Invalid dataset. Specify --dataset and the name")

    return dataset


def import_models(gen_model_name, dis_model_name, train_proc_name):
    if gen_model_name == 'default':
        from models.generators.default.generator import Generator_model as Generator
    else:
        raise Exception("Invalid generator model. Specify --gen-model and the name")

    if dis_model_name == 'basic_dense_ff':
        from models.discriminators.basic_dense_ff.discriminator import Discriminator_model_dense_ff as Discriminator
    elif dis_model_name == 'rnn':
        from models.discriminators.rnn.discriminator import Discriminator_model_rnn as Discriminator
    elif dis_model_name == 'conv_net':
        raise Exception("Not yet implemented")
    elif dis_model_name == 'relgan':
        from models.discriminators.relgan.discriminator import Discriminator_model_relgan as Discriminator
    else:
        raise Exception("Invalid discriminator model. Specify --dis-model and the name")

    if train_proc_name == 'default':
        from models.training_procedures.default.training_gen_and_dis import train_generator as train_gen, train_discriminator as train_dis
    elif train_proc_name == 'ACGD':
        from models.training_procedures.acgd.training_gen_and_dis import train_generator as train_gen, train_discriminator as train_dis
    elif train_proc_name == 'modified_minimax':
        from models.training_procedures.modified_minimax_loss.training_gen_and_dis import train_generator as train_gen, train_discriminator as train_dis
    else:
        raise Exception("Invalid training procedure. Specify --train-proc and the name")
    
    return Generator, Discriminator, train_gen, train_dis

def get_criterion(criterion_name):
    if criterion_name == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == 'BCE':
        criterion = nn.BCELoss()
    elif criterion_name == 'BCEwL':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise Exception("Invalid criterion. Specify --criterion and the name")
    return criterion


def plot_graph(current_run_desc, dataset_name, current_run_time, save_graph=True, plot_name="training_plot.png", show_graph=False):
    filepath = f"save/{current_run_desc}/{dataset_name}/{current_run_time}/log/log.txt"
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
    plt.suptitle(f"'{current_run_desc}' on the '{dataset_name}' dataset. Training started at {start_time[:-3]} stopped at {stop_time[:-3]}")
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

    if not save_graph:
        if input("Save output training graph? y/n\n") == 'y':
            if plot_name is not None:
                plot_filename = str(plot_name) + ".png"
            else:
                plot_filename = "output.png"
            plt.savefig('save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+"/" + plot_filename)
    else:
        if plot_name is not None:
            plot_filename = str(plot_name) + ".png"
        else:
            plot_filename = "output.png"
        plt.savefig('save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time+"/" + plot_filename)
        
    if show_graph:
        plt.show()


def main():
    args = check_argv()

    use_cuda = args.cuda
    if torch.cuda.is_available():
        if not use_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        if use_cuda:
            raise Exception("CUDA device not found")
    device = torch.device("cuda" if use_cuda else "cpu")

    # Fix seed
    np.random.seed(args.seed);
    torch.manual_seed(args.seed);
    # For multi-GPU setup
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)

    dataset_name = args.dataset
    dataset = get_dataset(dataset_name, args.force_reload)
    vocab_size = len(dataset.dictionary.idx2word)

    # Import the relevant generator, discriminator and training_procedure
    gen_model_name = args.gen_model
    dis_model_name = args.dis_model
    train_proc_name = args.train_proc
    Generator, Discriminator, train_gen, train_dis = import_models(gen_model_name, dis_model_name, train_proc_name)
    criterion = get_criterion(args.criterion)

    current_run_desc = f"{gen_model_name}-{dis_model_name}-{train_proc_name}"
    # Retrieve the model args
    gen_embedding_size = args.gen_embed_dim
    dis_embedding_size = args.dis_embed_dim
    gen_lstm_size = args.gen_lstm_size
    dis_lstm_size = args.dis_lstm_size
    # Retrieve the training args
    num_epochs =  args.num_epochs
    batch_size = args.batch_size
    gen_lr = args.gen_lr
    dis_lr = args.dis_lr
    seq_len = args.seq_len
    top_k = args.top_k
    num_batches = len(dataset.training_dataset) // (batch_size*seq_len)
    # Retrieve the Gumbel-Softmax args
    initial_temperature = args.init_temp
    temp_control_policy = args.temp_control
    N_temperature = int(num_batches*num_epochs*args.N_temp)

    # Print the settings to the terminal
    print("Computation: GPU" if use_cuda else "Computation: CPU")
    print("--- Dataset settings ---")
    print("Dataset:", dataset_name)
    print("Vocab size:",vocab_size)
    print("--- Model settings ---")
    print("Generator model:",gen_model_name)
    print("Discriminator model:",dis_model_name)
    if args.load_checkpoint is not None:
        print("\tLoading checkpoint:",args.load_checkpoint)
    if args.load_gen is not None:
        print("\tLoading generator:",args.load_gen)
    if args.load_dis is not None:
        print("\tLoading discriminator:",args.load_dis)
    print("Generator Word embedding dimensions:",gen_embedding_size)
    print("Discriminator Word embedding dimensions:",dis_embedding_size)
    print("Generator LSTM hidden state dimensions:",gen_lstm_size)
    print("Discriminator LSTM hidden state dimensions:",dis_lstm_size)
    print("--- Training settings ---")
    print("Generator learning rate:", gen_lr)
    print("Discriminator learning rate:", dis_lr)
    print("Training procedure:",train_proc_name)
    print("Criterion:",args.criterion)
    print("Batch size:",batch_size)
    print("Sequence length:", seq_len)
    print("Num epochs:",num_epochs)
    print("Num mini-batches:", num_batches)
    print("Top k:", top_k)
    if args.plot_name is not None:
        print("Training plot name:", args.plot_name)
    print("--- Gumbel-Softmax settings ---")
    print("Initial temperature:", initial_temperature)
    print("Temperature annealed to 1 after", N_temperature, "iterations")
    print("Temperature control policy:", temp_control_policy)
    print("-" * 80)
    print()

    if not args.begin_training:
        if input("Start training? y/n\n") != "y":
            print("-" * 80)
            print("Exiting...")
            sys.exit()
    
    ################
    # Start training
    ################
    int_to_vocab, vocab_to_int = dataset.dictionary.idx2word, dataset.dictionary.word2idx
    current_run_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # time for saving filename appropriately

    gen = Generator(use_cuda, vocab_size, batch_size, seq_len, gen_embedding_size, gen_lstm_size, initial_temperature, control_policy=temp_control_policy)
    dis = Discriminator(use_cuda, vocab_size, batch_size, seq_len, dis_embedding_size, dis_lstm_size)

    # Send to computation device
    gen.to(device)
    dis.to(device)

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
        tp.write("--- Dataset settings ---")
        tp.write(f"Dataset: {dataset_name}\n")
        tp.write(f"Vocab size: {vocab_size}\n")
        tp.write("--- Model settings ---")
        tp.write(f"Generator model: {gen_model_name}\n")
        tp.write(f"Discriminator model: {dis_model_name}\n")
        if args.load_checkpoint is not None:
            tp.write(f"Loading checkpoint: {args.load_checkpoint}\n")
        if args.load_gen is not None:
            tp.write(f"Loading generator: {args.load_gen}\n")
        if args.load_dis is not None:
            tp.write(f"Loading discriminator: {args.load_dis}\n")
        tp.write(f"Generator Word embedding dimensions: {gen_embedding_size}\n")
        tp.write(f"Discriminator Word embedding dimensions: {dis_embedding_size}\n")
        tp.write(f"Generator LSTM hidden state dimensions: {gen_lstm_size}\n")
        tp.write(f"Discriminator LSTM hidden state dimensions: {dis_lstm_size}\n")
        tp.write("--- Training settings ---")
        tp.write(f"Generator learning rate: {gen_lr}\n")
        tp.write(f"Discriminator learning rate: {dis_lr}\n")
        tp.write(f"Training procedure: {train_proc_name}\n")
        tp.write(f"Criterion: {args.criterion}\n")
        tp.write(f"Batch size: {batch_size}\n")
        tp.write(f"Sequence length: {seq_len}\n")
        tp.write(f"Num epochs: {num_epochs}\n")
        tp.write(f"Num mini-batches: {num_batches}\n")
        tp.write("Top k: "+ str(top_k)+'\n')
        if args.plot_name is not None:
            tp.write(f"Training Plot: {args.plot_name}\n")
        tp.write("--- Gumbel-Softmax settings ---")
        tp.write(f"Initial temperature: {initial_temperature}\n")
        tp.write(f"Temperature annealed to 1 after {N_temperature} iterations\n")
        tp.write(f"Temperature control policy: {temp_control_policy}\n")
        tp.write("\n")
        tp.write("python train_gan.py")
        tp.write(f" --gen-model {gen_model_name} --dis-model {dis_model_name} --train-proc {train_proc_name} --dataset {args.dataset} --criterion {args.criterion}")
        if args.load_checkpoint is not None:
            tp.write(f" --load-checkpoint {args.load_checkpoint}")
        if args.load_gen is not None:
            tp.write(f" --load-gen {args.load_gen}")
        if args.load_dis is not None:
            tp.write(f" --load-dis {args.load_dis}")
        tp.write(f" --num-epochs {args.num_epochs} --gen-embed-dim {args.gen_embed_dim} --dis-embed-dim {args.dis_embed_dim} --gen-lstm-size {args.gen_lstm_size} --dis-lstm-size {args.dis_lstm_size} --batch-size {args.batch_size}")
        tp.write(f" --seed {args.seed} --seq-len {args.seq_len}")
        tp.write(f" --top-k {args.top_k} --init-temp {args.init_temp} --N-temp {args.N_temp} --temp-control {args.temp_control} --gen-lr {args.gen_lr} --dis-lr {args.dis_lr}")
        tp.write(f" --num-sentences-portion {args.num_sentences_portion} --training-text-portion {args.training_text_portion}")
        if args.plot_name is not None:
            tp.write(f" --plot-name {args.plot_name}")
        if args.force_reload: tp.write(" --force-reload")
        if use_cuda: tp.write(" --cuda")
        tp.close()

    print()
    print('#' * 80)
    print()
    print("Started:", current_run_time)
    print()

    last_5_times = [10,10,10,10,10] # for printing out the average time per sample
    iter_time = time.time()

    try:
        # Put model in training mode (enables things like dropout and batchnorm)
        gen.train()
        dis.train()
        
        # calc initial value for perplexity using just train.txt
        cmd = f"python calc_ppl.py --dataset {dataset_name} --begin-calculation --suppress-print --calc-original-ppl --save-dir save/{current_run_desc}/{dataset_name}/{current_run_time}"
        if use_cuda: cmd += " --cuda"
        os.system(cmd)
        lines = [line.rstrip('\n') for line in open(f"save/{current_run_desc}/{dataset_name}/{current_run_time}/cur_ppl.txt")]
        perplexity = lines[0]

        # calculate number of lines in training text for more accurate time remaining estimate
        proc_sen_count = subprocess.Popen(['wc', '-l', dataset.train_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        o, e = proc_sen_count.communicate()
        num_lines_training_text = int(o.decode('ascii').split(' ',maxsplit=1)[0]) # split the text at whitespace, stop after one occurrence

        time_for_ppl_calc = 60.0 # initial estimate
        example_text = ""
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

                save_training_log(epoch, n_batch, dis_error.item(), gen_error.item(), perplexity, dataset_name, current_run_desc, current_run_time)

                last_5_times[iterations%5] = time.time() - iter_time
                iter_time = time.time()
                avg_time = round(sum(last_5_times)/5.0,2)
                
                time_remaining = avg_time*(num_epochs*num_batches-iterations)
                time_remaining += time_for_ppl_calc*(num_epochs-epoch)

                if time_remaining > 86399: 
                    time_remaining = "more than " + str(round(time_remaining//86399)) + " day(s)"
                else:
                    time_remaining = time.strftime('%H:%M:%S', time.gmtime(time_remaining))

                # Print current training parameters
                print(f"Epoch: {epoch+1}/{num_epochs}",
                        f"\nBatch number: {n_batch+1}/{num_batches}",
                        f"\nD(G(z)): {gen_error}",
                        f"\n   D(x): {dis_error}",
                        f"\ngentemp: {gen.temperature}",
                        f"\nPerplexity: {perplexity}",
                        f"\n\n{example_text}",
                        f"\n\n{avg_time}s/iter, {time_for_ppl_calc}s/ppl_calc, {time_remaining} remaining")

            # Save the current training model after every epoch
            save_gan_training_params(gen, dis, gen_optimiser, dis_optimiser, epoch, 
                                        n_batch, gen_error, dis_error, dataset_name, 
                                        current_run_desc, current_run_time)

            gen.eval() # change generator into evaluation mode (disables dropout/batch-normalisation)
            example_batch_size = 1
            example_seq_len = 30

            random_SOS = torch.tensor([[vocab_to_int["<s>"]]]*example_batch_size) # random start of sentence
            if use_cuda: random_SOS = random_SOS.cuda()

            example_text = gen.sample_sentence(int_to_vocab, random_SOS, example_batch_size, example_seq_len)
            example_text = " ".join(example_text)
            example_text = f"\"{example_text}\""
            gen.train()

            # Calculate perplexity
            cmd = f"python calc_ppl.py --dataset {dataset_name} --num-sentences-portion {args.num_sentences_portion} --training-text-portion {args.training_text_portion} --seq-len {seq_len} --top-k {top_k} --gen-model {gen_model_name}  --embed-dim {gen_embedding_size} --lstm-size {gen_lstm_size} --delete-duplicates --begin-calculation --delete-generated-text --suppress-print --load-checkpoint save/{current_run_desc}/{dataset_name}/{current_run_time}/saved_models/saved_training_e-{epoch}.pth --save-dir save/{current_run_desc}/{dataset_name}/{current_run_time}"
            if use_cuda: cmd += " --cuda"
            _ppl_time_start = time.time()
            os.system(cmd)
            time_for_ppl_calc = round(time.time() - _ppl_time_start, 2)
            lines = [line.rstrip('\n') for line in open('save/'+current_run_desc+'/'+dataset_name+'/'+current_run_time + "/cur_ppl.txt")]
            perplexity = lines[0]
            
            # Log generated text
            save_example_generation(example_text, epoch, n_batch, dis_error.item(), gen_error.item(), 
                                    perplexity, dataset_name, current_run_desc, current_run_time)
                
    except KeyboardInterrupt:
        print()
        print('-' * 80)
        print('Exiting from training early')

    print()
    print('#' * 80)
    print()

    # save graph
    plot_graph(current_run_desc, dataset_name, current_run_time, args.save_graph, args.plot_name, args.show_graph)

if __name__ == "__main__":
    main()
