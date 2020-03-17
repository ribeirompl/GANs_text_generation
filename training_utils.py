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
import torch
from datetime import datetime


# MAYBE: add more params regarding the actual batch sizes etc. However may not work nicely if code is refactored/changed
# MAYBE: also save loss/criterion model?
def save_gan_training_params(gen_model=None, dis_model=None, gen_optimiser=None, dis_optimiser=None, epoch=None, iteration=None, gen_error=None, dis_error=None, dataset_name=None, current_run_desc=None, current_run_time=None, save_folder="save/"):
# create directory for saving model
    save_dir = save_folder+current_run_desc+'/'+dataset_name+'/'+current_run_time+'/saved_models/'
    os.makedirs(save_dir, exist_ok=True)
    dict_save = {}
    
    if gen_model is not None: dict_save['gen_model'] = gen_model.state_dict()
    if dis_model is not None: dict_save['dis_model'] = dis_model.state_dict()
    if gen_optimiser is not None: dict_save['gen_optimiser'] = gen_optimiser.state_dict()
    if dis_optimiser is not None: dict_save['dis_optimiser'] = dis_optimiser.state_dict()
    if epoch is not None: dict_save['epoch'] = epoch
    if iteration is not None: dict_save['iteration'] = iteration
    if dis_error is not None: dict_save['dis_error'] = dis_error
    if gen_error is not None: dict_save['gen_error'] = gen_error

    torch.save(dict_save,
                # os.path.join(save_dir,'saved_training.pth')) # overwrite prev one
                os.path.join(save_dir,f"saved_training_e-{epoch}.pth"))

def load_gan_training_params(filename, gen_model=None, dis_model=None, gen_optimiser=None, dis_optimiser=None):
    """
    net = Generator_model(n_vocab, seq_size, embedding_size, lstm_size)
    net.cuda()
    load_model(net, filename_of_model)
    """
    epoch, iteration, dis_error, gen_error = None, None, None, None
    
    state = torch.load(filename)
    if gen_model is not None and 'gen_model' in state: gen_model.load_state_dict(state['gen_model'])
    if dis_model is not None and 'dis_model' in state: dis_model.load_state_dict(state['dis_model'])
    if gen_optimiser is not None and 'gen_optimiser' in state: gen_optimiser.load_state_dict(state['gen_optimiser'])
    if dis_optimiser is not None and 'dis_optimiser' in state: dis_optimiser.load_state_dict(state['dis_optimiser'])
    if 'epoch' in state: epoch = state['epoch']
    if 'iteration' in state: iteration = state['iteration']
    if 'gen_error' in state: gen_error = state['gen_error']
    if 'dis_error' in state: dis_error = state['dis_error']
    
    # NB: remember to do gen_model.train/gen_model.eval and dis_model.train/dis_model.eval
    return epoch, iteration, dis_error, gen_error

def save_example_generation(example_text, epoch, iteration, dis_error=None, gen_error=None, perplexity=None, dataset_name=None, current_run_desc=None, current_run_time=None, save_folder="save/"):  
    """
    Save example generation
    """
    # create  directory for saving example generation
    save_dir = save_folder+current_run_desc+'/'+dataset_name+'/'+current_run_time+'/generated_data/'
    os.makedirs(save_dir, exist_ok=True)
    text_file = open(os.path.join(save_dir,"example_text.txt"), "a")
    text_file.write("epoch,"+str(epoch)+",iteration,"+str(iteration)+",gen_error,"+str(gen_error)+",dis_error,"+str(dis_error)+",perplexity,"+str(perplexity)+'\n')
    text_file.write(example_text+'\n')
    text_file.close()

def save_training_log(epoch, iteration, dis_error=None, gen_error=None, perplexity=None, dataset_name=None, current_run_desc=None, current_run_time=None, save_folder="save/"):
    # create  directory for saving example generation
    save_dir = save_folder+current_run_desc+'/'+dataset_name+'/'+current_run_time+'/log/'
    os.makedirs(save_dir, exist_ok=True)
    if os.path.isfile(os.path.join(save_dir,'log.txt')):
        text_file = open(os.path.join(save_dir,"log.txt"), "a")
    else:
        text_file = open(os.path.join(save_dir,"log.txt"), "w")

        text_file.write("epoch,iteration,gen_error,dis_error, perplexity, cur_time\n")
    
    cur_time = datetime.now().strftime('%H-%M-%S')
    text_file.write(str(epoch)+","+str(iteration)+","+str(gen_error)+","+str(dis_error)+","+str(perplexity)+","+cur_time+"\n")
    text_file.close()

def is_notebook():
    """
    Returns true if within a jupyter-notebook environment
    source: https://stackoverflow.com/a/39662359
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# Below is unused
def save_model(net_model, net_name, dataset_name, current_run_desc, current_run_time, iteration, save_folder="save/"):
    # create directory for saving model
    save_dir = save_folder+current_run_desc+'/'+dataset_name+'/'+current_run_time+'/basic_models/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(net_model.state_dict(),os.path.join(save_dir,'{}-model-{}.pth'.format(net_name, iteration)))

def load_model(net, filename):
    """
    net = Generator_model(n_vocab, seq_size, embedding_size, lstm_size)
    net.cuda()
    load_model(net, filename_of_model)
    """
    net.load_state_dict(torch.load(filename))
