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

import torch
import torch.nn.functional as F

def pretrain_gen(gen, gen_optimiser, criterion, batch_in, batch_out, hidden, use_cuda, top_k=1, i_temperature=2500, N_temperature=5000):
    """
    Function for pre-training the Generator

    Parameters
    ----------
    gen : Generator_model
        The Generator model to be trained
    gen_optimiser : torch.optim
        The Generator's optimiser
    criterion : torch.nn.modules.loss
        The specific loss module used
    batch_in : torch.Tensor() -- (batch_size * seq_len)
        Batch of input text
    batch_out : torch.Tensor() -- (batch_size * seq_len)
        Batch of target text
    hidden : torch.Tensor() -- ((1 * batch_size * hidden_state_dim),(1 * batch_size * hidden_state_dim))
        Hidden state initial value
    use_cuda : bool
        True or False depending on whether using GPU or CPU
    top_k : int (default=1)
        When predicting the next word, the k'th highest probable words are selected and one is chosen from them
    i_temperature : int (default=2500)
        The current iteration of the training process, used for calculating the new temperature parameter
    N_temperature : int (default=5000)
        The total number of iterations before the temperature parameter should be 1

    Returns
    -------
    error : float32
        The error of the Generator
    """

    gen_optimiser.zero_grad() # Reset gradients, because PyTorch accumulates them

    probs, prev_state = gen(batch_in, hidden)
    
    error = criterion(probs.transpose(1, 2), batch_out)

    prev_state = (prev_state[0].detach, prev_state[1].detach)

    error.backward()

    # Update the network's weights with the calculated gradients
    gen_optimiser.step()

    return error.item()

def pretrain_gen_sample_sentence(gen, initial_words, hidden, vocab_to_int, int_to_vocab, use_cuda, seq_len, top_k=1):
    """
    Function for generating sample text from the Generator

    Parameters
    ----------
    gen : Generator_model
        The Generator model to be trained
    initial_words : torch.Tensor() -- (seq_len)
        Initial starter words
    hidden : torch.Tensor() -- ((1 * batch_size * hidden_state_dim),(1 * batch_size * hidden_state_dim))
        Hidden state initial value
    vocab_to_int : dict
        Dictionary containing vocabulary:index key value pairs
    int_to_vocab : dict
        Dictionary containing index:vocabulary key value pairs
    use_cuda : bool
        True or False depending on whether using GPU or CPU
    seq_len : int
        Length of output sentence
    top_k : int (default=1)
        When predicting the next word, the k'th highest probable words are selected and one is chosen from them

    Returns
    -------
    sentence : [String]
        The generated sentence
    """
    gen.eval()
    sentence = initial_words
    for word in initial_words:
        index = torch.tensor([[vocab_to_int[word]]])
        if use_cuda: index = index.cuda()
        probs, hidden = gen(index, hidden)
    
    _, next_possible_tokens = torch.topk(probs[-1], k=top_k)
    # next_possible_tokens size: (1 * top_k)
    
    choice_idx = torch.randint(next_possible_tokens.size(0), (1,1))
    if use_cuda: choice_idx = choice_idx.cuda()
    next_token = next_possible_tokens[choice_idx] # get next word token from the chosen index
    sentence.append(int_to_vocab[next_token.item()])

    num_needed_words = seq_len - len(sentence)    
    for i in range(num_needed_words):
        # TODO rather unsqueeze next_token
        index = torch.tensor([[next_token.item()]])
        if use_cuda: index = index.cuda()
        probs, hidden = gen(index, hidden)
    
        _, next_possible_tokens = torch.topk(probs[-1], k=top_k)
        # next_possible_tokens size: (1 * top_k)

        choice_idx = torch.randint(next_possible_tokens.size(0), (1,1))
        if use_cuda: choice_idx = choice_idx.cuda()
        next_token = next_possible_tokens[choice_idx] # get next word token from the chosen index

        sentence.append(int_to_vocab[next_token.item()])

    gen.train()

    return sentence
