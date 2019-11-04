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

def train_generator(gen, dis, gen_optimiser, criterion, use_cuda, top_k=1, i_temperature=2500, N_temperature=5000, initial_word=None):
    """
    Function for training the Generator against the Discriminator

    Parameters
    ----------
    gen : Generator_model
        The Generator model to be trained
    dis : Discriminator_model
        The Discriminator model to use for predicting real/fake data
    gen_optimiser : torch.optim
        The Generator's optimiser
    criterion : torch.nn.modules.loss
        The specific loss module used
    use_cuda : bool
        True or False depending on whether using GPU or CPU
    top_k : int (default=1)
        When predicting the next word, the k'th highest probable words are selected and one is chosen from them
    i_temperature : int (default=2500)
        The current iteration of the training process, used for calculating the new temperature parameter
    N_temperature : int (default=5000)
        The total number of iterations before the temperature parameter should be 1
    initial_word : torch.Tensor([String]) -- (batch_size * 1) (default=random word in vocabulary)
        The initial word to use when generating a text sequence. Typically the BOS token.

    Returns
    -------
    error : torch.Tensor(float32)
        The error of the Generator from trying to fool the Discriminator
    """

    # Check if initial_word is given, else use a random word in the vocabulary
    if type(initial_word) == type(None): # comparing type, since tensors can't be compared to Nonetype
        random_word = torch.randint(gen.vocab_size,(gen.batch_size,1)) # Generate [batch_size*1] size tensor of random starting word indexes
    else:
        random_word = initial_word
    if use_cuda: random_word = random_word.cuda() # Send to GPU

    gen_optimiser.zero_grad() # Reset gradients, because PyTorch accumulates them

    # Generate fake data (semi-one_hot representation)
    _, all_probs = gen.predict(random_word, i_temperature=i_temperature, N_temperature=N_temperature,
                                batch_size=gen.batch_size, seq_len=gen.seq_len, top_k=top_k)

    prediction_fake = dis(all_probs) # pass the fake data to the Discriminator (forward method), and acquire the predictions
    
    # The Generator wants the Discriminator to predict 1's (real data)
    fake_data_target = torch.ones_like(prediction_fake)
    if use_cuda: fake_data_target = fake_data_target.cuda()

    # Calculate error and gradients
    error = criterion(prediction_fake, fake_data_target)
    error.backward()

    # Update the network's weights with the calculated gradients
    gen_optimiser.step()

    return error

def train_discriminator(gen, dis, dis_optimiser, criterion, real_data_batch, use_cuda, top_k=1, i_temperature=2500, N_temperature=5000, initial_word=None):
    """
    Function for training the Discriminator against the Generator

    Parameters
    ----------
    gen : Generator_model
        The Generator model to use for generating fake data
    dis : Discriminator_model
        The Discriminator model to be trained
    dis_optimiser : torch.optim
        The Discriminator's optimiser
    criterion : torch.nn.modules.loss
        The specific loss module used
    real_data_batch : torch.Tensor -- (batch_size * seq_len)
        A batch of real text sequences
        e.g. [["I","AM","GOING","HOME"],["WHAT","IS","THE","TIME"]] (but the index/embedded representation of the words)
    use_cuda : bool
        True or False depending on whether using GPU or CPU
    top_k : int (default=1)
        When predicting the next word, the k'th highest probable words are selected and one is chosen from them
    i_temperature : int (default=2500)
        The current iteration of the training process, used for calculating the new temperature parameter
    N_temperature : int (default=5000)
        The total number of iterations before the temperature parameter should be 1
    initial_word : torch.Tensor([String]) -- (batch_size * 1) (default=random word in vocabulary)
        The initial word to use when generating a text sequence. Typically the BOS token.

    Returns
    -------
    total_error : torch.Tensor(float32)
        The error of the Discriminator from trying to fool the Generator
    prediction_real : torch.Tensor(float32) -- (batch_size)
        The certainty with which the Discriminator predicts real data as real (1)
    prediction_fake : torch.Tensor(float32) -- (batch_size)
        The certainty with which the Discriminator predicts fake data as fake (0)
    """

    # Check if initial_word is given, else use a random word in the vocabulary
    if type(initial_word) == type(None): # comparing type, since tensors can't be compared to Nonetype
        random_word = torch.randint(gen.vocab_size,(gen.batch_size,1)) # Generate [batch_size*1] size tensor of random starting word indexes
    else:
        random_word = initial_word 
    if use_cuda: random_word = random_word.cuda() # Send to GPU

    dis_optimiser.zero_grad() # Reset gradients, as PyTorch accumulates them

    ####################
    # Train on real data
    ####################
    real_data_one_hot = F.one_hot(real_data_batch, gen.vocab_size) # Encode real data to one_hot
    real_data_one_hot = real_data_one_hot.float() # Convert to float, to perform some float operations.
    # The network's weights are floats, thus this also needs to be float
    
    prediction_real = dis(real_data_one_hot) # pass the real data to the Discriminator (forward method), and acquire the predictions
    
    # The Discriminator wants to predict the real data as real (1's)
    real_data_target = torch.ones_like(prediction_real).float() # Convert to float, to perform float operations with the weights.

    # calculate error and gradients
    error_real = criterion(prediction_real, real_data_target)
    error_real.backward()

    ####################
    # Train on fake data
    ####################
    _, all_probs = gen.predict(random_word, i_temperature=i_temperature, N_temperature=N_temperature,
                            batch_size=gen.batch_size, seq_len=gen.seq_len, top_k=top_k) # generate fake data
    all_probs.detach() # detach to avoid calculating the gradients through the Generator in this step

    prediction_fake = dis(all_probs) # pass the fake data to the Discriminator (forward method), and acquire the predictions
    
    # The Discriminator wants to predict the Generator's fake data as fake (0's)
    fake_data_target = torch.zeros_like(prediction_fake)
    
    # Calculate error and gradients
    error_fake = criterion(prediction_fake, fake_data_target)
    error_fake.backward()
    
    # Update weights with gradients
    dis_optimiser.step()
    
    total_error = error_real + error_fake
    return total_error, prediction_real, prediction_fake