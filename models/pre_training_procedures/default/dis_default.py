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

def pretrain_dis(dis, dis_optimiser, criterion, batch_real, batch_fake, use_cuda):
    """
    Function for training the Discriminator against the Generator

    Parameters
    ----------
    dis : Discriminator_model
        The Discriminator model to be trained
    dis_optimiser : torch.optim
        The Discriminator's optimiser
    criterion : torch.nn.modules.loss
        The specific loss module used
    batch_real : torch.Tensor -- (batch_size * seq_len)
        A batch of real text sequences
        e.g. [["I","AM","GOING","HOME"],["WHAT","IS","THE","TIME"]] (but the index/embedded representation of the words)
    batch_fake : torch.Tensor -- (batch_size * seq_len)
        A batch of fake text sequences
    use_cuda : bool
        True or False depending on whether using GPU or CPU

    Returns
    -------
    total_error : torch.Tensor(float32)
        The error of the Discriminator from trying to fool the Generator
    prediction_real : torch.Tensor(float32) -- (batch_size)
        The certainty with which the Discriminator predicts real data as real (1)
    prediction_fake : torch.Tensor(float32) -- (batch_size)
        The certainty with which the Discriminator predicts fake data as fake (0)
    """

    dis_optimiser.zero_grad() # Reset gradients, as PyTorch accumulates them

    ####################
    # Train on real data
    ####################
    real_data_one_hot = F.one_hot(batch_real, dis.vocab_size) # Encode real data to one_hot
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
    fake_data_one_hot = F.one_hot(batch_fake, dis.vocab_size) # Encode real data to one_hot
    fake_data_one_hot = fake_data_one_hot.float() # Convert to float, to perform some float operations.
    # The network's weights are floats, thus this also needs to be float

    prediction_fake = dis(fake_data_one_hot) # pass the fake data to the Discriminator (forward method), and acquire the predictions
    
    # The Discriminator wants to predict the Generator's fake data as fake (0's)
    fake_data_target = torch.zeros_like(prediction_fake)
    
    # Calculate error and gradients
    error_fake = criterion(prediction_fake, fake_data_target)
    error_fake.backward()
    
    # Update weights with gradients
    dis_optimiser.step()
    
    total_error = error_real + error_fake
    return total_error, prediction_real, prediction_fake
