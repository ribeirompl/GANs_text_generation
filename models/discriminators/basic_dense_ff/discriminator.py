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
import torch.nn as nn
import torch.nn.functional as F

class Discriminator_model_dense_ff(nn.Module):
    """
    Discriminator model for a GAN
    Dense feed-forward network

    Parameters
    ----------
    use_cuda : bool
        True or False depending on whether using GPU or CPU
    vocab_size : int
        Size of the dataset's vocabulary
    batch_size : int 
        Size of the batches used during training
    seq_len : int
        The sequence length within each batch
    embedding_size : int
        Size of the vector representation for each word
    hidden_state_dim : int
        Size of the hidden state (h_n) and cell state (c_n) vectors of the LSTM
    
    Architecture
    ------------
    "partial sentence" -> Embedding Layer -> LSTM -> Linear -> Gumbel-Softmax -> out
    """

    def __init__(self, use_cuda, vocab_size, batch_size, seq_len, embedding_size, hidden_state_dim):
        super().__init__() # inherit the parent's methods and variables (from nn.Module)
        
        # Store instance variables
        self.use_cuda = use_cuda
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.hidden_state_dim = hidden_state_dim
        
        # initialise the network architecture's components
        self.vocab_to_one = nn.Sequential(
            # use leaky_relu for discriminator as well as negative slope=0.2 as recommended by https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/

            nn.Linear(vocab_size, embedding_size, bias=False), # bias=False to not learn additive bias
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(embedding_size, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.seq_len_to_one = nn.Sequential(
            nn.Linear(seq_len, 1, bias=True),
            nn.Sigmoid()
        )

        # init params?
        # torch.nn.init.xavier_uniform_(self.embeddings_layer.weight)
        # torch.nn.init.xavier_uniform_(self.hidden_layer1.weight)
        # self.hidden_layer1.weight.bias.data.fill_(0.01)
        # torch.nn.init.xavier_uniform_(self.output_linear.weight)
        # self.output_linear.weight.bias.data.fill_(0.01)
        
    def forward(self, inp):
        """
        Forward method of Discriminator model

        Parameters
        ----------
        inp: torch.tensor(float32) -- (batch_size * seq_len * vocab_size)
            input is the semi-one_hot representation from the Generator model output,
            or the one_hot representation of the real sentences.

        Returns
        -------
        out : torch.Tensor(float32) -- (batch_size)
            The probability of the input sequence being real or fake/generated, for each
            sequence in the batch.
        """

        # NB: make sure inp is requires_grad
        out = self.vocab_to_one(inp)
        # out size: (batch_size * seq_len * 1)
        out = out.permute(0,2,1)
        # out size: (batch_size * 1 * seq_len)
        out = self.seq_len_to_one(out)
        # out size: (batch_size * 1 * 1)
        out = out.squeeze()
        # out size: (batch_size)

        return out
    
    def zero_state(self):
        """
        Helper function to reset the states at the beginning of each epoch

        Returns
        -------
        _ : (torch.Tensor(float32), torch.Tensor(float32)) -- (1 * batch_size * hidden_state_dim)
        """
        
        return (torch.zeros(1, self.batch_size, self.hidden_state_dim),
                torch.zeros(1, self.batch_size, self.hidden_state_dim))