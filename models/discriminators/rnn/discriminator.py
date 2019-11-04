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

class Discriminator_model_rnn(nn.Module):
    def __init__(self, use_cuda, vocab_size, batch_size, seq_len, embedding_size, hidden_state_dim):
        super().__init__()
        
        self.use_cuda = use_cuda
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.hidden_state_dim = hidden_state_dim
        
        self.embeddings_layer = nn.Linear(vocab_size, embedding_size, bias=False) # bias=False to not learn additive bias
        
        self.lstm_layer = nn.LSTM(embedding_size, hidden_state_dim, batch_first=True)

        self.lstm2one = nn.Linear(hidden_state_dim, 1, bias=True)
        self.seq_len2one = nn.Linear(seq_len, 1, bias=True)
        self.sigmoid_out = nn.Sigmoid()
        # torch.nn.init.xavier_uniform_(self.embeddings_layer.weight)
        # torch.nn.init.xavier_uniform_(self.hidden_layer1.weight)
        # self.hidden_layer1.weight.bias.data.fill_(0.01)
        # torch.nn.init.xavier_uniform_(self.output_linear.weight)
        # self.output_linear.weight.bias.data.fill_(0.01)

        # MAYBE: try? conv highway network thing from https://github.com/weilinie/RelGAN
        
    def forward(self, inp):
        """
        inp: batch_size * seq_len * vocab_size
        """
        # print(inp)
        # use leaky_relu for discriminator as well as negative slope=0.2 as recommended by https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
        
        # NB: make sure inp is requires_grad
        # print("inp:",inp)
        embeds = F.leaky_relu(self.embeddings_layer(inp), negative_slope=0.2) # batch_size * seq_len * embedding_size
        # print("embeds:",embeds)

        output, states = self.lstm_layer(embeds, self.zero_state(batch_size = embeds.size(0)))   # output: batch_size * seq_len * hidden_state_dim
                                                                # states: (1 * batch_size * hidden_state_dim, 1 * batch_size * hidden_state_dim)

        
        # not sure about below
        #output = output.contiguous().view(-1, self.hidden_state_dim) # reshape the output value to make it the proper dimensions for the fully connected layer

        out = F.leaky_relu(self.lstm2one(output), negative_slope=0.2) # batch_size * seq_len * 1
        # print("out:",out)
        out = out.permute(0,2,1) # batch_size * 1 * seq_len
        # print("outp:",out)
        out = self.seq_len2one(out) # batch_size * 1 * 1?
        # print("outo:",out)
        probs = self.sigmoid_out(out) # batch_size * 1?
        # print("probs:",probs)
        probs = probs.squeeze() # batch_size
        # input()
        return probs
    
    def zero_state(self, batch_size=None):
        # helper function to reset the states at the beginning of each epoch
        if batch_size == None:
            batch_size = self.batch_size
        if self.use_cuda:
            hidden = (torch.zeros(1, self.batch_size, self.hidden_state_dim).cuda(),
                    torch.zeros(1, self.batch_size, self.hidden_state_dim).cuda())
        else:
            hidden = (torch.zeros(1, self.batch_size, self.hidden_state_dim),
                    torch.zeros(1, self.batch_size, self.hidden_state_dim))
        return hidden