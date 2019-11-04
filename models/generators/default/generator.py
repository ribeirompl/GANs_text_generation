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
import numpy as np

class Generator_model(nn.Module):
    """
    Generator model for a GAN

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
    initial_temperature : float
        The initial value of the temperature parameter for the Gumbel-Softmax approximation (default = 5)
    skip_gumbel : bool
        Whether to skip the gumbel-softmax procedure for pre-trainings
    
    Architecture
    ------------
    "partial sentence" -> Embedding Layer -> LSTM -> Linear -> Gumbel-Softmax -> out
    """

    def __init__(self, use_cuda, vocab_size, batch_size, seq_len, embedding_size, hidden_state_dim, initial_temperature=5, skip_gumbel=False, control_policy='lin'):
        super().__init__() # inherit the parent's methods and variables (from nn.Module)
        
        # Store instance variables
        self.use_cuda = use_cuda
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.hidden_state_dim = hidden_state_dim
        self.skip_gumbel = skip_gumbel
        self.control_policy = control_policy

        # Initialise temperature parameters
        self.initial_temperature = initial_temperature # store initial temperature param for calculation
        self.temperature = initial_temperature # the current value of the temperature parameter

        # initialise the network architecture's components
        self.embeddings_layer = nn.Embedding(vocab_size, embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_state_dim, batch_first=True)
        self.lstm_to_output = nn.Linear(hidden_state_dim, vocab_size)
    
#         self.init_params()
        
    def forward(self, sentences, prev_states):
        """
        Forward method of Generator model

        Parameters
        ----------
        sentences : torch.Tensor(float32) -- (batch_size * 1)
            single word in the text sequence
        prev_states : (h, c) -- ((1 * batch_size * hidden_state_dim), (1 * batch_size * hidden_state_dim))
            The previous states for the LSTM

        Returns
        -------
        probs : torch.Tensor(float32) -- (batch_size * vocab_size)
            Output containing semi-one_hot representation of the vocab, indicating the most probable word
        states : (h, c) -- ((1 * batch_size * hidden_state_dim), (1 * batch_size * hidden_state_dim))
            The hidden state and cell state output of the LSTM
        """

        embeds = self.embeddings_layer(sentences) # batch_size * 1 * embedding_dim

        output, states = self.lstm_layer(embeds, prev_states)
        # output: batch_size * 1 * hidden_state_dim
        # states: (1 * batch_size * hidden_state_dim, 1 * batch_size * hidden_state_dim)
        
        output = output.squeeze(1) # batch_size * hidden_state_dim
        
        probs = self.lstm_to_output(output) # batch_size * vocab_size
        
        # for pre-training
        if not self.skip_gumbel:
            gumbel = self.add_gumbel(probs) # batch_size * vocab_size

            if self.training: # check if currently in training mode
                probs = F.softmax(gumbel/self.temperature, dim=1) # batch_size * vocab_size
            else:
                probs = F.softmax(gumbel/1, dim=1) # batch_size * vocab_sizes

        return probs, states

    def predict(self, inp, i_temperature=2500, N_temperature=5000, batch_size=16, seq_len=32, top_k=1, control_policy='lin'):
        """
        Predict method of Generator model

        Parameters
        ----------
        inp : torch.Tensor -- (batch_size * initial_string_len)
            The initial sentence string. Each text sequence should be the same length
            e.g. [["I","AM","GOING"],["WHAT","IS","THE"]] (but the index/embedded representation of the words)
        i_temperature : int
            The current iteration of the training process, used for calculating the new temperature parameter
        N_temperature : int
            The total number of iterations before the temperature parameter should be 1
        batch_size : int
            Current batch size (can be different value for evaluation, compared to during training)
        seq_len : int
            Current sequence length (can be different value for evaluation, compared to during training)
        top_k : int
            When predicting the next word, the k'th highest probable words are selected and one is chosen from them

        Returns
        -------
        out : torch.Tensor -- (batch_size * seq_len)
            The output sentence string in index/embedded form
        all_probs : torch.Tensor(float32) -- (batch_size * seq_len * vocab_size)
            All the outputs from the LSTM at each timestep/text sequence, for input to the Discriminator
        """

        if self.use_cuda: inp = inp.cuda() # send inp to GPU
        initial_string_len = inp.size(1) # retrieve the string length of inp
        num_needed_words = seq_len - initial_string_len # number of words to generate

        # update temperature parameter
        self.temperature = Generator_model.get_fixed_temperature(self.initial_temperature, i=i_temperature, N=N_temperature, control_policy=self.control_policy)

        if self.use_cuda: 
            all_probs = torch.zeros(batch_size, seq_len, self.vocab_size).cuda()
            states = (self._noise(batch_size).cuda(), self._noise(batch_size).cuda())
        else:
            all_probs = torch.zeros(batch_size, seq_len, self.vocab_size)
            states = (self._noise(batch_size), self._noise(batch_size))
        
        # process the initial sentence string and retrieve the probs
        for i in range(initial_string_len):
            probs, states = self.forward(inp[:,i].unsqueeze(1), states) # unsqueeze dimension, for inp to be (batch_size * 1), not (batch_size)
            all_probs[:, i] = probs
        
        out = inp # store the initial sentence string in the output

        # generate words until the length of out is seq_len
        for i in range(num_needed_words):
            next_token = out[:,-1].unsqueeze(1) # unsqueeze to have size (batch_size * 1), instead of (batch_size)
            
            probs, states = self.forward(next_token, states)

            _ , next_possible_tokens = torch.topk(probs, k=top_k, dim=1, sorted=False) # retrieve the top k possible word tokens
            # next_possible_tokens size: (batch_size * top_k)

            next_possible_tokens = next_possible_tokens.detach() # detach to stop calculating gradients

            # generate random int in range(top_k), with size: (batch_size * 1) for choosing the next word token
            choice_idx = torch.randint(next_possible_tokens.size(1), (next_possible_tokens.size(0),1))
            if self.use_cuda: choice_idx = choice_idx.cuda()

            next_token = torch.gather(next_possible_tokens, 1, choice_idx) # get next word token from the chosen index
            
            out = torch.cat((out, next_token), dim=1) # concatenate the chosen word token to the end of out
            
            all_probs[:, initial_string_len+i] = probs # store the probs for input to the discriminator

        return out, all_probs

    def sample_sentence(self, idx2word_dict, inp, batch_size=16, seq_len=32, top_k=1):
        """
        Wrapper method of predict, to convert to string and trim only sentence from output
        (Used in evaluation mode)

        Parameters
        ----------
        idx2word_dict : dict{vocab_size}
            Dictionary containing the index to word pairs, for converting result to human-intelligible sentence
        inp : torch.Tensor -- (batch_size * initial_string_len)
            The initial sentence string 
            e.g. [["I","AM","GOING"],["WHAT","IS","THE"]] (but the index/embedded representation of the words)
        batch_size : int
            Current batch size (can be different value for evaluation, compared to during training)
        seq_len : int
            Current sequence length (can be different value for evaluation, compared to during training)
        top_k : int
            When predicting the next word, the k'th highest probable words are selected and one is chosen from them

        Returns
        -------
        sentences : list of strings -- (batch_size * seq_len)
            The output sentence string in word form
        """

        out, _ = self.predict(inp, batch_size=batch_size, seq_len=seq_len, top_k=top_k)
        _nested_word_sequences = [[idx2word_dict[_item] for _item in _list] for _list in out] # convert indices to words
        sentences = [" ".join(_word_sequences) for _word_sequences in _nested_word_sequences] # list of sentences

        return sentences

    def _noise(self, batch_size=None, hidden_state_dim=None):
        """
        Generates from gaussian-sampled random values

        Parameters
        ----------
        batch_size : int (optional)
            Current batch size (if not given, the original initialised batch_size is used)
        hidden_state_dim : int (optional)
            Size of the LSTM memory states (if not given, the original initialised batch_size is used)
        
        Returns
        -------
        out : torch.tensor(float32, requires_grad=True) -- (1 * batch_size * hidden_state_dim)
            gaussian sampled noise values
        """

        if batch_size is None: batch_size = self.batch_size
        if hidden_state_dim is None: hidden_state_dim = self.hidden_state_dim
        
        out = torch.randn(1, batch_size, hidden_state_dim)
        out.requires_grad = True
        return out

    @staticmethod
    def get_fixed_temperature(initial_temperature, i, N, control_policy):
        """
        Adapted from https://github.com/williamSYSU (MIT license)

        Calculate the next temperature value for annealing the temperature during training down to 1
        (for the Gumbel-Softmax approximation)

        Parameters
        ----------
        initial_temperature : float
            The initial temperature, not the current temperature. e.g. 5
        i : int
            The current iteration of the training process, used for calculating the new temperature parameter
        N : int
            The total number of iterations before the temperature parameter should be 1
        control_policy : String
            The temperature control policy to use for annealing the temperature parameter

        Returns
        -------
        new_temp : float
            The new temperature parameter value
        """
        
        # return 1, if iteration is already greater than the max number of iterations
        if i > N:
            return 1

        if control_policy == 'none': # no temperature control policy
            new_temp = initial_temperature 
        elif control_policy == 'lin': # linear decrease
            new_temp = 1 + (N-i)/(N-1) * (initial_temperature - 1)
        elif control_policy == 'exp': # exponential decrease
            new_temp = initial_temperature**((N-i)/N)  
        elif control_policy == 'log': # logarithmic decrease
            new_temp = 1 + (initial_temperature - 1) / np.log(N) * np.log(N-i + 1)
        elif control_policy == 'sqrt':
            new_temp = (initial_temperature - 1) / np.sqrt(N - 1) * np.sqrt(N-i) + 1
        elif control_policy == 'ericjang': # https://arxiv.org/pdf/1611.01144.pdf
            new_temp = max(0.5, np.exp(-3e-5*((i/N)*23105))) # decays from 1 to 0.5 after 23105, therefore adjusted
        else:
            raise Exception("Unknown control_policy. Select: 'none', 'lin', 'exp', 'log' or 'sqrt'")

        # unnecessary, but just for precaution
        if new_temp < 1e-10: new_temp=1e-10 # fix temperature to above 0

        return new_temp

    def add_gumbel(self, o_t, eps=1e-10):
        """
        Adapted from https://gist.github.com/ericjang/1001afd374c2c3b7752545ce6d9ed349#file-gumbel-softmax-py (MIT license)
        
        Add o_t by a vector sampled from Gumbel(0,1)
        
        Parameters
        ----------
        o_t: torch.Tensor
            The Tensor that should have values, sampled from the Gumbel distribution, added to it
        
        Returns:
        _ : torch.Tensor
            Sampled Gumbel values added to the input o_t vector
        """

        U = torch.rand(o_t.size()).cuda() if self.use_cuda else torch.rand(o_t.size())
        gumbel = -torch.log(-torch.log(U + eps) + eps)

        return o_t + gumbel

    # @deprecated
    def calculate_perplexity(self, input_sentence):
        """
        # currently doesn't support batches, due to differing sentence lengths
            input_sentence: tensor(1 * string_len)
            e.g. [["<s>","I","AM","GOING","HOME","</s>"]] (but idx representation of the words)

        """
        
        hidden = self.zero_state(1)
        if self.use_cuda: hidden = (hidden[0].cuda(), hidden[1].cuda())
        perplexity = torch.tensor([])
        if self.use_cuda: perplexity = perplexity.cuda()

        for i in range (input_sentence.size(1)-1):
            # loop length:
            # if string == "a","b","c".
            # then it is done for "a" and for "a","b"

            probs, states = self.forward(input_sentence[:,i].unsqueeze(1), hidden) # unsqueeze to have size: batch_size * 1, instead of batch_size

            # the target is always the next character
            targets = input_sentence[:,i+1]
            

            # use cross entropy loss to compare output of forward pass with targets
            # loss = nn.CrossEntropyLoss(probs, targets).item()
            loss = F.cross_entropy(probs, targets).unsqueeze(0)
            # exponentiate cross-entropy loss to calculate perplexity
            
            perplexity = torch.cat((perplexity, torch.exp(loss)))

        perplexity = torch.mean(perplexity)
        return perplexity

    def zero_state(self, batch_size):
        # helper function to reset the states at the beginning of each epoch when pre-training
        return (torch.zeros(1, batch_size, self.hidden_state_dim),
                torch.zeros(1, batch_size, self.hidden_state_dim))