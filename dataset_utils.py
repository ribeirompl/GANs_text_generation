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

# Adapted from "https://github.com/pytorch/examples/blob/master/word_language_model/data.py" by Marco Ribeiro 2019
# Added comments and removed fixed train/valid/test filenames
# Added beginning and changed end of sentence identifiers (<s> and </s>)
# Changed Corpus object to allow external pre-cleaning step before tokenizing

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader


def get_batches(input_text, batch_size, seq_len):
    """
    input_text: list of entire corpus e.g. [1,3,54,55,39,45,2]
    Return generator object
    """
    # Alternative but not equivalent method (does not implement seq_len):
    # DataLoader(input_text, batch_size=32, num_workers=0)
    num_batches = len(input_text) // (batch_size*seq_len)
    
    for i in range(0, num_batches*batch_size*seq_len, batch_size*seq_len):
        out = torch.stack([input_text[i+j*seq_len:i+(j+1)*seq_len] for j in range(batch_size)])
        yield out

def print_vocab_to_file(idx2word, output_filepath="out.csv"):
    file = open(output_filepath,"w")
    for word in idx2word:
        file.write(word+"\n")
    file.close()
    
def save_tokenized_dataset(save_filepath, train_path, valid_path, test_path, clean_func, start_identifier=None, end_identifier=None, force_reload=False, remove_words_with_count_less_than=0, skip_count_check=False):
    """
    Tokenize and save the dataset if it hasn't already been done,
    otherwise just load it from the file
    """
    
    if os.path.isfile(os.path.join(save_filepath,'saved_dataset.data')) and not force_reload:
        this_dataset = torch.load(os.path.join(save_filepath,'saved_dataset.data'))
    else:
        this_dataset = Initialize_Dataset(train_path, valid_path, test_path, clean_func, start_identifier, end_identifier, remove_words_with_count_less_than, skip_count_check)
        os.makedirs(save_filepath, exist_ok=True)
        torch.save(this_dataset, os.path.join(save_filepath,'saved_dataset.data'))
        
    return this_dataset


class _Generic_Dataset(Dataset):
    """
    Generic dataset class that implements the required methods, getitem and len
    """
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self,i):
        return self.data[i]
    
    def __len__(self):
        return len(self.data)


class Initialize_Dataset(Dataset):
    def __init__(self, train_path=None, valid_path=None, test_path=None, clean_func = lambda x:x, start_identifier=None, end_identifier=None, remove_words_with_count_less_than=0, skip_count_check=False):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        corpus = Corpus(train_path, valid_path, test_path)
        self.dictionary, train_data, valid_data, test_data = corpus.clean_tokenize(clean_func, start_identifier=start_identifier, end_identifier=end_identifier, remove_words_with_count_less_than=remove_words_with_count_less_than, skip_count_check=skip_count_check)
        self.training_dataset = None if train_path is None else _Generic_Dataset(train_data)
        self.validation_dataset = None if valid_path is None else _Generic_Dataset(valid_data)
        self.test_dataset = None if test_path is None else _Generic_Dataset(test_data)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {} # dictionary of words in vocab with linked index
        self.idx2word = [] # list of words in vocab

    def add_word(self, word):
        # returns the index of the word in the list (whether it was added or not)
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    # Initialise object, optionally clean text, call tokenize.
    def __init__(self, train_path, valid_path=None, test_path=None):
        self.dictionary = Dictionary()

        assert os.path.exists(train_path)

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.train = None
        self.valid = None
        self.test = None

    def clean_tokenize(self, clean_func, start_identifier=None, end_identifier=None, remove_words_with_count_less_than=0, skip_count_check=False):
        # clean text and then tokenize

        count_dict = {}

        if self.train_path is not None:
            self.train, count_dict = self._clean_tokenize(self.train_path, clean_func, start_identifier, end_identifier, count_dict)

        if self.valid_path is not None:
            self.valid, count_dict = self._clean_tokenize(self.valid_path, clean_func, start_identifier, end_identifier, count_dict)
        
        if self.test_path is not None:
            self.test, count_dict = self._clean_tokenize(self.test_path, clean_func, start_identifier, end_identifier, count_dict)

        # add _unk_ tag to dictionary
        if skip_count_check == False:
            new_dictionary = Dictionary()
            new_dictionary.add_word("_unk_")

            if self.train_path is not None:
                for i, word_idx in enumerate(self.train):
                    word_idx = word_idx.item() # convert away from tensor
                    if count_dict[word_idx] < remove_words_with_count_less_than:
                        self.train[i] = new_dictionary.word2idx["_unk_"]
                    else:
                        new_dictionary.add_word(self.dictionary.idx2word[word_idx])
            
            if self.valid_path is not None:
                for i, word_idx in enumerate(self.valid):
                    word_idx = word_idx.item() # convert away from tensor
                    if count_dict[word_idx] < remove_words_with_count_less_than:
                        self.valid[i] = new_dictionary.word2idx["_unk_"]
                    else:
                        new_dictionary.add_word(self.dictionary.idx2word[word_idx])

            if self.test_path is not None:
                for i, word_idx in enumerate(self.test):
                    word_idx = word_idx.item() # convert away from tensor
                    if count_dict[word_idx] < remove_words_with_count_less_than:
                        self.test[i] = new_dictionary.word2idx["_unk_"]
                    else:
                        new_dictionary.add_word(self.dictionary.idx2word[word_idx])
                
            self.dictionary = new_dictionary

        return self.dictionary, self.train, self.valid, self.test
    
    # deprecated
    def tokenize(self, start_identifier=None, end_identifier=None):
        # only tokenize, no pre-cleaning of text
        if self.train_path is not None:
            self.train = self._clean_tokenize(self.train_path, None, start_identifier, end_identifier)

        if self.valid_path is not None:
            self.valid = self._clean_tokenize(self.valid_path, None, start_identifier, end_identifier)
        
        if self.test_path is not None:
            self.test = self._clean_tokenize(self.test_path, None, start_identifier, end_identifier)
        
        return self.dictionary, self.train, self.valid, self.test
    
    def _clean_tokenize(self, path, clean_func=None, start_identifier=None, end_identifier=None, count_dict={}):
        # generic tokenizing method with optional cleaning
        start_identifier = [] if start_identifier is None else [start_identifier]
        end_identifier = [] if end_identifier is None else [end_identifier]

        text_data = open(path).readlines()

        idss = []
        for line in text_data:
            if clean_func is not None:
                line = clean_func(line)
            words = start_identifier + line.split() + end_identifier
            ids = []
            for word in words:
                self.dictionary.add_word(word) # temporary dictionary, used to lookup word when adding _unk_
                idx = self.dictionary.word2idx[word]
                ids.append(idx)
                if idx not in count_dict:
                    count_dict[idx] = 1
                else:
                    count_dict[idx] += 1
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)

        return ids, count_dict


# Test function
if __name__ == "__main__":
    __ = get_batches(torch.tensor([x for x in range(21)]),2,5)
    test_array = []
    for _ in __:
        test_array.append(_.numpy().tolist())

    assert test_array == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]], "Should be [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]]"
    
    print("Everything passed")