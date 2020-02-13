import os
import re
import sys
import time
import numpy as np
import torch
# import argparse

from datetime import datetime

# My modules
from dataset_utils import save_tokenized_dataset
from models.generators.default.generator import Generator_model as Generator

use_cuda = True

np.random.seed(234); # Fix seed
torch.manual_seed(234); # Fix seed

dataset = save_tokenized_dataset('./dataset/preread/ez_cs/',
                                        './dataset/ez_cs_dataset/train.txt',
                                        './dataset/ez_cs_dataset/dev.txt',
                                        './dataset/ez_cs_dataset/test.txt',
                                        # lambda sentence: sentence.replace("_zul","").replace("_eng",""),
                                        lambda x: x,
                                        "<s>", "</s>", force_reload=False, skip_count_check=True)

print("Dataset: EZ_dataset")
print("Vocab size:",len(dataset.dictionary.idx2word))
print("-" * 80)

gen_embedding_size = 48
gen_lstm_size = 48
batch_size = 16
seq_len = 5
initial_temperature = 5.0
num_batches = len(dataset.training_dataset) // (batch_size*seq_len)
print("num batches:",num_batches)

vocab_size = len(dataset.dictionary.idx2word)
int_to_vocab, vocab_to_int = dataset.dictionary.idx2word, dataset.dictionary.word2idx
current_run_time = datetime.now().strftime('%Y-%m-%d_%H-%M') # time for saving filename appropriately

gen = Generator(use_cuda, vocab_size, batch_size, seq_len, gen_embedding_size, gen_lstm_size, initial_temperature=5.0, control_policy='lin')
gen.cuda()

state = torch.load("./saved_training_e-4_i-856.pth")
gen.load_state_dict(state['gen_model'])

gen.eval() # change generator into evaluation mode (disables dropout/batch-normalisation)
example_batch_size = 50
example_seq_len = 30

random_SOS = torch.tensor([[vocab_to_int["<s>"]]]*example_batch_size) # random start of sentence
if use_cuda: random_SOS = random_SOS.cuda()

total_text = ""
lines=0

start = time.time()
try:
    while lines < 3000:
        example_text = gen.sample_sentence(int_to_vocab, random_SOS, example_batch_size, example_seq_len)
        example_text = "\n".join(example_text)
        example_text = re.sub(r"^<s> ", r"", example_text)
        example_text = re.sub(r"</s> ", r"\n", example_text)
        example_text = re.sub(r" </s>", r"\n", example_text)
        example_text = re.sub(r"<s> ", r"", example_text)
        example_text = re.sub(r" <s>", r"", example_text)
        example_text = re.sub(r"<s>", r"", example_text)
        example_text = re.sub(r"\n\n", r"\n", example_text)
        example_text = re.sub(r"^\n", r"", example_text)
        example_text = re.sub(r"\n$", r"", example_text)
        example_text = re.sub(r"^ ", r"", example_text)
        example_text = re.sub(r" $", r"", example_text)
        lines += example_text.count("\n") +1
        total_text += "\n{}".format(example_text)

except KeyboardInterrupt:
    pass
print("time:", time.time()-start)
with open("output_text-"+current_run_time+".txt", 'w') as f:
    f.write(total_text)
# print(total_text)