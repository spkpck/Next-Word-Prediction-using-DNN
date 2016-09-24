import numpy as np
import theano as theano
import theano.tensor as T
import time
import operator
from utils import load_data, load_model_parameters_theano, generate_sentences, generate_word
from gru_theano import *

import sys
# Load data (this may take a few minutes)
VOCABULARY_SIZE = 8000
X_train, y_train, word_to_index, index_to_word = load_data("data/reddit-comments-2015.csv", VOCABULARY_SIZE)#reddit-comments-2015.csv
# Load parameters of pre-trained model
model = load_model_parameters_theano("data/pretrained.npz")
#generate_sentences(model, 5, index_to_word, word_to_index)
new_sent = raw_input('Start typing:')
print ("_____________________________________________")
generate_word(model,5, index_to_word, word_to_index,new_sent)
while(1):
    ui= raw_input("Type in the next word")
    new_sent = new_sent +" "+ ui
    print (new_sent)
    generate_word(model,5, index_to_word, word_to_index,new_sent)
    if ui=='.':
        break