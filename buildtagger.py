# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np

from collections import defaultdict
import pprint

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    reader = open(train_file)
    train_lines = reader.readlines()
    reader.close()

    # Markers for start and end of sentence
    START_MARKER = '<s>'
    END_MARKER = '</s>'

    # Frequency of (word, tag) tuples
    word_tag_freq = defaultdict(int)

    # Frequency of tag
    tag_freq = defaultdict(int)
    
    # Frequency of (tag I, tag J) tuples, where tag J is the tag assigned directly after tag I
    bitag_freq = defaultdict(int)

    for i in range(0, len(train_lines)):
        cur_line = train_lines[i].strip()
        cur_tags = cur_line.split(' ')

        prev_tag = START_MARKER
        tag_freq[START_MARKER] += 1
        for j in range(0, len(cur_tags)):
            word, tag = getWordAndTag(cur_tags[j])

            word_tag_freq[(word, tag)] += 1
            tag_freq[tag] += 1
            bitag_freq[(prev_tag, tag)] += 1

            prev_tag = tag

        bitag_freq[(prev_tag, END_MARKER)] += 1
    # print_freqs(word_tag_freq, tag_freq, bitag_freq)

    """
    Transition probabilities: a_ij
    Probability of transitioning from state s_i to state s_j (states represent POS tags)
    P(tag J | tag I)
    """
    transition = {}

    """
    Emission probabilities: b_i(o_t)
    Probability of observing an observation o_t from state s_i (observations are words)
    P(word T | tag I)
    """
    emission = {}
    
    print('Finished...')

def getWordAndTag(string):
    splitIdx = string.rfind('/')
    word = string[:splitIdx]
    tag = string[splitIdx+1:]
    return word, tag

def print_freqs(word_tag_freq, tag_freq, bitag_freq):
    print("\nword_tag_freq:")
    pprint.pprint(word_tag_freq)
    print("\ntag_freq:")
    pprint.pprint(tag_freq)
    print("\nbitag_freq:")
    pprint.pprint(bitag_freq)

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
