# python buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np

from collections import defaultdict
import _pickle as pickle

# Markers for start and end of sentence
START_MARKER = '<s>'
END_MARKER = '</s>'

# All tags that appear in the corpus
tags = set()
# All words that appear in the corpus
words = set()

# TODO: change name
word_to_tag = {}

""" Back-off Probabilities """
transition_backoff = defaultdict(int)
emission_backoff = defaultdict(int)

""" Singleton counts """
transition_singleton = defaultdict(int)
emission_singleton = defaultdict(int)

""" 1-count smoothed probabilities """
transition_one_count = defaultdict(int)
emission_smoothed = defaultdict(int)

# Num tokens (words)
num_tokens = 0

def train_model(train_file, model_file):
    word_tag_freq, tag_freq, bitag_freq = get_freqs(train_file)

    transition, emission = get_probs(word_tag_freq, tag_freq, bitag_freq)

    compute_backoff()
    compute_transition_singleton(tag_freq, word_tag_freq)
    compute_smoothed_probabilities(train_file, bitag_freq, tag_freq, word_tag_freq)

    save_model(model_file, transition, emission, tags, word_tag_freq, tag_freq, bitag_freq) 
    
    print('Finished...')
    
def save_model(model_file, transition, emission, tags, word_tag_freq, tag_freq, bitag_freq):
    model = {
        "tags" : tags, 
        "transition": transition,
        "emission": emission,
        "tag_freq": tag_freq, 
        "bitag_freq": bitag_freq,

        "word_to_tag" : word_to_tag,
        "transition_backoff" : transition_backoff, 
        "emission_backoff" : emission_backoff,
        "transition_singleton" : transition_singleton,
        "emission_singleton" : emission_singleton,
        "transition_smoothed" : transition_one_count,
        "emission_smoothed" : emission_smoothed,
        "num_tokens" : num_tokens
    }

    output = open(model_file, 'wb')
    pickle.dump(model, output)
    output.close()

def compute_backoff():
    V = len(tags)

    for word in emission_backoff:
        emission_backoff[word] = float(1 + emission_backoff[word]) / float(num_tokens + V)

    for tag in transition_backoff:
        transition_backoff[tag] = float(transition_backoff[tag]) / float(num_tokens)

def compute_transition_singleton(tag_freq, word_tag_freq):
    for tag in tags:
        if tag_freq[tag] == 1:
            transition_singleton[tag] += 1

    for word in words:
        for tag in tags:
            word_tag = (word, tag)
            if word_tag in word_tag_freq and word_tag_freq[word_tag] == 1:
                emission_singleton[tag] += 1

def compute_smoothed_probabilities(train_file, bitag_freq, tag_freq, word_tag_freq):
    reader = open(train_file)
    train_lines = reader.readlines()
    reader.close()

    for i in range(0, len(train_lines)):
        cur_line = train_lines[i].strip()
        cur_word_tag_pairs = cur_line.split(' ')

        prev_tag = START_MARKER
        for j in range(0, len(cur_word_tag_pairs)):
            word, tag = splitWordAndTag(cur_word_tag_pairs[j])

            bitag = (prev_tag, tag)

            lamda = 1 + transition_singleton[prev_tag]
            transition_one_count[bitag] = \
                float(bitag_freq[bitag] + lamda * transition_backoff[tag]) / float(tag_freq[prev_tag] + lamda) # math.log(

            prev_tag = tag

    for word, tags_set in word_to_tag.items():
        for tag in tags_set:
            word_tag = (word, tag)

            lamda = 1 + emission_singleton[tag]
            emission_smoothed[word_tag] = \
                float(word_tag_freq[word_tag] + lamda * emission_backoff[word]) / float(tag_freq[tag] + lamda) # math.log(

def get_freqs(train_file):   
    global num_tokens 

    reader = open(train_file)
    train_lines = reader.readlines()
    reader.close()

    # Frequency of (word, tag) tuples
    word_tag_freq = defaultdict(int)

    # Frequency of tag
    tag_freq = defaultdict(int)

    # Frequency of (tag I, tag J) tuples, where tag J is the tag assigned directly after tag I
    bitag_freq = defaultdict(int)

    for i in range(0, len(train_lines)):
        cur_line = train_lines[i].strip()
        cur_word_tag_pairs = cur_line.split(' ')

        prev_tag = START_MARKER
        tag_freq[START_MARKER] += 1

        # TODO: keep these?
        num_tokens += 1 
        # tags.add(START_MARKER)

        for j in range(0, len(cur_word_tag_pairs)):
            word, tag = splitWordAndTag(cur_word_tag_pairs[j])

            tag_freq[tag] += 1
            word_tag_freq[(word, tag)] += 1
            bitag_freq[(prev_tag, tag)] += 1 # order of the sequence matters

            num_tokens += 1

            tags.add(tag)
            words.add(word)

            transition_backoff[tag] += 1
            emission_backoff[word] += 1

            if word not in word_to_tag:
                word_to_tag[word] = set()
            word_to_tag[word].add(tag)

            prev_tag = tag

        bitag_freq[(prev_tag, END_MARKER)] += 1

    return word_tag_freq, tag_freq, bitag_freq

def get_probs(word_tag_freq, tag_freq, bitag_freq):   
    """
    Transition probabilities: a_ij
    Probability of transitioning from state s_i to state s_j (states represent POS tags)
    a(tag J | tag I) = count(tag I, tag J - in this order) / count(tag I)
    Usage: transition[(tag I, tag J)]
    """
    transition = defaultdict(int)

    """
    Emission probabilities: b_i(o_t)
    Probability of observing an observation o_t from state s_i (observations are words)
    b(word T | tag I) = count(word T and tag I) / count(tag I)
    Usage: emission[(word, tag)]
    """
    emission = defaultdict(int)
    
    V = len(tags)
    for (prev_tag, curr_tag) in bitag_freq:
        transition[(prev_tag, curr_tag)] =  float(1 + bitag_freq[(prev_tag, curr_tag)]) / float(V + tag_freq[prev_tag]) # math.log(
    
    for (word, tag) in word_tag_freq:
        emission[(word, tag)] = float(word_tag_freq[(word, tag)]) / float(tag_freq[tag]) # math.log(

    return transition, emission    

def splitWordAndTag(string):
    splitIdx = string.rfind('/')
    word = string[:splitIdx]
    tag = string[splitIdx+1:]
    return word, tag

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
