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

# All words and tags that appear in the corpus
words = set()
tags = set()

# Frequency of tag
tag_freq = defaultdict(int)

# Frequency of (tag I, tag J) tuples, where tag J is the tag appearing directly after tag I
bitag_freq = defaultdict(int)

# Frequency of (word, tag) pairs
word_tag_freq = defaultdict(int)

# The set of tags that have been assigned to the word in the corpus
tags_for_word = {}

# Number of tokens (words) in the corpus
num_tokens = 0

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

# Back-off Probabilities
transition_backoff = defaultdict(int)
emission_backoff = defaultdict(int)

# Singleton counts
transition_singleton = defaultdict(int)
emission_singleton = defaultdict(int)

# 1-count smoothed probabilities
transition_smoothed = defaultdict(int)
emission_smoothed = defaultdict(int)

def train_model(train_file, model_file):
    compute_freqs(train_file)
    compute_basic_probs()

    compute_backoff_probs()
    compute_transition_singletons()
    compute_smoothed_probs(train_file)

    save_model(model_file) 
    
    print('Finished...')

def compute_backoff_probs():
    V = len(tags)

    for word in emission_backoff:
        emission_backoff[word] = float(1 + emission_backoff[word]) / (num_tokens + V)

    for tag in transition_backoff:
        transition_backoff[tag] = float(transition_backoff[tag]) / num_tokens

def compute_transition_singletons():
    for tag in tags:
        if tag_freq[tag] == 1:
            transition_singleton[tag] += 1

    for word in words:
        for tag in tags:
            if (word, tag) in word_tag_freq and word_tag_freq[(word, tag)] == 1:
                emission_singleton[tag] += 1

def compute_smoothed_probs(train_file):
    reader = open(train_file)
    train_lines = reader.readlines()
    reader.close()

    for i in range(0, len(train_lines)):
        cur_line = train_lines[i].strip()
        cur_word_tag_pairs = cur_line.split(' ')

        prev_tag = START_MARKER

        for j in range(0, len(cur_word_tag_pairs)):
            word, tag = splitWordAndTag(cur_word_tag_pairs[j])

            LAMBDA = 1 + transition_singleton[prev_tag]

            transition_smoothed[(prev_tag, tag)] = \
                math.log(float(bitag_freq[(prev_tag, tag)] + LAMBDA * transition_backoff[tag]) / (tag_freq[prev_tag] + LAMBDA))
                # float(bitag_freq[(prev_tag, tag)] + LAMBDA * transition_backoff[tag]) / (tag_freq[prev_tag] + LAMBDA)

            prev_tag = tag

    for word, assigned_tags in tags_for_word.items():
        for tag in assigned_tags:
            word_tag = (word, tag)

            LAMBDA = 1 + emission_singleton[tag]

            emission_smoothed[word_tag] = \
                math.log(float(word_tag_freq[word_tag] + LAMBDA * emission_backoff[word]) / (tag_freq[tag] + LAMBDA))
                # float(word_tag_freq[word_tag] + LAMBDA * emission_backoff[word]) / (tag_freq[tag] + LAMBDA)

def compute_basic_probs():   
    global transition, emission
    
    num_tags = len(tags)
    for (prev_tag, curr_tag) in bitag_freq:
        # transition[(prev_tag, curr_tag)] =  float(1 + bitag_freq[(prev_tag, curr_tag)]) / (num_tags + tag_freq[prev_tag])
        transition[(prev_tag, curr_tag)] =  math.log(float(1 + bitag_freq[(prev_tag, curr_tag)]) / (num_tags + tag_freq[prev_tag]))
    
    for (word, tag) in word_tag_freq:
        # emission[(word, tag)] = float(word_tag_freq[(word, tag)]) / tag_freq[tag]
        emission[(word, tag)] = math.log(float(word_tag_freq[(word, tag)]) / tag_freq[tag])

def compute_freqs(train_file):   
    global num_tokens, tag_freq, bitag_freq, word_tag_freq

    reader = open(train_file)
    train_lines = reader.readlines()
    reader.close()

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
            bitag_freq[(prev_tag, tag)] += 1 # order of the sequence matters
            word_tag_freq[(word, tag)] += 1

            num_tokens += 1

            tags.add(tag)
            words.add(word)

            transition_backoff[tag] += 1
            emission_backoff[word] += 1

            # To speed up viterbi calculation
            if word not in tags_for_word:
                tags_for_word[word] = set()
            tags_for_word[word].add(tag)

            prev_tag = tag

        bitag_freq[(prev_tag, END_MARKER)] += 1

def splitWordAndTag(string):
    splitIdx = string.rfind('/')
    word = string[:splitIdx]
    tag = string[splitIdx+1:]
    return word, tag

def save_model(model_file):
    model = {
        "tag_freq": tag_freq, 
        "bitag_freq": bitag_freq,

        "tags" : tags, 
        "num_tokens" : num_tokens,
        "tags_for_word" : tags_for_word,

        "transition": transition,
        "emission": emission,

        "transition_backoff" : transition_backoff, 
        "emission_backoff" : emission_backoff,

        "transition_singleton" : transition_singleton,
        "emission_singleton" : emission_singleton,
        
        "transition_smoothed" : transition_smoothed,
        "emission_smoothed" : emission_smoothed
    }

    output = open(model_file, 'wb')
    pickle.dump(model, output)
    output.close()

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
