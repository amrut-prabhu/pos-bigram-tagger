# python runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np

from collections import defaultdict
import _pickle as pickle

DEFAULT_TRANSITION_BACKOFF = 0.000000001
MIN_TRANSITION = -99999999

# Markers for start and end of sentence
START_MARKER = '<s>'
END_MARKER = '</s>'

tag_freq = defaultdict(int)
bitag_freq = {}

tags = {}
num_tokens = {}
tags_for_word = {}

transition = {} 
emission = {}

transition_backoff = {}
emission_backoff = {}

transition_singleton = {}
emission_singleton = {}

transition_smoothed = {}
emission_smoothed = {}

def tag_sentence(test_file, model_file, out_file):
    load_model(model_file)

    reader = open(test_file)
    test_lines = reader.readlines()
    reader.close()

    output_writer = open(out_file, 'w')

    for i in range(0, len(test_lines)):
        cur_line = test_lines[i].strip()
        cur_words = cur_line.split(' ')

        tagged_sentence = run_viterbi(cur_words)
        output_writer.write(tagged_sentence + '\n')
    
    output_writer.close()

    print('Finished...')

"""
obs: sequence of words in the sentence
"""
def run_viterbi(obs):
    """
    Viterbi probabilities: v(s, t)
    Maximum probability of all paths ending in state s_j at time t
    v(i, t) = max{ v(j, t-1) * a_ji * b_i(o_t) }   for j=1 to N
    """
    viterbi = []

    for t in range(0, len(obs)):
        viterbi.append({})

        for st in get_tags_for_word(obs[t]): 
            if t == 0:
                viterbi[t][st] = {
                    "prob": get_smoothed_transition(START_MARKER, st) + get_smoothed_emission(obs[t], st), 
                    "back_ptr": None
                }
            else:
                # Find state that maximises transition probability from the previous state
                max_transition_prob, back_ptr =  get_max_transition(viterbi, obs[t - 1], st, t - 1)
                        
                # Set most probable state and value of probability
                viterbi[t][st] = {
                    "prob": max_transition_prob + get_smoothed_emission(obs[t], st), 
                    "back_ptr": back_ptr
                }

    viterbi_tags = []

    # Get the most probable final state and its backtrack
    max_transition_prob, back_ptr =  get_max_transition(viterbi, obs[-1], END_MARKER, -1)
            
    viterbi_tags.append(back_ptr)
    current = back_ptr

    # Follow the backtrack till the first observation
    for t in range(len(viterbi) - 1, 0, -1):
        viterbi_tags.insert(0, viterbi[t][current]["back_ptr"])
        current = viterbi[t][current]["back_ptr"]

    return ' '.join([word + '/' + tag for word, tag in zip(obs, viterbi_tags)])

def get_max_transition(viterbi, curr_obs, curr_state, curr_state_idx):
    max_transition_prob = MIN_TRANSITION
    back_ptr = None

    for prev_state in get_tags_for_word(curr_obs):
        transition_prob = viterbi[curr_state_idx][prev_state]["prob"] + get_smoothed_transition(prev_state, curr_state)

        if transition_prob > max_transition_prob:
            max_transition_prob = transition_prob
            back_ptr = prev_state

    return max_transition_prob, back_ptr

def get_tags_for_word(word):
    if word in tags_for_word:
        return list(tags_for_word[word])
    return list(tags)

def get_emission_backoff(word):
    if word in emission_backoff:
        return emission_backoff[word]

    num_tags = len(tags)
    return float(1) / (num_tokens + num_tags)

def get_smoothed_emission(word, tag):
    if (word, tag) in emission_smoothed:
        return emission_smoothed[(word, tag)]
    else:
        lamda = 1 + emission_singleton.get(tag, 0)
        return math.log(float(lamda * get_emission_backoff(word)) / (tag_freq[tag] + lamda))

def get_smoothed_transition(prev_tag, tag):
    if (prev_tag, tag) in transition_smoothed:
        transition_probability = transition_smoothed[(prev_tag, tag)]
    else:
        lamda = 1 + transition_singleton.get(prev_tag, 0)
        transition_probability = math.log(float(lamda * transition_backoff.get(tag, DEFAULT_TRANSITION_BACKOFF)) / (tag_freq[prev_tag] + lamda))

    return transition_probability

def load_model(model_file):
    global tag_freq, bitag_freq
    global tags, tags_for_word, num_tokens
    global transition, emission

    global transition_backoff, emission_backoff
    global transition_singleton, emission_singleton
    global transition_smoothed, emission_smoothed

    f = open(model_file, "rb")
    model = pickle.load(f)

    tag_freq = model["tag_freq"]
    bitag_freq = model["bitag_freq"]

    tags = model["tags"]
    tags_for_word = model["tags_for_word"]
    num_tokens = model["num_tokens"]
    
    transition = model["transition"]
    emission = model["emission"]

    transition_backoff = model["transition_backoff"]
    emission_backoff = model["emission_backoff"]
    
    transition_singleton = model["transition_singleton"]
    emission_singleton = model["emission_singleton"]
    
    transition_smoothed = model["transition_smoothed"]
    emission_smoothed = model["emission_smoothed"]

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
