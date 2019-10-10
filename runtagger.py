# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np

from collections import defaultdict
import pprint
# import json
import _pickle as pickle

START_MARKER = '<s>'
END_MARKER = '</s>'

transition = {} 
emission = {}
word_to_tag = {}
tag_freq = {}
tags = {}

transition_backoff = {}
emission_backoff = {}
transition_singleton = {}
emission_singleton = {}
transition_one_count = {}
emission_smoothed = {}
bitag_freq = {}
num_tokens = {}

def tag_sentence(test_file, model_file, out_file):
    load_model(model_file)
    # print_probs(transition, emission)

    reader = open(test_file)
    test_lines = reader.readlines()
    reader.close()

    output_writer = open(out_file, 'w')

    for i in range(0, len(test_lines)):
        cur_line = test_lines[i].strip()
        cur_words = cur_line.split(' ')

        tagged_sentence = run_viterbi(cur_words, transition, emission)
        output_writer.write(tagged_sentence + '\n')
    
    output_writer.close()

    print('Finished...')

# done
def get_tag_freq(u):
    if u in tag_freq:
        return tag_freq[u]
    return 0

# done
def get_tags(word):
    if word in word_to_tag:
        return word_to_tag[word]
    return tags

# done
def get_emission_backoff(word):
    if word in emission_backoff:
        return emission_backoff[word]

    V = len(tags)
    return float(1) / float(num_tokens + V)

# done
def get_smoothed_emission(word, tag):
    if (word, tag) in emission_smoothed:
        return emission_smoothed[(word, tag)]
    else:
        lamda = 1 + emission_singleton.get(tag, 0)
        return float(lamda * get_emission_backoff(word)) / float(tag_freq[tag] + lamda) # math.log(

# done
def get_smoothed_transition(tag_j, tag_i):
    if (tag_j, tag_i) in transition_one_count:
        """ transition probability P(tag_i | tag_i - 1) """
        transition_probability = transition_one_count[(tag_j, tag_i)]
    else:
        """ Smoothed transition probability """
        tag_freq = get_tag_freq(tag_j)
        lamda = 1 + transition_singleton.get(tag_j, 0)
        transition_probability = float(lamda * transition_backoff[tag_i]) / float(tag_freq + lamda) # math.log(
    return transition_probability


"""
obs: sequence of words in the sentence
states: POS tags
transition: transition probabilities
emission: emission probabilities
"""
def run_viterbi(obs, transition, emission):
    """
    Viterbi probabilities: v(s, t)
    Maximum probability of all paths ending in state s_j at time t
    v(i, t) = max{ v(j, t-1) * a_ji * b_i(o_t) }   for j=1 to N
    """
    viterbi = []

    global tags
    states = list(tags)
    # for st in states:
    #     viterbi[0][st] = {
    #         "prob": get_smoothed_transition(START_MARKER, st) * get_smoothed_emission(obs[0], st), # +
    #         "back_ptr": None
    #     }
        
    for t in range(0, len(obs)):
        viterbi.append({})

        # TODO: change this and the next loop to int
        for st in states:   # for st in get_tags(obs[t]):
            """ Base case needs to be handled here """
            if t == 0:
                """ # calculate score using P(Ci|'^', '^') and P(Wj|Ci) """
                viterbi[t][st] = {
                    "prob": get_smoothed_transition(START_MARKER, st) * get_smoothed_emission(obs[t], st), # +
                    "back_ptr": None
                }
            else:

                # Find state that maximises transition probability from the previous state
                max_transition_prob = -100000
                back_ptr = None

                # max_viterbi = HMM.transition_minimum

                for prev_state in states:  # for st in get_tags(obs[t - 1]):
                    transition_prob = viterbi[t - 1][prev_state]["prob"] * get_smoothed_transition(prev_state, st) # +

                    if transition_prob > max_transition_prob:
                        max_transition_prob = transition_prob
                        back_ptr = prev_state
                        
                # Set most probable state and value of probability
                viterbi[t][st] = {
                    "prob": max_transition_prob * get_smoothed_emission(obs[t], st), # +
                    "back_ptr": back_ptr
                }

    tags = []

    # Get the most probable final state and its backtrack
    max_transition_prob = -100000
    # max_transition_prob = viterbi[len(obs)][states[0]]["prob"] * transition[(states[0], END_MARKER)]
    back_ptr = None

    for prev_state in states:
        transition_prob = viterbi[-1][prev_state]["prob"] * get_smoothed_transition(prev_state, END_MARKER)

        if transition_prob > max_transition_prob:
            max_transition_prob = transition_prob
            back_ptr = prev_state
            
    tags.append(prev_state)
    current = prev_state

    # Follow the backtrack till the first observation
    for t in range(len(viterbi) - 1, 0, -1):
        tags.insert(0, viterbi[t][current]["back_ptr"])
        current = viterbi[t][current]["back_ptr"]

    # TODO: take log of probabilities, when generating them in buildtagger
    # Instead of P1 * P2, do log(P1) + log(P2)

    return ' '.join([word + '/' + tag for word, tag in zip(obs, tags)])

def load_model(model_file):
    global transition
    global emission
    global word_to_tag
    global tag_freq
    global tags

    global transition_backoff
    global emission_backoff
    global transition_singleton
    global emission_singleton
    global transition_one_count
    global emission_smoothed
    global bitag_freq
    global num_tokens
    
    f = open(model_file, "rb")
    dictionaries = pickle.load(f)

    transition = dictionaries["transition"]
    emission = dictionaries["emission"]
    word_to_tag = dictionaries["word_to_tag"]
    tag_freq = dictionaries["tag_freq"]
    tags = dictionaries["tags"]

    """ New probabilities """
    transition_backoff = dictionaries["transition_backoff"]
    emission_backoff = dictionaries["emission_backoff"]
    transition_singleton = dictionaries["transition_singleton"]
    emission_singleton = dictionaries["emission_singleton"]
    transition_one_count = dictionaries["transition_smoothed"]
    emission_smoothed = dictionaries["emission_smoothed"]
    bitag_freq = dictionaries["bitag_freq"]
    num_tokens = dictionaries["num_tokens"]

    # with open(model_file) as json_file:
    #     model = json.load(json_file)
    
    # [transition_json, emission_json, tags] = model

    # # TODO: store keys as string instead of tuple?
    # emission = defaultdict(lambda:0.000000001)
    # for k,v in emission_json.items():
    #     [word, tag] = k.rsplit(':', 1)
    #     emission[(word,tag)] = v

    # transition = defaultdict(lambda:0.000000001)
    # for k,v in transition_json.items():
    #     [word, tag] = k.rsplit(':', 1)
    #     transition[(word,tag)] = v

    # return transition, emission, tags

def print_probs(transition, emission):
    print("\ntransition:")
    pprint.pprint(transition)
    print("\nemission:")
    pprint.pprint(emission)

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
