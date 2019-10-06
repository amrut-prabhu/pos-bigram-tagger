# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np

from collections import defaultdict
import pprint
import json

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.

    transition, emission = load_model(model_file)

    """
    Viterbi probabilities: v(s, t)
    Maximum probability of all paths ending in state s_j at time t
    v(i, t) = max{ v(j, t-1) * a_ji * b_i(o_t) }   for j=1 to N
    """
    viterbi = []
    
    print('Finished...')

def load_model(model_file):
    with open(model_file) as json_file:
        model = json.load(json_file)
    
    [transition_json, emission_json] = model

    # TODO: store keys as string instead of tuple?
    emission = defaultdict(int)
    for k,v in emission_json.items():
        [word, tag] = k.rsplit(':', 1)
        emission[(word,tag)] = v

    transition = defaultdict(int)
    for k,v in transition_json.items():
        [word, tag] = k.rsplit(':', 1)
        transition[(word,tag)] = v

    return transition_json, emission_json

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
