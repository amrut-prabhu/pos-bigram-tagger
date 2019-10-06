# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.

    """
    Transition probabilities: a_ij
    Probability of transitioning from state s_i to state s_j (states represent POS tags)
    P(tag J | tag I)
    """
    transition = {}

    """
    Emission probabilities: b_i(o_t)
    Probability of observing an observation o_t from state s_i (observations are words)
    P(word | tag)
    """
    emission = {}

    """
    Viterbi probabilities: v(s, t)
    Maximum probability of all paths ending in state s_j at time t
    v(i, t) = max[ v(j, t-1) * a_ji * b_i(o_t) ]   for j=1 to N
    """
    viterbi = []
    
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
