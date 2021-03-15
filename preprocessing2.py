# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import sys, traceback
from itertools import chain
import numpy as np
import tensorflow as tf
from ast import literal_eval

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

FORMAT = 'WINDOWS 1252'

def preformat(line):
    return '<sos> ' + re.sub(r'([!?.:;])',r' \1',
            line.replace('\'s',' is')
                .replace('\'re',' are')
                .replace('\'m',' am')
                .replace('n\'t',' not')
                .replace('\'ll',' will')
                .replace('\'ve',' have')
                .replace('\'d',' would')
                .replace('\n', '').lower()) + ' <eos>'



def read_txt_file(filepath, splitby = None):
    res = []
    with open(filepath, mode = 'r', encoding=FORMAT) as f:
        for line in f.readlines():
            try:
                line = line.replace('\n','')
                if splitby is not None:
                    line = line.split(splitby)

                res.append(line)
            except Exception:
                traceback.print_exc()
    return res


def dataset(conversations_file, conversation_lines, num_words, num_lines = None):

    dialogues = read_txt_file(conversations_file, splitby= ' +++$+++ ')
    dialogues = [literal_eval(dialog[3]) for dialog in dialogues]

    lines = read_txt_file(conversation_lines, splitby= ' +++$+++ ')
    corpus_dict = {line[0]:line[4] for line in lines}

    inputs_lineidx, outputs_lineidx = zip(*[conversation for dialogue in dialogues for conversation in list(zip(dialogue[:-1], dialogue[1:]))])

    inputs = [preformat(corpus_dict[lineidx]) for lineidx in inputs_lineidx]
    outputs = [preformat(corpus_dict[lineidx]) for lineidx in outputs_lineidx]

    questions, answers = inputs, outputs

    if None is not num_lines:
        questions = questions[:num_lines]
        answers = answers[:num_lines]

    alltexts = questions + answers

    tokenizer = Tokenizer(num_words=num_words, filters='"#$%&()*+,-/@[\\]^_`{|}~\t\n', oov_token='<out>')
    tokenizer.fit_on_texts(alltexts)

    questions = tokenizer.texts_to_sequences(questions)
    questions = pad_sequences(questions, padding='pre', value = 0) # <pad> => 0

    answers = tokenizer.texts_to_sequences(answers)
    answers = pad_sequences(answers, padding='post', value = 0) # <pad> => 0

    return np.array(questions), np.array(answers), tokenizer


if __name__ == "__main__":
    conversations_file = '../../Datasets/cornell movie-dialogs corpus/movie_conversations.txt'
    conversation_lines = '../../Datasets/cornell movie-dialogs corpus/movie_lines.txt'
    questions, answers, tokenizer = dataset(
                                    conversations_file = conversations_file,
                                    conversation_lines = conversation_lines,
                                    num_words = 3_000,
                                    num_lines = 100)
