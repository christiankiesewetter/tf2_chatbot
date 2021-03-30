# -*- coding: utf-8 -*-
import os
import re
import sys, traceback
from itertools import chain
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

FORMAT = 'WINDOWS-1252'

def preformat(line):
    return re.sub(r'([!?.:;])',r' \1',
            line.replace('\'s',' is')
                .replace('\'re',' are')
                .replace('\'m',' am')
                .replace('n\'t',' not')
                .replace('\'ll',' will')
                .replace('\'ve',' have')
                .replace('\'d',' would')
                .replace('\n', '').lower())



def read_txt_file(filepath):
    res = []
    with open(filepath, mode = 'r', encoding="WINDOWS 1252") as f:
        for line in f.readlines():
            try:
                result = preformat(line)
                res.append('<SOS> ' + result + ' <EOS>')
            except Exception:
                traceback.print_exc()
    return res


def dataset(question_file, answer_file, num_words, num_lines=None):
    questions = read_txt_file(filepath = question_file)
    answers = read_txt_file(filepath = answer_file)

    if None is not num_lines:
        questions = questions[:num_lines]
        answers = answers[:num_lines]

    alltexts = questions + answers

    tokenizer = Tokenizer(num_words=num_words, filters='"#$%&()*+,-/:;@[\\]^_`{|}~\t\n',)
    tokenizer.fit_on_texts(alltexts)

    questions = tokenizer.texts_to_sequences(questions)
    questions = pad_sequences(questions, padding='post')

    answers = tokenizer.texts_to_sequences(answers)
    answers = pad_sequences(answers, padding='post')

    return questions ,answers, tokenizer


if __name__ == "__main__":
    question_file = 'dataset/movie_questions_2.txt'
    answer_file = 'dataset/movie_answers_2.txt'
    questions, answers, tokenizer = dataset( question_file = question_file, answer_file = answer_file, num_words = 2000)
