# -*- coding: utf-8 -*-
import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from preprocessing2 import dataset
from model import Encoder, Decoder
from training_utils import train_step, validation_step

np.random.seed(42) # What's the Answer?
###################################################################

# Data #
CHECKPOINT_DIR = './training_checkpoints'
BASE_PATH = "./"
DATASET_PATH = "../Datasets/cornell movie-dialogs corpus"

# Training #
EPOCHS = 100
VALIDATION_SPLIT = .15

# Model Hyperparameters #
BATCH_SIZE = 16
VOCAB_SIZE = 3500
EMBEDDING_DIM_ENCODER = 512
EMBEDDING_DIM_DECODER = 512
UNITS_ENCODER = 512
UNITS_DECODER = 512

# Optimizer and Learning Rate
LEARNING_RATE = 1e-04
LEARNING_RATE_DECAY = 0.8
LEARNING_EPOCH_DECAY = 2
MIN_LEARNING_RATE = 1e-07

# Dropout Rate
DROPOUT_RATE = 0.2

# CUSTOMIZE
NUM_LINES = None
WARMUP = 5
TRAIN_STEPS_PER_EPOCH = None
VALID_STEPS_PER_EPOCH = None

##################################################################

if __name__ == '__main__':
    '''
    Most of the tf.contrib packages have either been moved into tf core or
    into the tensorflow-addons packages

    To recreate the Workflow as described with tf1.x we thus have to replace the
    given parts with the new packages or implementations
    '''
    print('Start Preprocessing', flush=True)
    # Generate and initialize training set
    movie_conversations_file = os.path.join(DATASET_PATH, 'movie_conversations.txt')
    movie_lines_file = os.path.join(DATASET_PATH, 'movie_lines.txt')

    # The preprocessing is basically doing the same thing as in the
    # explained example. The only difference here is that we're using
    # some native Keras libraries (Tokenizer) to create the Dataset, which
    # makes is a bit easier to understand.
    ##### SEE : preprocessing.py
    questions, answers, tokenizer = dataset(
                                    conversations_file = movie_conversations_file,
                                    conversation_lines = movie_lines_file,
                                    num_words = VOCAB_SIZE,
                                    num_lines = NUM_LINES)


    # saving
    if not os.path.isdir(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    with open(os.path.join(CHECKPOINT_DIR, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol = pickle.DEFAULT_PROTOCOL)

    print('Tokenizer created...', flush=True)

    # Create Randomized Validation Split
    length_entries = questions.shape[0]

    train_steps_per_epoch = int((length_entries * (1-VALIDATION_SPLIT)) / BATCH_SIZE)
    valid_steps_per_epoch = int((length_entries * VALIDATION_SPLIT) / BATCH_SIZE)

    # If customized
    if TRAIN_STEPS_PER_EPOCH is not None:
        train_steps_per_epoch = TRAIN_STEPS_PER_EPOCH

    if VALID_STEPS_PER_EPOCH is not None:
        valid_steps_per_epoch = VALID_STEPS_PER_EPOCH


    # pick from random numbers 15%
    entry_idx = np.arange(0,length_entries)
    np.random.shuffle(entry_idx)

    val_idx = entry_idx[:int(length_entries * VALIDATION_SPLIT)]
    train_idx = entry_idx[int(length_entries * VALIDATION_SPLIT):]

    print('Creating Keras Datasets', flush=True)

    ds_training = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(questions[train_idx]),
            tf.data.Dataset.from_tensor_slices(answers[train_idx]))).batch(
                                BATCH_SIZE, drop_remainder = True)

    ds_validation = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(questions[val_idx]),
            tf.data.Dataset.from_tensor_slices(answers[val_idx]))).batch(
                                BATCH_SIZE, drop_remainder = True)


    # Generate Models
    ##### SEE : model.py
    encoder = Encoder(
                vocab_size = VOCAB_SIZE,
                embedding_dims = EMBEDDING_DIM_ENCODER,
                encoder_units = UNITS_ENCODER,
                batch_size = BATCH_SIZE,
                dropout = DROPOUT_RATE)


    decoder = Decoder(
                vocab_size = VOCAB_SIZE,
                embedding_dims = EMBEDDING_DIM_DECODER,
                decoder_units = UNITS_DECODER,
                batch_size = BATCH_SIZE,
                max_length_input = questions.shape[1],
                max_length_output = answers.shape[1])


    # Setting up the learning rate with decay and minimal learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = LEARNING_RATE,
        decay_steps = train_steps_per_epoch * LEARNING_EPOCH_DECAY, # reduce after 'LEARNING_EPOCH_DECAY' epochs!
        decay_rate = LEARNING_RATE_DECAY,
        staircase = True)


    # Feed the Optimizer with this custom decay logic (we could've used Adam
    # directly but I'm not sure in how far it accepts minimum learning rates as well)

    optimizer = Adam(learning_rate = lr_schedule)

    # We have to checkpoint all three elements to restore the model later on
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer = optimizer,
                                     encoder = encoder,
                                     decoder = decoder)


    print(f'Begin Training... {length_entries} Entries, divided into {train_steps_per_epoch} Steps per Training Epoch', flush=True)
    print(f'Sequence Length Input/Output {questions.shape[1]}/{answers.shape[1]}', flush=True)


    ### TRAINING
    min_epoch_loss = 1_000 # Fake it till you make it ;)
    for epoch in range(EPOCHS):
        avg_train_loss = 0
        start = time.time()
        encoder_hidden_state = encoder.initialize_hidden_state()

        for (batch_index, (in_seq, out_seq)) in enumerate(ds_training.take(train_steps_per_epoch)):

            batch_time = time.time()
            batch_loss = train_step(
                            encoder = encoder,
                            decoder = decoder,
                            input_sequence = in_seq,
                            target_sequence = out_seq,
                            encoder_hidden_state = encoder_hidden_state,
                            optimizer = optimizer,
                            batch_size = BATCH_SIZE,
                            clip = 5.)

            avg_train_loss += batch_loss / train_steps_per_epoch

            if batch_index % 500 == 0:
                print(f'Epoch {epoch + 1}/{EPOCHS} - Step {batch_index + 1}/{train_steps_per_epoch} - {(time.time() - batch_time):.1f}s', flush=True)
                print(f'Loss: {batch_loss.numpy():.4f} - Next Learning Rate: {optimizer._decayed_lr(tf.float32):.8f}', flush=True)

        print(f'Training:\tEpoch {epoch + 1}/{EPOCHS} AVGLoss: {avg_train_loss:.4f} - {(time.time() - start)/60:.1f} min', flush=True)

        ### Validation ###
        avg_valid_loss = 0
        val_encoder_hidden_state = encoder.initialize_hidden_state()
        for (batch_index, (in_seq, out_seq)) in enumerate(ds_training.take(valid_steps_per_epoch)):
            validation_batch_loss = validation_step(
                                        encoder = encoder,
                                        decoder = decoder,
                                        input_sequence = in_seq,
                                        target_sequence = out_seq,
                                        encoder_hidden_state = val_encoder_hidden_state,
                                        optimizer = optimizer,
                                        batch_size = BATCH_SIZE)

            avg_valid_loss += validation_batch_loss / valid_steps_per_epoch

        print(f'Validation:\tEpoch {epoch + 1}/{EPOCHS} AVGLoss: {avg_valid_loss:.4f}', flush=True)

        if min_epoch_loss > avg_valid_loss and epoch > WARMUP:
            min_epoch_loss = avg_valid_loss
            checkpoint.save(file_prefix = checkpoint_prefix)
