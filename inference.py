# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import pickle
from model import Encoder, Decoder
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_addons.seq2seq import BasicDecoder, AttentionWrapper, BahdanauAttention
from tensorflow_addons.seq2seq.sampler import TrainingSampler
from tensorflow.keras.optimizers import Adam



class InferenceModel:
    def __init__(self,
                checkpoint_dir,
                vocab_size,
                embedding_dim,
                units,
                input_length,
                output_length,
                batch_size):

        self.batch_size = batch_size
        self.units = units
        self.input_length = input_length
        self.output_length = output_length

        with open(os.path.join(checkpoint_dir, 'tokenizer.pickle'), 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        self.encoder = Encoder(
                    vocab_size = vocab_size,
                    embedding_dims = embedding_dim,
                    encoder_units = units,
                    batch_size = batch_size)

        self.decoder = Decoder(
                    vocab_size = vocab_size,
                    embedding_dims = embedding_dim,
                    decoder_units = units,
                    batch_size = batch_size,
                    max_length_input = input_length,
                    max_length_output = output_length)

        checkpoint = tf.train.Checkpoint(encoder = self.encoder,
                                         decoder = self.decoder)
        checkpoint.restore(
            tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(
                                self.decoder.embedding)

        # Instantiate BasicDecoder object
        self.decoder_instance = tfa.seq2seq.BasicDecoder(
                                        cell = self.decoder.rnn_cell,
                                        sampler = greedy_sampler,
                                        output_layer = self.decoder.fc,
                                        maximum_iterations = 160)


    def __call__(self, input_sequence):
        input_sequence = '<sos> ' + input_sequence + ' <eos>'
        input_sequence = self.tokenizer.texts_to_sequences([input_sequence])

        input_sequence = tf.constant(input_sequence)
        enc_start_state = self.encoder.initialize_hidden_state()

        enc_out, enc_h, enc_c, _, _ = self.encoder(input_sequence, enc_start_state)

        dec_h = enc_h
        dec_c = enc_c

        start_tokens = tf.constant([self.tokenizer.word_index['<sos>']])
        end_token = self.tokenizer.word_index['<eos>']

        # Setup Memory in decoder stack
        self.decoder.attn.setup_memory(enc_out)

        # set decoder_initial_state
        decoder_initial_state = self.decoder.build_initial_state(
                                            self.batch_size,
                                            [dec_h, dec_c],
                                            tf.float32)

        outputs, state, lengths = self.decoder_instance(
                            None,
                            start_tokens = start_tokens,
                            end_token = end_token,
                            initial_state = decoder_initial_state)

        phrase = " ".join([self.tokenizer.index_word[o] for o in outputs.sample_id.numpy()[0] if o != 0])

        return phrase, outputs, state, lengths


if __name__ == '__main__':

    inference = InferenceModel(
                checkpoint_dir = './training_checkpoints',
                vocab_size = 3000,
                embedding_dim = 512,
                units = 512,
                input_length = 382,
                output_length = 686,
                batch_size = 1)

    keep_talking = True

    while keep_talking:
        text = input("Your turn:")
        phrase, _, _, _ = inference(text.lower())
        print(phrase)
        keep_talking = (text.lower() != 'enough')
