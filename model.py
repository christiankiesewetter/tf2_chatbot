# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.seq2seq import BasicDecoder, AttentionWrapper, BahdanauAttention
from tensorflow_addons.seq2seq.sampler import TrainingSampler, GreedyEmbeddingSampler
from tensorflow.keras.layers import Input, Dense, LSTM, LSTMCell, Embedding

###################
# Another very nice explanation on what is being done here:
# https://medium.com/@dhirensk/tensorflow-addons-seq2seq-example-using-attention-and-beam-search-9f463b58bc6b
###################


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dims, encoder_units,
        batch_size, dropout = 0.0):

        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = encoder_units
        self.embedding = Embedding(vocab_size, embedding_dims)
        self.lstm_layer = LSTM(self.enc_units,
                               return_sequences=True,
                               return_state=True,
                               dropout = dropout,
                               recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x, initial_state = hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_size, self.enc_units)),
                tf.zeros((self.batch_size, self.enc_units))]



class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dims, decoder_units,
        batch_size, max_length_input, max_length_output):

        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = decoder_units
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

        self.embedding = Embedding(vocab_size, embedding_dims)
        self.fc = Dense(vocab_size)
        self.decoder_rnn_cell = LSTMCell(self.dec_units)


        self.sampler = TrainingSampler() # Enables Teacher Forcing.
        
        self.attn = BahdanauAttention(
                        units = self.dec_units,
                        memory = None,
                        memory_sequence_length = self.batch_size * [self.max_length_input])

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = AttentionWrapper(
                            self.decoder_rnn_cell,
                            self.attn,
                            attention_layer_size = self.dec_units)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = BasicDecoder(
                        self.rnn_cell,
                        sampler = self.sampler,
                        output_layer = self.fc)



    def build_initial_state(self, batch_size, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(
                                    batch_size=batch_size, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(
                                    cell_state=encoder_state)
        return decoder_initial_state



    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x,
                            initial_state = initial_state,
                            sequence_length = self.batch_size * [self.max_length_output-1])
        return outputs
