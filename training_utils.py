# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import tensorflow_addons as tfa


@tf.function
def train_step(encoder, decoder, input_sequence, target_sequence,
            encoder_hidden_state, optimizer, batch_size, clip):

    loss = 0
    with tf.GradientTape() as tape:

        encoder_output, encoder_hidden_state, encoder_context_state = \
                encoder(input_sequence, encoder_hidden_state)

        decoder.attn.setup_memory(encoder_output)

        decoder_input, decoder_target = \
                target_sequence[:,:-1], target_sequence[:,1:]

        decoder_initial_state = decoder.build_initial_state(
                                    batch_size,
                                    [encoder_hidden_state, encoder_context_state],
                                    tf.float32)

        prediction = decoder(decoder_input, decoder_initial_state)

        logits = prediction.rnn_output

        loss = tfa.seq2seq.sequence_loss(
                logits, decoder_target,
                tf.ones([batch_size, decoder_target.shape[1]]))

    # Clip Gradients and BackProp
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    gradients = [(tf.clip_by_value(grad, -clip, clip)) for grad in gradients]
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


def validation_step(encoder, decoder, input_sequence, target_sequence,
            encoder_hidden_state, optimizer, batch_size):

    loss = 0

    encoder_output, encoder_hidden_state, encoder_context_state = \
            encoder(input_sequence, encoder_hidden_state)

    decoder.attn.setup_memory(encoder_output)

    decoder_input, decoder_target = \
            target_sequence[:,:-1], target_sequence[:,1:]

    decoder_initial_state = decoder.build_initial_state(
                                batch_size,
                                [encoder_hidden_state, encoder_context_state],
                                tf.float32)

    prediction = decoder(decoder_input, decoder_initial_state)

    logits = prediction.rnn_output

    loss = tfa.seq2seq.sequence_loss(
            logits, decoder_target,
            tf.ones([batch_size, decoder_target.shape[1]]))

    return loss
