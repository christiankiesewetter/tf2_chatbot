# TF2.x Re-Implementation of the Chatbot provided by 'Deep Learning A-Z (TM) - How to build a Chatbot'
# Introduction
This repository shows how to re-implement the chatbot provided by Kirill Eremenko's and Hadelin de Ponteves' Deep Learning Course on Udemy. Since I was wishing for a 'contemporary' TF2.X solution, I decided to try and create one myself, so i could put it up on GitHub for everyone.
I - of course - am not a professional in this field, so there might indeed still be some mistakes.
I tried to comment most of the code, so you could simply follow the sources.

My premises were:
* to build the network from scratch only with given tools provided by the TF2.X environment
* use the Keras Tokenizer for preprocessing.
### Training
To train the model, simply adjust the directories and (hyper-)parameters given in the training.py file, for your needs. Execute the training in the console by typing:
<code>python training.py</code>
or simply execute the script in your virtual environment of choice.
There will be a folder created, where the checkpoints are stored.

(!) Be warned, the training might take a long time.

### Inference
To execute the inference, run <code>python inference.py</code>

I provided my last checkpoint together with a pickled version of my tokenizer.
If you don't want to use my pickled tokenizer (which i totally would understand), feel free to execute the training.py code including line 77 with unchanged parameters. If you use the original files for training from the Cornell Movie Database Dialogue Dataset, the Tokenizer should most likely give the same results as the pickled one.

## Changes Compared to Original Implementation
### Parameters
* Steps per Epoch are adjusted so that, every epoch trains on all batches of the training dataset. Therefore there's no static variable for steps per epoch. Feel free to try out less steps per epoch, by changing
* TRAIN_STEPS_PER_EPOCH / VALID_STEPS_PER_EPOCH - steps that are evaluated per epoch - leave None, if you want to use the whole train/validation set
* I reduced the Batch Size, so my GPU wouldn't start barking because of lacking resources (feel free to play around).
* WARMUP - Epochs to wait before first checkpoint saving
* NUM_LINES - number of questions and answers used. Leave as None if you want to use the entire dataset.
* VERBOSE - True: show loss output for every batch, False: show AVG Batch Epoch loss only.
* DROPOUT_RATE - set to 0.2 (again, feel free to play around)
### Preprocessing
* I changed the behaviour of the preprocessing and tried - as an experiment - to keep sentence endings like [!?.:;] in the vocabulary
* It seems that, the original datset txt files are not UTF-8 encoded. At least some characters could not be read by that. I instead tried WINDOWS 1252 encoding, which worked like a charm.


## ToDo:
* [x]  Use Original Dataset as in example
* [x]  Adjust Parameters (BSize, Epochs, Steps per Epoch, )

* [x] Optimize only after Batch
* [x] Teacher Forcing?

* [x] Get Logit Output from Sampler / Dense Layer Decoder
* [x] Make Bidirectional Encoder

* [x] Use Validation Set per Epoch
* [x] Inference

* [x] Interactive Console Input

* [ ] Check the Legal Stuff Disclaimer and Licenses
* [ ] Upload to Repository

* [ ] Get Course Leaders Confirmation that this is no BS :)

* [ ] Use BeamSearch


steps per epoch = entries / batch size
