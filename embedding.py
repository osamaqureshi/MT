import numpy as np
import pandas as pd
import re, string, nltk, spacy
import os, sys, csv, random, time
from collections import Counter
from pickle import dump, load
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def tokenize(corpus):
  corpus = ['<start> '+line+' <end>' for line in corpus]
  tokenizer = keras.preprocessing.text.Tokenizer(filters='')
  tokenizer.fit_on_texts(corpus)
  tensor = tokenizer.texts_to_sequences(corpus)
  tensor = keras.preprocessing.sequence.pad_sequences(tensor,  padding='post')
  return tensor, tokenizer

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)
    return x, state, attention_weights
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)
    for t in range(1, targ.shape[1]):
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions)
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss

if __name__== "__main__" :

	en = load(open('Europarl/en.pkl', 'rb'))
	fr = load(open('Europarl/fr.pkl', 'rb'))

	input_tensor, inp_lang = tokenize(en)
	target_tensor, targ_lang = tokenize(fr)

	buffer_size = input_tensor.shape[0]
	batch_size = 64
	steps_per_epoch = len(X_train)//batch_size
	embedding_dim = 256
	units = 1024
	vocab_inp_size = len(inp_lang.word_index)
	vocab_tar_size = len(targ_lang.word_index)
	max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]


	x_train, x_test, y_train, y_test = train_test_split(input_tensor, target_tensor, test_size=0.2, random_state=42)
	dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size)
	dataset = dataset.batch(batch_size, drop_remainder=True)
	test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

	encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
	decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
	learning_rate = CustomSchedule(units)
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

	checkpoint_dir = 'Checkpoints/Attention/embedding'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

	%%time
	EPOCHS = 100
	  
	for epoch in range(EPOCHS):
	  start = time.time()
	  enc_hidden = encoder.initialize_hidden_state()
	  total_loss = 0

	  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
	    batch_loss = train_step(inp, targ, enc_hidden)
	    total_loss += batch_loss

	    if batch % 100 == 0:
	    	print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
	  if (epoch + 1) % 5 == 0:
	    checkpoint.save(file_prefix = checkpoint_prefix)

	  print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
	  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


	df = []
	with open('Predictions/Attention/embeddings/predictions.txt','a') as f:
		for i, (x,y) in enumerate(zip(x_test, y_test)):
		  sentence = inp_lang.sequences_to_texts([x])[0]
		  pred, _ = translate(sentence)
		  y = targ_lang.sequences_to_texts([y])
		  df += [[y, pred]]

		  if i % 1000 == 0: print(i)
		  	print('Saving ',i)
		    for line in df:
		    	f.write(line[0]+'\t'+line[1]+'\n')
		    df = []