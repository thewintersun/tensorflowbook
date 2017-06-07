#coding=utf-8
import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

from six.moves import xrange as range

try:
    from python_speech_features import mfcc
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError



# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
num_features = 13
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 200
num_hidden = 50
num_layers = 1
batch_size = 1
initial_learning_rate = 1e-2
momentum = 0.9


num_examples = 1
num_batches_per_epoch = int(num_examples/batch_size)

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def get_audio_feature():
  audio_filename = "LDC93S1.wav"
  fs, audio = wav.read(audio_filename)
  inputs = mfcc(audio, samplerate=fs)
  # Tranform in 3D array
  feature_inputs = np.asarray(inputs[np.newaxis, :])
  feature_inputs = (feature_inputs - np.mean(feature_inputs))/np.std(feature_inputs)
  feature_seq_len = [feature_inputs.shape[1]]
  
  return feature_inputs, feature_seq_len
  
def get_audio_label():
  target_filename = 'label.txt'
  
  with open(target_filename, 'r') as f:
    #she had your dark suit in greasy wash water all year
    line = f.readlines()[0].strip()
    targets = line.replace(' ', '  ')
    #['she', '', 'had', '', 'your', '', 'dark', '', 'suit', '', 'in', '', 'greasy', '', 'wash', '', 'water', '', 'all', '', 'year']
    targets = targets.split(' ')
    

    #['s' 'h' 'e' '<space>' 'h' 'a' 'd' '<space>' 'y' 'o' 'u' 'r' '<space>' 'd'
    # 'a' 'r' 'k' '<space>' 's' 'u' 'i' 't' '<space>' 'i' 'n' '<space>' 'g' 'r'
    # 'e' 'a' 's' 'y' '<space>' 'w' 'a' 's' 'h' '<space>' 'w' 'a' 't' 'e' 'r'
    #'<space>' 'a' 'l' 'l' '<space>' 'y' 'e' 'a' 'r']
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

    # [19  8  5  0  8  1  4  0 25 15 21 18  0  4  1 18 11  0 19 21  9 20  0  9 14
    # 0  7 18  5  1 19 25  0 23  1 19  8  0 23  1 20  5 18  0  1 12 12  0 25  5
    # 1 18]
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                      for x in targets])

    # Creating sparse representation to feed the placeholder
    train_targets = sparse_tuple_from([targets])
  return train_targets



def inference(inputs, seq_len):
  cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
  stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)
  outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
  shape = tf.shape(inputs)
  batch_s, max_timesteps = shape[0], shape[1]

  # Reshaping to apply the same weights over the timesteps
  outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
  W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
  # Zero initialization
  # Tip: Is tf.zeros_initializer the same?
  b = tf.Variable(tf.constant(0., shape=[num_classes]))

  # Doing the affine projection
  logits = tf.matmul(outputs, W) + b

  # Reshaping back to the original shape
  logits = tf.reshape(logits, [batch_s, -1, num_classes])

  # Time major
  logits = tf.transpose(logits, (1, 0, 2))
  return logits

def main():
  # e.g: log filter bank or MFCC features
  # Has size [batch_size, max_stepsize, num_features], but the
  # batch_size and max_stepsize can vary along each step
  inputs = tf.placeholder(tf.float32, [None, None, num_features])

  # Here we use sparse_placeholder that will generate a
  # SparseTensor required by ctc_loss op.
  targets = tf.sparse_placeholder(tf.int32)

  # 1d array of size [batch_size]
  seq_len = tf.placeholder(tf.int32, [None])

  logits = inference(inputs, seq_len)
    
  loss = tf.nn.ctc_loss(targets, logits, seq_len)
  cost = tf.reduce_mean(loss)

  optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

  # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
  # (it's slower but you'll get better results)
  decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

  # Inaccuracy: label error rate
  ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

  with tf.Session() as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()


    for curr_epoch in range(num_epochs):
      train_cost = train_ler = 0
      start = time.time()

      for batch in range(num_batches_per_epoch):
        train_inputs, train_seq_len = get_audio_feature()
        train_targets = get_audio_label()
        feed = {inputs: train_inputs,
                  targets: train_targets,
                  seq_len: train_seq_len}

        batch_cost, _ = session.run([cost, optimizer], feed)
        train_cost += batch_cost*batch_size
        train_ler += session.run(ler, feed_dict=feed)*batch_size

      train_cost /= num_examples
      train_ler /= num_examples

      log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
      print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                          time.time() - start))
    # Decoding
    test_inputs, test_seq_len = get_audio_feature()
    test_targets = get_audio_label()
    test_feed = {inputs: test_inputs,
                  targets: test_targets,
                  seq_len: test_seq_len}
    d = session.run(decoded[0], feed_dict=test_feed)
    str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

    print('Decoded:\n%s' % str_decoded)

if __name__ == "__main__":
  main()