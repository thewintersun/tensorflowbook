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



# 常量
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# mfcc默认提取出来的一帧13个特征
num_features = 13
# 26个英文字母 + 1个空白 + 1个no label = 28 label个数
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# 迭代次数
num_epochs = 1000
# lstm隐藏单元数
num_hidden = 40
# 2层lstm网络
num_layers = 2
# batch_size设置为1
batch_size = 1
# 初始学习率
initial_learning_rate = 0.01

# 样本个数
num_examples = 1
# 一个epoch有多少个batch
num_batches_per_epoch = int(num_examples/batch_size)


def sparse_tuple_from(sequences, dtype=np.int32):
    """得到一个list的稀疏表示，为了直接将数据赋值给tensorflow的tf.sparse_placeholder稀疏矩阵
    Args:
        sequences: 序列的列表
    Returns:
        一个三元组，和tensorflow的tf.sparse_placeholder同结构
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
  '''
  获取wav文件提取mfcc特征之后的数据
  '''
  
  audio_filename = "audio.wav"
  
  #读取wav文件内容，fs为采样率， audio为数据
  fs, audio = wav.read(audio_filename)
  
  #提取mfcc特征
  inputs = mfcc(audio, samplerate=fs)
  # 对特征数据进行归一化，减去均值除以方差
  feature_inputs = np.asarray(inputs[np.newaxis, :])
  feature_inputs = (feature_inputs - np.mean(feature_inputs))/np.std(feature_inputs)
  
  #特征数据的序列长度
  feature_seq_len = [feature_inputs.shape[1]]
  
  return feature_inputs, feature_seq_len
  
def get_audio_label():
  # 对应wav文件内容的文本内容
  target_filename = 'label.txt'
  
  with open(target_filename, 'r') as f:
    #文本的内容是“she had your dark suit in greasy wash water all year”
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
  
  #采用2层双向LSTM
  cell_fw = tf.contrib.rnn.LSTMCell(num_hidden, 
                        initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=0.1),
                        state_is_tuple=True)
  cells_fw = [cell_fw] * num_layers

  cell_bw = tf.contrib.rnn.LSTMCell(num_hidden, 
                        initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=0.1),
                        state_is_tuple=True)
  cells_bw = [cell_bw] * num_layers

  outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                 cells_bw,
                                                                 inputs,
                                                               dtype=tf.float32,
                                                        sequence_length=seq_len)

  '''
  stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)
  outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
  '''

  shape = tf.shape(inputs)
  batch_s, max_timesteps = shape[0], shape[1]

  # Reshaping to apply the same weights over the timesteps
  outputs = tf.reshape(outputs, [-1, num_hidden])

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

  optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)

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
