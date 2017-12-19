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
num_epochs = 200
# lstm隐藏单元数
num_hidden = 40
# 2层lstm网络
num_layers = 1
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
  '''
  将label文本转换成整数序列，然后再换成稀疏三元组
  '''
  target_filename = 'label.txt'
  
  with open(target_filename, 'r') as f:
    #原始文本为“she had your dark suit in greasy wash water all year”
    line = f.readlines()[0].strip()
    targets = line.replace(' ', '  ')
    # 放入list中，空格用''代替
    #['she', '', 'had', '', 'your', '', 'dark', '', 'suit', '', 'in', '', 'greasy', '', 'wash', '', 'water', '', 'all', '', 'year']
    targets = targets.split(' ')
    
    # 每个字母作为一个label,转换成如下：
    #['s' 'h' 'e' '<space>' 'h' 'a' 'd' '<space>' 'y' 'o' 'u' 'r' '<space>' 'd'
    # 'a' 'r' 'k' '<space>' 's' 'u' 'i' 't' '<space>' 'i' 'n' '<space>' 'g' 'r'
    # 'e' 'a' 's' 'y' '<space>' 'w' 'a' 's' 'h' '<space>' 'w' 'a' 't' 'e' 'r'
    #'<space>' 'a' 'l' 'l' '<space>' 'y' 'e' 'a' 'r']
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

    # 将label转换成整数序列表示:
    # [19  8  5  0  8  1  4  0 25 15 21 18  0  4  1 18 11  0 19 21  9 20  0  9 14
    # 0  7 18  5  1 19 25  0 23  1 19  8  0 23  1 20  5 18  0  1 12 12  0 25  5
    # 1 18]
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                      for x in targets])

    # 将列表转换成稀疏三元组
    train_targets = sparse_tuple_from([targets])
  return train_targets



def inference(inputs, seq_len):
  '''
  2层双向LSTM的网络结构定义
  
  Args：
  inputs： 输入数据，形状是[batch_size, 序列最大长度，一帧特征的个数13]
        序列最大长度是指，一个样本在转成特征矩阵之后保存在一个矩阵中，
		在n个样本组成的batch中，因为不同的样本的序列长度不一样，在组成的3维数据中，
		第2维的长度要足够容纳下所有的样本的特征序列长度。
  seq_len: batch里每个样本的有效的序列长度
  '''
  
  #定义一个向前计算的LSTM单元，40个隐藏单元
  cell_fw = tf.contrib.rnn.LSTMCell(num_hidden, 
                        initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=0.1),
                        state_is_tuple=True)

  # 组成一个有2个cell的list
  cells_fw = [cell_fw] * num_layers
  # 定义一个向后计算的LSTM单元，40个隐藏单元
  cell_bw = tf.contrib.rnn.LSTMCell(num_hidden, 
                        initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=0.1),
                        state_is_tuple=True)
  # 组成一个有2个cell的list
  cells_bw = [cell_bw] * num_layers

  # 将前面定义向前计算和向后计算的2个cell的list组成双向lstm网络
  # sequence_length为实际有效的长度，大小为batch_size，
  # 相当于表示batch中每个样本的实际有用的序列长度有多长。
  # 输出的outputs宽度是隐藏单元的个数，即num_hidden的大小
  outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                 cells_bw,
                                                                 inputs,
                                                               dtype=tf.float32,
                                                        sequence_length=seq_len)

  #获得输入数据的形状
  shape = tf.shape(inputs)
  batch_s, max_timesteps = shape[0], shape[1]

  # 将2层LSTM的输出转换成宽度为40的矩阵
  # 后面进行全连接计算
  outputs = tf.reshape(outputs, [-1, num_hidden])

  W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
  
  b = tf.Variable(tf.constant(0., shape=[num_classes]))

  # 进行全连接线性计算
  logits = tf.matmul(outputs, W) + b

  # 将全连接计算的结果，由宽度40变成宽度80，
  # 即最后的输入给CTC的数据宽度必须是26+2的宽度
  logits = tf.reshape(logits, [batch_s, -1, num_classes])

  # 转置，将第一维和第二维交换。
  # 变成序列的长度放第一维，batch_size放第二维。
  # 也是为了适应Tensorflow的CTC的输入格式
  logits = tf.transpose(logits, (1, 0, 2))
  
  return logits

def main():
  # 输入特征数据，形状为：[batch_size, 序列长度，一帧特征数]
  inputs = tf.placeholder(tf.float32, [None, None, num_features])

  # 输入数据的label，定义成稀疏sparse_placeholder会生成稀疏的tensor：SparseTensor
  # 这个结构可以直接输入给ctc求loss
  targets = tf.sparse_placeholder(tf.int32)

  # 序列的长度，大小是[batch_size]大小
  # 表示的是batch中每个样本的有效序列长度是多少
  seq_len = tf.placeholder(tf.int32, [None])

  # 向前计算网络，定义网络结构，输入是特征数据，输出提供给ctc计算损失值。
  logits = inference(inputs, seq_len)
   
  # ctc计算损失
  # 参数targets必须是一个值为int32的稀疏tensor的结构：tf.SparseTensor
  # 参数logits是前面lstm网络的输出
  # 参数seq_len是这个batch的样本中，每个样本的序列长度。
  loss = tf.nn.ctc_loss(targets, logits, seq_len)
  
  # 计算损失的平均值
  cost = tf.reduce_mean(loss)

  # 采用冲量优化方法
  optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)

  # 还有另外一个ctc的函数：tf.contrib.ctc.ctc_beam_search_decoder
  # 本函数会得到更好的结果，但是效果比ctc_beam_search_decoder低
  # 返回的结果中，decode是ctc解码的结果，即输入的数据解码出结果序列是什么
  decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

  # 采用计算编辑距离的方式计算，计算decode后结果的错误率。
  ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as session:
    # 初始化变量
    tf.global_variables_initializer().run()

    for curr_epoch in range(num_epochs):
      train_cost = train_ler = 0
      start = time.time()

      for batch in range(num_batches_per_epoch):
        #获取训练数据，本例中只去一个样本的训练数据
        train_inputs, train_seq_len = get_audio_feature()
        # 获取这个样本的label
        train_targets = get_audio_label()
        feed = {inputs: train_inputs,
                  targets: train_targets,
                  seq_len: train_seq_len}

        # 一次训练，更新参数
        batch_cost, _ = session.run([cost, optimizer], feed)
        # 计算累加的训练的损失值
        train_cost += batch_cost * batch_size
        # 计算训练集的错误率
        train_ler += session.run(ler, feed_dict=feed)*batch_size

      train_cost /= num_examples
      train_ler /= num_examples

      # 打印每一轮迭代的损失值，错误率
      log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
      print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                          time.time() - start))
    # 在进行了1200次训练之后，计算一次实际的测试，并且输出
    # 读取测试数据，这里读取的和训练数据的同一个样本
    test_inputs, test_seq_len = get_audio_feature()
    test_targets = get_audio_label()
    test_feed = {inputs: test_inputs,
                  targets: test_targets,
                  seq_len: test_seq_len}
    d = session.run(decoded[0], feed_dict=test_feed)
    # 将得到的测试语音经过ctc解码后的整数序列转换成字母
    str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
    # 将no label转换成空
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    # 将空白转换成空格
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    # 打印最后的结果
    print('Decoded:\n%s' % str_decoded)

if __name__ == "__main__":
  main()
