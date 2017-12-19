#coding=utf-8 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import codecs
import re
import numpy as np
import tensorflow as tf

import random

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 10, "Numbers of training examples each step processes ")
flags.DEFINE_integer("embedding_size", 200, " ")

flags.DEFINE_string("word2vec_path", "./embeding.txt", " ")
flags.DEFINE_string("train_file", "./msr_training.utf8", " ")
flags.DEFINE_string("test_file", "./msr_test_gold.utf8", " ")
flags.DEFINE_string("model_save_path", "./save/", " ")

flags.DEFINE_float("dropout", 0.75, " ")
flags.DEFINE_float("gpu_memory_fraction", 0.1, " ")
FLAGS = flags.FLAGS


INITIAL_LEARNING_RATE = 0.003


class ChineseSegLSTM(object):
  ''' 
  参考复旦的论文: 
  Long Short-Term Memory Neural Networks for chinese word segmentation的中文分词
  '''
  def __init__(self):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    self._word2vec_dict={}
    self._word2vec_num=0
    self._word2vec_size=FLAGS.embedding_size
    
    self._train_list = []
    self._label_list = []

    self._test_list = []
    self._test_label_list = []
    self._test_index_in_epoch = 0      #取test batch的时候记录位置
    self._train_index_in_epoch = 0     #取test train batch的时候记录位置
    
    self._batch_size = FLAGS.batch_size
    self._embedding_size = FLAGS.embedding_size
    self._window_size = 5

    self._init_word2vec_dict(FLAGS.word2vec_path)
    self._max_feature_len = 600

  def run_training(self):
    '''
    训练过程
    '''
    self._read_train_file()
    self._read_test_file()
    print("read file complete")

    global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
    opt = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE)
    input_data = tf.placeholder(tf.float32, [self._batch_size, self._max_feature_len, self._window_size * self._embedding_size])
    labels = tf.placeholder(tf.int32, [self._batch_size, self._max_feature_len])
    seq_len = tf.placeholder(tf.int32, [self._batch_size])
    logits = self.inference( input_data, seq_len)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=labels, logits=logits, name="cross_entropy")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    grads = opt.compute_gradients(cross_entropy_mean)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    train_op = tf.group(apply_gradient_op )
    
    total_correct = 0
    for i in range(self._batch_size):
      per_logits_reshape = logits[i][:seq_len[i]]
      per_labels_reshape = labels[i][:seq_len[i]]
      correct = tf.nn.in_top_k(per_logits_reshape, per_labels_reshape, 1)
      eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
      total_correct += eval_correct
    self.saver = tf.train.Saver()
      
    init = tf.initialize_all_variables()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=gpu_options)) as sess:
     
      sess.run(init)
      print("init variable complete")

      for i in range(10000000):
        train_batch, label_batch, seq_len_list = self.next_train_batch(self._batch_size )
        
 

        start_time = time.time()
        feed_dict ={input_data:train_batch, labels:label_batch, seq_len: seq_len_list}
        _, loss_value = sess.run([train_op,cross_entropy_mean],feed_dict=feed_dict)
        duration = time.time() - start_time
        if i % 50 == 0 and i > 0:
          print("step: %d loss %f" % (i, loss_value))
        if i % 100 == 0 and i > 0:
          test_batch, test_label_batch, test_seq_len_list = self.next_test_batch(self._batch_size  )
          feed_dict ={input_data:test_batch, labels:test_label_batch, seq_len: test_seq_len_list}
          _, loss_value,true_count   = sess.run([train_op,cross_entropy_mean, total_correct],feed_dict=feed_dict)
          total_count = 0
          for k in range(len(test_seq_len_list)):
            total_count += test_seq_len_list[k]
          print("test: %d loss %f correct_ratio %f" % (i, loss_value, true_count/total_count))
        if i % 1000 == 0 and i > 0:
          self.saver.save(sess, os.path.join(FLAGS.model_save_path, "model.ckpt"), global_step=i)
          print("save model at %s, step: %d" %( os.path.join(FLAGS.model_save_path, "model.ckpt"), i))


  def inference(self, input_data, seq_len):
    ''' 向前计算过程描述
    Args:
        layer_number:表示训练的层数， 用来在进行逐层训练的时候指定
    '''
    num_hidden = 120
    num_layers = 2

    w3 = tf.get_variable("w3", [num_hidden*2, 256],
                      initializer=tf.random_normal_initializer(stddev=0.1))
    b3 = tf.get_variable("b3", [256],
                      initializer=tf.constant_initializer(0.0))
    w2 = tf.get_variable("w2", [256, 4],
                      initializer=tf.random_normal_initializer(stddev=0.1))
    b2 = tf.get_variable("b2", [4],
                      initializer=tf.constant_initializer(0.0))

    fw_cell = tf.contrib.rnn.LSTMCell(
        num_hidden, initializer=tf.random_normal_initializer(stddev=0.1))
    bw_cell = tf.contrib.rnn.LSTMCell(
        num_hidden, initializer=tf.random_normal_initializer(stddev=0.1))
    
    fw_cells = [fw_cell] * num_layers
    bw_cells = [bw_cell] * num_layers

    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        fw_cells, bw_cells, input_data, sequence_length=seq_len, dtype=tf.float32
        )
    outputs = tf.reshape(outputs, [-1, num_hidden * 2])

    fc3 = tf.nn.relu(tf.nn.xw_plus_b(outputs, w3, b3))
    #再做一个线性变化
    logits = tf.nn.xw_plus_b(fc3,w2 , b2)
    logits = tf.reshape(logits, [self._batch_size, -1, 4])
    return logits


  def loss(self,logits,placeholder_id ):
     
    labels = tf.cast(self._label_data_holdplace[placeholder_id], tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name="cross_entropy")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')



  def _reset_test_index(self):
    '''将测试数据集合设置从头开始读取'''
    self._test_index_in_epoch = 0


  def _reset_test_train_index(self):
    '''将训练数据集合设置从头开始读取'''
    self._train_index_in_epoch = 0


  def next_train_batch(self, batch_size):
    '''读取测试训练数据本身的数据'''
    start = self._train_index_in_epoch

    context_size = self._window_size // 2
    data_set = []
    label_set = []
    seq_len_list = []

    getsize = 0
    while getsize < batch_size:
      seq_len = 0
      if start >= len(self._train_list):
        start = 0

      char_list = self._train_list[start]

      char_vector_list = []
      char_label_list = []

      for i in range(len(char_list)):
        char = char_list[i]
        if char == "PAD":
          continue

        word_context=[]
        for k in range(-context_size, -context_size + self._window_size):
          char = char_list[i+k]
          word_context.extend(self._get_vector(char))


        char_vector_list.append(word_context)

        char_label_list.append(self._gen_label_value(self._label_list[start][i]))
        seq_len += 1

      
      #补0
      pad_vector = []
      for k in range(self._window_size):
        pad_vector.extend(self._get_vector("PAD"))


      pad_number = self._max_feature_len - seq_len

      char_vector_list = char_vector_list + [pad_vector] * pad_number
      char_label_list = char_label_list + [0] * pad_number
        
      if seq_len > self._max_feature_len:
        start += 1
        continue

      data_set.append(char_vector_list)
      label_set.append(char_label_list)
      seq_len_list.append(seq_len)
      start += 1
      getsize += 1

    self._train_index_in_epoch = start
    return data_set, label_set, seq_len_list


  def next_test_batch(self, batch_size):
    ''' 读取训练数据一个batch ,得到字向量 '''
    start = self._test_index_in_epoch

    context_size = int(self._window_size / 2)
    data_set = []
    label_set = []
    seq_len_list = []

    getsize = 0
    while getsize < batch_size:
      seq_len = 0
      if start >= len(self._test_list):
        start = 0

      char_list = self._test_list[start]

      char_vector_list = []
      char_label_list = []

      for i in range(len(char_list)):
        char = char_list[i]
        if char == "PAD":
          continue

        word_context=[]
        for k in range(-context_size, -context_size + self._window_size):
          char = char_list[i + k]
          word_context.extend(self._get_vector(char))


        char_vector_list.append(word_context)

        char_label_list.append(self._gen_label_value(self._test_label_list[start][i]))

        seq_len += 1


      #补0
      pad_vector = []
      for k in range(self._window_size):
        pad_vector.extend(self._get_vector("0"))

      pad_number = self._max_feature_len - seq_len

      char_vector_list = char_vector_list + [pad_vector] * pad_number
      char_label_list = char_label_list + [0] * pad_number
        
      if seq_len > self._max_feature_len:
        start += 1
        continue

      data_set.append(char_vector_list)
      label_set.append(char_label_list)
      seq_len_list.append(seq_len)
      start += 1
      getsize += 1

    self._test_index_in_epoch = start
    return data_set, label_set, seq_len_list



  def _read_test_file(self):
    ''' 读取一个测试文件， 读取测试数据'''
    converted_file = "./msr_test_gold.convert"
    #self._character_tagging(FLAGS.test_file,converted_file)
    self._test_list, self._test_label_list  = self._load_input_and_label(converted_file)
    

  def _gen_label_value(self,k):
    s = [1,0,0,0]
    b = [0,1,0,0]
    m = [0,0,1,0]
    e = [0,0,0,1]

    if k=='S':
      return 0
    if k=='B':
      return 1
    if k=='M':
      return 2
    if k=='E':
      return 3
    return 0


  def _get_zero_list(self,embedding_size):
    value=[]
    for i in range(embedding_size):
      value.append(0)
    return value
    
    
  def _read_train_file(self):
    ''' 
    读取训练文件， 
    首先根据训练文件中分词的情况，
    将训练文件中每个字做一个S/B/M/E的标记
    然后将数据的正确输入和label保存到内存中
    '''
    converted_file = "./msr_training.convert"
    #self._character_tagging(FLAGS.train_file, converted_file)
    self._train_list, self._label_list = self._load_input_and_label(converted_file)


  def _load_input_and_label(self, file_path):
    '''
    将数据读入内存。
    每一句话的前面用三个PAD填充
    Args:
      file_path: 输入的经过转换的文件
    '''
    all_data_list = []
    all_label_list = []

    with codecs.open(file_path, 'r', 'utf-8') as input_data:
      
      for line in input_data:
        char_list = []
        label_list = []
        
        word_list = line.strip().split()
        
        if len(word_list) ==0:
          continue

        char_list.append("PAD")
        char_list.append("PAD")
        char_list.append("PAD")
        label_list.append("PAD")
        label_list.append("PAD")
        label_list.append("PAD")

        for word in word_list:
          word_array = word.split('/')
          char_list.append(word_array[0])
          label_list.append(word_array[1])
        
        char_list.append("PAD")
        char_list.append("PAD")
        char_list.append("PAD")
        label_list.append("PAD")
        label_list.append("PAD")
        label_list.append("PAD")

        all_data_list.append(char_list)
        all_label_list.append(label_list)

    return all_data_list, all_label_list


  def _init_word2vec_dict(self, word2vec_path):
    '''
    将word2vec训练好的字向量加载到内存
    '''
    if not os.path.exists(word2vec_path):
      print("file %s not exist" % word2vec_path)
      return
    with codecs.open(word2vec_path, "r","utf-8") as f:
      #第一行是大小值，先略过
      isfirstline = True
      for line in f:
        list = line.strip().split(' ')
        if isfirstline == True:
          isfirstline=False
          self._word2vec_num=int(list[0])
          self._word2vec_size=int(list[1])
          continue

        key=list[0]
        value=list[1:]
        value=self.stringlist2floatlist(value)
        self._word2vec_dict[key]=value


  def _get_vector(self,key):
    '''
    通过一个字获取这个字的字向量， 
    如果没有的话， 就随机生成一个， 并且将随机生成的保存
    '''

    if key=="PAD":
      return self._get_zero_list(self._word2vec_size)

    if key not in self._word2vec_dict:
      #如果没有这个key，就生成一个随机的， 放到字典里
      value=[]
      for i in range(self._word2vec_size):
        value.append(random.uniform(-1,1))
      self._word2vec_dict[key]=value
      self._word2vec_num+=1
      return value
    return self._word2vec_dict[key]



  def _character_tagging(self,input_file, output_file):
    ''' 
    对训练数据进行预处理， 
    将每个字根据分词的情况，标记S，B，M，E的标记
    S表示单独一个字；
    B代表一个词的开始
    M代表一个词的中间的字
    E代表一个词的最后一个字

    Agrs:
      input_file: 输入文本文件
      output_file: 输出做完标记后的文本文件  
    '''
    print("_character_tagging: ", input_file)
    with codecs.open(input_file, 'r', 'utf-8') as input_data, \
      codecs.open(output_file, 'w', 'utf-8') as output_data:
      for line in input_data.readlines():
        word_list = line.strip().split()

        for word in word_list:
          word = self._replace_number_alphabet(word)
          if len(word) == 1:
            output_data.write(word + "/S ")
          else:
            if len(word) == 3 and self._is_special_word(word)>0:
              output_data.write(word + "/S ")
              continue
            index =0
            if len(word)>3 and self._is_special_word(word[:3])>0:
              output_data.write(word[:3] + "/B ")
              index = 3
            else:
              output_data.write(word[0] + "/B ")
              index =1
            
            while index < (len(word)-3):
              if self._is_special_word(word[index:index+3])>0:
                output_data.write(word[index:index+3] + "/M ")
                index += 3
              else:
                output_data.write(word[index]  + "/M ")
                index +=1
            if index==(len(word)-3) and self._is_special_word(word[index:index+3])>0:
              output_data.write(word[index:index+3] + "/E ")
            else:
              while index < len(word)-1:
                output_data.write(word[index]  + "/M ")
                index +=1
              output_data.write(word[index] + "/E ")
        output_data.write("\n")

  def _replace_number_alphabet(self,word):
    ''' 
    将一个单词内的连续的数字，
    或者连续的字母用特殊符号来代替
    '''
    special_word = "$E$"
    special_number = "$N$"
    word = re.sub('([a-zA-Z]+)', special_word, word)
    word = re.sub('([0-9]+)', special_number, word)
    return word

  def _is_special_word(self,w1):
    if w1=="$E$":
      return 1
    if w1 == "$N$":
      return 2
    return 0

  def get_char_list(self,sentence):
    ''' 
    输入一句话，返回每个字的list， 
    遇到$E$, $N$这种特殊字符的时候，
    作为一个字保存为list的一个元素 
    '''  
    char_list = []
    index=0
    while index<len(sentence):
      if index <= (len(sentence)-3) and self._is_special_word(sentence[index:index+3])>0:
        char_list.append(sentence[index:index+3])
        index+=3
        continue
      char_list.append(sentence[index])
      index+=1
    return char_list
      
  def stringlist2floatlist(self,list):
    floatlist=[]
    for i in range(len(list)):
      floatlist.append(float(list[i]))
    return floatlist

def main(_):
  wordseg = ChineseSegLSTM()
  #sess, maxindex = wordseg.reload_network()
  string="千呼万唤始出来，等了好久，之前仅对 Nexus 5 开放的功能，终于正式上线了。"
  #string=string.replace(' ','')
  #print(wordseg.sentence_segment(sess, maxindex,string))
  wordseg.run_training()
  #wordseg.eval()

if __name__ == "__main__":
  tf.app.run()
