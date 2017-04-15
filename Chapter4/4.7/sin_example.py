#coding=utf-8

import tensorflow as tf
import math


'''
用tensorflow来拟合一个正弦函数

'''

def get_train_data():
  '''返回一个训练样本(train_x, train_y), 
    其中train_x是随机的自变量， train_y是train_x的sin函数值
  '''
  train_x = np.random.randn(1)
  train_y = math.sin(train_x)
  return train_x, train_y


def get_test_data():
  pass

def inference(input_data):
  with tf.variable_scope('hidden1'):
    weights = tf.get_variable("weight", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    hidden1 = tf.sigmoid(tf.matmul(input_data, weights) + biases)

  with tf.variable_scope('hidden2'):
    weights = tf.get_variable("weight", [16, 16],  tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    hidden2 = tf.sigmoid(tf.matmul(hidden1, weights) + biases)

  with tf.variable_scope('hidden3'):
    weights = tf.get_variable("weight", [16, 16],  tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    hidden3 = tf.sigmoid(tf.matmul(hidden2, weights) + biases)

  with tf.variable_scope('output_layer'):
    weights = tf.get_variable("weight", [16],  tf.float32, initializer=tf.random_normal_i
nitializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initial
izer(0.0, 0.1))
    output = tf.sigmoid(tf.matmul(hidden3, weights) + biases)

  return output


def train():
  pass



if __name__ == "__main__":
  pass
  train()
