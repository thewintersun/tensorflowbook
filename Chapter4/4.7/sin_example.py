#coding=utf-8

import tensorflow as tf
import math
import numpy as np

'''
用tensorflow来拟合一个正弦函数

'''

def get_train_data():
  '''返回一个训练样本(train_x, train_y), 
    其中train_x是随机的自变量， train_y是train_x的sin函数值
  '''
  train_x = np.random.uniform(-10.0, 10.0, (1))
  train_y = math.sin(train_x)
  return train_x, train_y


def get_test_data():
  '''返回一个测试数据(test_x, test_y), 
    
  '''
  test_x = np.random.uniform(-10.0, 10.0, (1))
  test_y = math.sin(test_x)
  return test_x, test_y
  
def test(input_data):
  with tf.variable_scope('test'):
    weights = tf.get_variable("weight", [1, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [1, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    hidden1 = tf.sigmoid(tf.multiply(input_data, weights) + biases)
  with tf.variable_scope('test2'):
    weights = tf.get_variable("weight", [16, 16],  tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    mul = tf.matmul(hidden1, weights)
    hidden2 = tf.sigmoid(mul + biases)
  with tf.variable_scope('test3'):
    weights = tf.get_variable("weight", [16, 16],  tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    hidden3 = tf.sigmoid(tf.matmul(hidden2, weights) + biases)

  with tf.variable_scope('testoutput_layer'):
    weights = tf.get_variable("weight", [16, 1],  tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    output = tf.sigmoid(tf.matmul(hidden3, weights) + biases)
  return tf.shape(hidden1), tf.shape(weights), output

def inference(input_data):
  with tf.variable_scope('hidden1'):
    weights = tf.get_variable("weight", [1, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [1, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    hidden1 = tf.sigmoid(tf.multiply(input_data, weights) + biases)


  with tf.variable_scope('hidden2'):
    weights = tf.get_variable("weight", [16, 16],  tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    mul = tf.matmul(hidden1, weights)
    hidden2 = tf.sigmoid(mul + biases)

  with tf.variable_scope('hidden3'):
    weights = tf.get_variable("weight", [16, 16],  tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    hidden3 = tf.sigmoid(tf.matmul(hidden2, weights) + biases)

  with tf.variable_scope('output_layer'):
    weights = tf.get_variable("weight", [16, 1],  tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    biases  = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
    output = tf.sigmoid(tf.matmul(hidden3, weights) + biases)

  return output


def train():
  learning_rate = 0.0001
  x = tf.placeholder(tf.float32)
  y = tf.placeholder(tf.float32)
  
  
  net_out = inference(x)
  loss = tf.square(net_out - y)
  opt = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = opt.minimize(loss)
  
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)
    print("start traing....")
    for i in range(10000000):
      train_x, train_y = get_train_data()
      sess.run(train_op, feed_dict={x: train_x, y: train_y})
      '''
      t, t2, m = sess.run([tt, tt2, mm], feed_dict={x: train_x, y: train_y})
      print(t)
      print(t2)
      print(m)
	  '''

      if i % 10 == 0:
        test_x, test_y = get_test_data()
        loss_value = sess.run(loss, feed_dict={x: [test_x], y: test_y})
        print("step %d test loss is %f" %(i , loss_value))






if __name__ == "__main__":
  pass
  train()
