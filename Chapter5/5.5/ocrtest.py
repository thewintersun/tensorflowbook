#coding=utf-8

import tensorflow as tf
import os
import cv2


def weight_init(shape, name):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

def bias_init(shape, name):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))

def conv2d(x,w):
    return tf.nn.conv2d(x,w, strides=[1, 1, 1, 1], padding='VALID')

def max_pool(x, size):
    return tf.nn.max_pool(x, ksize=[1,size,size,1], strides = [1,size,size,1], padding='VALID')



def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'channels': tf.FixedLenFeature([], tf.int64),
          'image_data': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image_data'], tf.uint8)
  image = tf.reshape(image, [100, 100, 3])

  #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(filename, batch_size):
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename])

    image, label = read_and_decode(filename_queue)

    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=1,
        capacity=4,
        min_after_dequeue=2)

    return image, label

 
def inference(input_data):
  '''
  定义网络结构、向前计算过程
  '''
  with tf.name_scope('conv1'):
    w_conv1 = weight_init([10,10,3,16], 'conv1_w')
    b_conv1 = bias_init([16], 'conv1_b')
  
    #卷积之后图片大小变成100-10+1 = 91
    h_conv1 = tf.nn.relu(conv2d(input_data, w_conv1) + b_conv1)
    #池化之后图片大小变成45
    h_pool1 = max_pool(h_conv1, 2) #140

  with tf.name_scope('conv2'):
    w_conv2 = weight_init([5,5,16,16], 'conv2_w')
    b_conv2 = bias_init([16], 'conv2_b')

    #卷积之后图片大小变成 45 -5+1 = 41
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    #池化之后图片大小变成20
    h_pool2 = max_pool(h_conv2, 2) #68

  with tf.name_scope('conv3'):
    w_conv3 = weight_init([5,5,16,16], 'conv3_w')
    b_conv3 = bias_init([16], 'conv3_b')
  
    #卷积之后图片大小变成 20 -5+1 = 16
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
    #池化之后图片大小变成8
    h_pool3 = max_pool(h_conv3, 2) # 32

  with tf.name_scope('fc1'):
    w_fc1 = weight_init([8*8*16, 128], 'fc1_w')
    b_fc1 = bias_init([128], 'fc1_b')
        
    h_fc = tf.nn.relu(tf.matmul(tf.reshape(h_pool4,[-1,8*8*16]), w_fc1)  + b_fc1)

  #keep_prob = 0.8
  #h_fc_drop = tf.nn.dropout(h_fc,keep_prob)
  with tf.name_scope('fc2'):
    w_fc2 = weight_init([128, 2], 'fc2_w')
    b_fc2 = bias_init([2], 'fc2_b')

    h_fc2 = tf.matmul(h_fc, w_fc2) + b_fc2

  return h_fc2


def train():
  '''
  训练过程
  '''
  batch_size = 1
  train_images, train_labels = inputs("./tfrecord_data/train.tfrecord", batch_size )
  test_images, test_labels = inputs("./tfrecord_data/train.tfrecord", batch_size )
  with tf.variable_scope("inference") as scope:
    train_y_conv = inference(train_images)
    scope.reuse_variables()
    test_y_conv = inference(test_images)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=y_conv))
  train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(train_labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  init_op = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      while not coord.should_stop():
        for i in range(10000):
          sess.run(train_op)
          if i % 100 == 0:
            train_accuracy = sess.run(accuracy)
            print("step %d training_acc is %.4f" % (i, train_accuracy))

    except tf.errors.OutOfRangeError:
      pass
    finally:
      coord.request_stop()
  
  


  
if __name__ == "__main__":
  if not os.path.exists("./tfrecord_data/train.tfrecord") or \
    not os.path.exists("./tfrecord_data/test.tfrecord"):
    gen_tfrecord_data("./data/train", "./data/test/")

