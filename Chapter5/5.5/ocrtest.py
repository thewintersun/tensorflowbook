#coding=utf-8

import tensorflow as tf
import os
import cv2


def weight_init(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_init(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

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
  pass
  w_conv1 = weight_init([20,20,1,16])
  b_conv1 = weight_init([16])

  h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
  h_pool1 = max_pool(h_conv1, 2) #140

  #init layer 2:
  w_conv2 = weight_init([5,5,16,16])
  b_conv2 = weight_init([16])
        
  h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
  h_pool2 = max_pool(h_conv2, 2) #68

  #init layer 3:
  w_conv3 = weight_init([5,5,16,16])
  b_conv3 = weight_init([16])
       
  h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
  h_pool3 = max_pool(h_conv3, 2) # 32

  #init layer 4:
  w_conv4 = weight_init([5,5,16,16])
  b_conv4 = weight_init([16])
        
  h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4) + b_conv4)
  h_pool4 = max_pool(h_conv4, 2)   #14 14 16

  #init fc layer:
  w_fc1 = weight_init([14*14*16, 1024])
  b_fc1 = weight_init([1024])
        
  h_fc = tf.nn.relu(tf.matmul(tf.reshape(h_pool4,[-1,14*14*16]), w_fc1)  + b_fc1)

  keep_prob = tf.placeholder("float")
  #h_fc_drop = tf.nn.dropout(h_fc,keep_prob)

  w_fc2 = weight_init([1024, 2])
  b_fc2 = weight_init([2])

  h_fc2 = tf.matmul(h_fc, w_fc2) + b_fc2
  y_conv = tf.nn.softmax(h_fc2)

  

def train():
  '''
  训练过程
  '''
  batch_size = 1
  train_images, train_labels = inputs("./tfrecord_data/train.tfrecord", batch_size )

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      while not coord.should_stop():
        i = sess.run(images)
        print(i)
    except tf.errors.OutOfRangeError:
      pass
    finally:
      coord.request_stop()
  
  
  pass

  
if __name__ == "__main__":
  if not os.path.exists("./tfrecord_data/train.tfrecord") or \
    not os.path.exists("./tfrecord_data/test.tfrecord"):
    gen_tfrecord_data("./data/train", "./data/test/")

