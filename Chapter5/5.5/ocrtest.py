#coding=utf-8

import tensorflow as tf
import os
import cv2




def gen_tfrecord(zero_dir, one_dir, output_tfrecord_file):
  '''
  '''
  tf_writer = tf.python_io.TFRecordWriter(output_tfrecord_file)

  #为数字0的数据
  for file in os.listdir(zero_dir):
    file_path = os.path.join(zero_dir, file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 0

    example = tf.train.Example()

    feature = example.features.feature
    feature['height'].int64_list.value.append(rows)
    feature['width'].int64_list.value.append(cols)
    feature['channels'].int64_list.value.append(channels)
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())
  
  #为数字1的数据
  for file in os.listdir(one_dir):
    file_path = os.path.join(one_dir, file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 1

    example = tf.train.Example()

    feature = example.features.feature
    feature['height'].int64_list.value.append(rows)
    feature['width'].int64_list.value.append(cols)
    feature['channels'].int64_list.value.append(channels)
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())

  tf_writer.close()

    
def gen_tfrecord_data(train_data_dir, test_data_dir):
  '''
  生成训练和测试的tfrecord格式的数据
  '''
  train_data_zero_dir = os.path.join(train_data_dir, "0")  
  train_data_one_dir = os.path.join(train_data_dir, "1")  

  
  test_data_zero_dir = os.path.join(test_data_dir, "0")  
  test_data_one_dir = os.path.join(test_data_dir, "1")  

  output_dir = "./tfrecord_data"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  train_tfrecord_file = os.path.join(output_dir, "train.tfrecord")
  test_tfrecord_file = os.path.join(output_dir, "test.tfrecord")
  
  gen_tfrecord(train_data_zero_dir, train_data_one_dir, train_tfrecord_file)
  gen_tfrecord(test_data_zero_dir, test_data_one_dir, test_tfrecord_file)


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

 
def inference():
  pass

def train():
  pass

def test():
  pass
  images, labels = inputs("./tfrecord_data/train.tfrecord", 1 )
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
  
if __name__ == "__main__":
  if not os.path.exists("./tfrecord_data/train.tfrecord") or \
    not os.path.exists("./tfrecord_data/test.tfrecord"):
    gen_tfrecord_data("./data/train", "./data/test/")

  test()  
