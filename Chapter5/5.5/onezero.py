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
    img_data = cv2.imread(file_path)
    image_raw = img_data.tostring()
    rows = img_data.shape[0]
    cols = img_data.shape[1]
    channels = img_data.shape[2]
    label_data = 0

    example = tf.train.Example()

    feature = example.features.feature
    feature['height'].int64_list.value.append(rows)
    feature['width'].int64_list.value.append(cols)
    feature['channels'].int64_list.value.append(channels)
    feature['image_raw'].bytes_list.value.append(image_raw)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())
  
  #为数字1的数据
  for file in os.listdir(one_dir):
    file_path = os.path.join(zero_dir, file)
    img_data = cv2.imread(file_path)
    image_raw = img_data.tostring()
    rows = img_data.shape[0]
    cols = img_data.shape[1]
    channels = img_data.shape[2]
    label_data = 1

    example = tf.train.Example()

    feature = example.features.feature
    feature['height'].int64_list.value.append(rows)
    feature['width'].int64_list.value.append(cols)
    feature['channels'].int64_list.value.append(channels)
    feature['image_raw'].bytes_list.value.append(image_raw)
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
  
def inference():
  pass

def train():
  pass
  
  
if __name__ == "__main__":
  if not os.path.exists("./tfrecord_data/train.tfrecord") or 
    not os.path.exists("./tfrecord_data/test.tfrecord"):
    gen_tfrecord_data("./data/train", "./data/test/")

  