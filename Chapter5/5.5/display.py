#coding=utf-8

import tensorflow as tf
import os


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.02, 'gpu占用内存比例')
tf.app.flags.DEFINE_integer('batch_size', 10, 'batch_size大小')
tf.app.flags.DEFINE_string('model_dir', "./model/", '保存模型的文件夹')
tf.app.flags.DEFINE_string('event_dir', "./event/", '保存模型的文件夹')

def weight_init(shape, name):
    '''
    获取某个shape大小的参数
    '''
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

def bias_init(shape, name):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))

def conv2d(x,conv_w):
    return tf.nn.conv2d(x, conv_w, strides=[1, 1, 1, 1], padding='VALID')

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

  image = tf.cast(image, tf.float32)

  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(filename, batch_size):
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=2000)

    image, label = read_and_decode(filename_queue)

    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=1,
        capacity=4)

    return images, labels

 
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
        
    h_fc = tf.nn.relu(tf.matmul(tf.reshape(h_pool3,[-1,8*8*16]), w_fc1)  + b_fc1)

  #keep_prob = 0.8
  #h_fc_drop = tf.nn.dropout(h_fc,keep_prob)
  with tf.name_scope('fc2'):
    w_fc2 = weight_init([128, 2], 'fc2_w')
    b_fc2 = bias_init([2], 'fc2_b')

    h_fc2 = tf.matmul(h_fc, w_fc2) + b_fc2

  return h_fc2, h_conv1, h_conv2, h_conv3, h_pool3, h_fc


def train():
  '''
  训练过程
  '''
  batch_size = FLAGS.batch_size
  train_images, train_labels = inputs("./tfrecord_data/display.tfrecord", batch_size )
  
  with tf.variable_scope("inference") as scope:
    train_y_conv,  h_conv1, h_conv2, h_conv3, h_pool3, h_fc1 = inference(train_images)
    
  for i in range(16):
    image_channel = tf.slice(h_conv1, [0,0,0,i], [10, 91, 91,1])
    tf.summary.image("conv1_image", image_channel, 2)
  for i in range(16):
    image_channel = tf.slice(h_conv2, [0,0,0,i], [10, 41, 41,1])
    tf.summary.image("conv2_image", image_channel, 2) 
  for i in range(16):
    image_channel = tf.slice(h_conv3, [0,0,0,i], [10, 16, 16,1])
    tf.summary.image("conv3_image", image_channel, 2) 
  for i in range(16):
    image_channel = tf.slice(h_pool3, [0,0,0,i], [10, 8, 8,1])
    tf.summary.image("pool3_image", image_channel, 2) 
  summary_op = tf.summary.merge_all()


  init_op = tf.global_variables_initializer()
  local_init_op = tf.local_variables_initializer()
  
  saver = tf.train.Saver()
 
  gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
  config = tf.ConfigProto(gpu_options=gpu_options)
 
  with tf.Session(config=config) as sess:
    if not os.path.exists(FLAGS.model_dir):
      print("model dir not exists. %s" % FLAGS.model_dir)
      return 

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    save_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    print("reload model from %s, save_step = %d" % (ckpt.model_checkpoint_path, save_step))

    
    sess.run(local_init_op)

    summary_writer = tf.summary.FileWriter(FLAGS.event_dir, sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for i in range(20):
      h_fc1_value, summary_str = sess.run([train_y_conv, summary_op])
      summary_writer.add_summary(summary_str, i)
      print(h_fc1_value)

    summary_writer.close()
    coord.request_stop()

    coord.join(threads)

    #conv1_data = sess.run([tf.shape(h_conv1)])
    #print(conv1_data)

  
if __name__ == "__main__":
  if not os.path.exists("./tfrecord_data/train.tfrecord") or \
    not os.path.exists("./tfrecord_data/test.tfrecord"):
    gen_tfrecord_data("./data/train", "./data/test/")
 
  os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
  train()
