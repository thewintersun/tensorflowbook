#coding=utf-8

import tensorflow as tf
import os


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.02, 'gpu占用内存比例')
tf.app.flags.DEFINE_integer('batch_size', 10, 'batch_size大小')
tf.app.flags.DEFINE_integer('reload_model', 0, '是否reload之前训练好的模型')
tf.app.flags.DEFINE_string('model_dir', "./model/", '保存模型的文件夹')
tf.app.flags.DEFINE_string('event_dir', "./event/", '保存event数据的文件夹,给tensorboard展示用')

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

  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  #image = tf.cast(image, tf.float32) 

  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(filename, batch_size):
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=2000)

    image, label = read_and_decode(filename_queue)

    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=1,
        capacity=4,
        min_after_dequeue=2)

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

  return h_fc2


def train():
  '''
  训练过程
  '''
  global_step = tf.get_variable('global_step', [],
                                initializer=tf.constant_initializer(0),
                                trainable=False, dtype=tf.int32)

  batch_size = FLAGS.batch_size
  train_images, train_labels = inputs("./tfrecord_data/train.tfrecord", batch_size )
  test_images, test_labels = inputs("./tfrecord_data/train.tfrecord", batch_size )
  
  
  train_labels_one_hot = tf.one_hot(train_labels, 2, on_value=1.0, off_value=0.0)
  test_labels_one_hot = tf.one_hot(test_labels, 2, on_value=1.0, off_value=0.0)

  #因为任务比较简单，故意把学习率调小了，以拉长训练过程。
  learning_rate = 0.000001
  
  
  with tf.variable_scope("inference") as scope:
    train_y_conv = inference(train_images)
    scope.reuse_variables()
    test_y_conv = inference(test_images)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_one_hot, logits=train_y_conv))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.minimize(cross_entropy, global_step=global_step)

  train_correct_prediction = tf.equal(tf.argmax(train_y_conv, 1), tf.argmax(train_labels_one_hot, 1))
  train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

  test_correct_prediction = tf.equal(tf.argmax(test_y_conv, 1), tf.argmax(test_labels_one_hot, 1))
  test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
  
  
  init_op = tf.global_variables_initializer()
  local_init_op = tf.local_variables_initializer()
  
  saver = tf.train.Saver()
  tf.summary.scalar('cross_entropy_loss', cross_entropy)
  tf.summary.scalar('train_acc', train_accuracy)
  summary_op = tf.summary.merge_all()
 
  gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
  config = tf.ConfigProto(gpu_options=gpu_options)
 
  with tf.Session(config=config) as sess:
    if FLAGS.reload_model == 1:
      ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
      saver.restore(sess, ckpt.model_checkpoint_path)
      save_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      print("reload model from %s, save_step = %d" % (ckpt.model_checkpoint_path, save_step))
    else:
      print("Create model with fresh paramters.")
      sess.run(init_op)
      sess.run(local_init_op)

    summary_writer = tf.summary.FileWriter(FLAGS.event_dir, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      while not coord.should_stop():
        _, g_step = sess.run([train_op, global_step])
        if g_step % 2 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, g_step)

        if g_step % 100 == 0:
          train_accuracy_value, loss = sess.run([train_accuracy, cross_entropy])
          print("step %d training_acc is %.2f, loss is %.4f" % (g_step, train_accuracy_value, loss))
        if g_step % 1000 == 0:
          test_accuracy_value = sess.run(test_accuracy)
          print("step %d test_acc is %.2f" % (g_step, test_accuracy_value))

        if g_step % 2000 == 0:
          #保存一次模型
          print("save model to %s" % FLAGS.model_dir  + "model.ckpt." + str(g_step) )
          saver.save(sess, FLAGS.model_dir + "model.ckpt", global_step=global_step)

    except tf.errors.OutOfRangeError:
      pass
    finally:
      coord.request_stop()

    coord.join(threads) 

  
if __name__ == "__main__":
  if not os.path.exists("./tfrecord_data/train.tfrecord") or \
    not os.path.exists("./tfrecord_data/test.tfrecord"):
    gen_tfrecord_data("./data/train", "./data/test/")
 
  os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
  #test()
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
  train()
