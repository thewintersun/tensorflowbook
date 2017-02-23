import tensorflow as tf

#将文件名列表传入
filename_queue = tf.train.string_input_producer(["file0.tfrecords", "file1.tfrecords"],shuffle=True,num_epochs=2)

# 使用TFRecorder来读取
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
      serialized_example,
      features={
          'v1': tf.FixedLenFeature([], tf.int64),
          'v2': tf.FixedLenFeature([], tf.int64),
      })
	  
v1 = tf.cast(features['v1'], tf.int32)
v2 = tf.cast(features['v2'], tf.int32)
v_mul = tf.multiply(v1,v2)
print(v1.get_shape())

batch_v1, batch_v2 = tf.train.shuffle_batch([v1,v2], 
                                  batch_size=10,  #batch的大小
                                  num_threads=16, #处理线程数
                                  capacity=100,   #队列大小
                                  min_after_dequeue=20 #出队后队列最少保留样本数
                                  )
print(batch_v1.get_shape())

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

# 创建会话
sess = tf.Session()

# 初始化变量
sess.run(init_op)
sess.run(local_init_op)

# 输入数据进入队列
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        value1, value2, mul_result = sess.run([v1,v2,v_mul])
        print("%f\t%f\t%f"%(value1, value2, mul_result))

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()

# 等待线程结束
coord.join(threads)
sess.close()

