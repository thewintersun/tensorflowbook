import tensorflow as tf

#将文件名列表传入
filename_queue = tf.train.string_input_producer(["file0.bin", "file1.bin"],shuffle=True,num_epochs=2)

# 采用读取固定长度二进制数据的reader,一次读入2个float数
reader = tf.FixedLengthRecordReader(record_bytes=2*4)
key, value = reader.read(filename_queue)

# 将读入的数据按照float32的大小解码
decode_value = tf.decode_raw(value, tf.float32)
v1 = decode_value[0]
v2 = decode_value[1]
v_mul = tf.mul(v1,v2)

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

