import tensorflow as tf

#将文件名列表传入
filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"],shuffle=True,num_epochs=2)

# 采用读文本的reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 默认值是1.0,这里也默认指定了要读入数据的类型是float
record_defaults = [[1.0], [1.0]]
v1, v2 = tf.decode_csv(
    value, record_defaults=record_defaults)
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

