import tensorflow as tf

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	
if __name__ == "__main__":
	filename0 = "file0.tfrecords"
	print('Writing', filename0)
	writer = tf.python_io.TFRecordWriter(filename0)
	for index in range(10):
		example = tf.train.Example(features=tf.train.Features(feature={
			'v1': _int64_feature(index),
			'v2': _int64_feature(index + 1)}))
		writer.write(example.SerializeToString())
	writer.close()
	
	filename1 = "file1.tfrecords"
	writer = tf.python_io.TFRecordWriter(filename1)
	for index in range(10, 20):
		example = tf.train.Example(features=tf.train.Features(feature={
			'v1': _int64_feature(index),
			'v2': _int64_feature(index + 1)}))
		writer.write(example.SerializeToString())
	writer.close()
