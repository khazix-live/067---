import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

max_steps = 2000
batch_size = 100
num_examples_for_eval = 10000
data_dir = "Cifar_data/cifar-10-batches-bin"

def variable_with_weight_loss(shape, stddev, w1):
    variable_num=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(variable_num), w1, name="weights_loss")
        tf.add_to_collection("losses", weights_loss)
    return variable_num


# 导入数据
images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 占位符存储数据
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y = tf.placeholder(tf.int32, [batch_size])

# 建立第一个卷积层结构
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)   # 卷积核
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")  # 第一个卷积层
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))   # 偏置项
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))   # relu激活
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")  # 最大池化

# 第二个卷积层结构
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64],stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.2, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding="SAME")

# 全连接结构
# 全连接需要一维数据 使用flatten  ?
# tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
reshape = tf.reshape(pool2, [batch_size, -1])
# get_shape()[1].value表示获取reshape之后的第二个维度的值
dim = reshape.get_shape()[1].value

# 第一个全连接层  fc=fully connected
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.01)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

#建立第二个全连接层
weight2 = variable_with_weight_loss(shape=[384, 10], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y,tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

#函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op = tf.nn.in_top_k(result, y, 1)

# 变量初始化
init_variable = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_variable)
    tf.train.start_queue_runners()
    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op,loss], feed_dict={x: image_batch, y: label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (step, loss_value, examples_per_sec, sec_per_batch))

        # 准确率   math.ceil()函数用于求整
        num_batch = int(math.ceil(num_examples_for_eval / batch_size))
        true_count = 0
        total_sample_count = num_batch * batch_size

        # 在一个for循环里面统计所有预测正确的样例个数
        for j in range(num_batch):
            image_batch, label_batch = sess.run([images_test, labels_test])
            predictions = sess.run([top_k_op], feed_dict={x: image_batch, y: label_batch})
            true_count += np.sum(predictions)

        # 打印正确率信息
        print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
