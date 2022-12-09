# 导入tensorflow1.x模块
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生产200个随机点，目标逼近曲线
x_data = np.linspace(-1, 1, 200)[:, np.newaxis]  # 新的矩阵
#print(x_data)
# 噪声点
noise = np.random.normal(0, 0.1, x_data.shape)
# 绘制离散函数曲线方程 y=x²+noise
y_data = np.square(x_data) + noise
# 占位符储存数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 神经网络中间层
w1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]))
# 计算公式
re1 = tf.matmul(x, w1) + b1
# 加入激活函数
act_re = tf.nn.tanh(re1)

# 输出层
w2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.zeros([1, 1]))
# 计算公式
re2 = tf.matmul(act_re, w2) + b2
# 加入激活函数
prediction_re = tf.nn.tanh(re2)

# 损失函数
loss = tf.reduce_mean(tf.square(y-prediction_re))
# 反向传递
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练轮次
    for epoch in range(500):
        sess.run(train_step, feed_dict={x : x_data, y: y_data})

    prediction = sess.run(prediction_re, feed_dict={x : x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction, 'r-', lw=5)  # 曲线是预测值
    plt.show()
