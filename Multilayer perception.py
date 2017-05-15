"""使用tensorflow实现一个多层感知机"""
"""
代码中有三种减小过拟合的方式：dropout，adagrad，relu
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#建立一个interactivesession，这样后面执行各项操作就无需指定session了
sess = tf.InteractiveSession()

in_units = 784
h1_units = 300

W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev = 0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32,[None,784])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
# 使用dropout减小过拟合，它的大致思路是在训练时，将神经网络某一层的输出节点数据随机丢弃一部分，
# 我们可以理解为随机把一张图片的50%的点去掉，即随机将50%的点变为黑点
# 这种做法实质是肠燥了很多新的随机样本，增大样本量
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop,W2) + b2)

y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(3000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.75})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accruacy  = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print (accruacy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))