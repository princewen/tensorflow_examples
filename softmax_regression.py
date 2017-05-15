"""
Tensorflow 实现 softmax regression识别手写数字
"""
import tensorflow as tf
"""数据导入"""
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
#(55000, 784) (55000, 10)
print(mnist.train.images.shape,mnist.train.labels.shape)
#(10000, 784) (10000, 10)
print(mnist.test.images.shape,mnist.test.labels.shape)
#(5000, 784) (5000, 10)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

"""模型编写"""

#将这个session注册为默认的session，之后的运算也默认跑在这个seesion里
sess = tf.InteractiveSession()

#placeholder是输入数据的地方，两个参数分别是数据类型和
x = tf.placeholder(tf.float32,[None,784])
#权重项
W = tf.Variable(tf.zeros([784,10]))
#偏置系数
b = tf.Variable(tf.zeros([10]))

#tf.nn包含了大量神经网络的组件
# tf.matmul是tensorflow中的矩阵乘法函数
y = tf.nn.softmax(tf.matmul(x,W)+b)

#真实的y的label
y_ = tf.placeholder(tf.float32,[None,10])

# 信息熵要最小，y_是one_hot类型的，只有一个维度是1，如果在这一个维度上，预测的y的值比较大，信息熵就比较小，所以要尽可能减小cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

#每一步的训练就是使用一个随机梯度下降的方法，使得交叉信息熵最小，0.5是学习速率
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#正确预测的矩阵
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

#计算正确率
accruacy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#变量初始化
tf.global_variables_initializer().run()

#进行迭代1000次
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch_xs,y_:batch_ys})

#打印输出准确率
print (accruacy.eval(({x:mnist.test.images,y_:mnist.test.labels})))




