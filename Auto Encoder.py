"""用tensorflow实现一个自动编码器"""

import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

"""Xaiver初始化器做的事情是让权重被初始化得不大不小，这里初始化为均匀分布"""
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval = low,maxval= high,dtype=tf.float32)

"""使用sklearn中得数据处理工具，对数据进行归一化处理，处理为均值为0，标准差为1的标准正态分布"""
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test

"""从总体数据集中抽取batch_size大小的数据块"""
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

class AdditiveGaussianNoiseAutoEncoder(object):

    """
    n_input:输入神经元的个数
    n_hidden:隐藏层神经元的个数
    transfer_function:隐藏层激活函数，默认为softplus
    optimizer：优化器，默认是AdamOptimizer
    scale:高斯噪声系数，为数据增加一个高斯噪声
    """
    def __init__(self,n_input,n_hidden,transfer_function = tf.nn.softplus,optimizer = tf.train.AdamOptimizer(),scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        #初始化自动编码器的权重
        network_weights = self._initialize_weights()
        self.weights = network_weights


        self.x = tf.placeholder = (tf.float32,[None,self.n_input])
        #建立一个能够进行特征提取的隐含层，先将x加入高斯噪声，然后与隐含层权重想成并加上偏置，并使用transfer对结果进行激活
        self.hidden = self.transfer(tf.add(tf.matmul(tf.add(self.x,scale*tf.random_normal((n_input,))),self.weights['w1']),self.weights['b1']))
        #对隐含层输出的数据进行重建操作
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        # 判断重建后的数据与原始输入数据的偏差
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        # 优化器不断优化，使误差最小化
        self.optimizer = optimizer.minimize(self.cost)
 
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    """初始化权重以及偏置项"""
    def _initialize_weights(self):

        all_weights = dict()

        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        #all_weights['w1'] = tf.zeros([self.n_input,self.n_hidden],dtype=tf.float32)
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input]),dtype=tf.float32)

        return all_weights

    """对单独的一个batch进行训练"""
    def partial_fit(self,X):

        cost,opt = self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        return cost


    """对测试集进行性能评估"""
    def cal_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})

    """输出自编码器隐藏层的输出结果"""
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})

    """将隐藏层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据"""
    def generate(self,hidden=None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])

        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    """整体运行一遍复原过程"""
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})

    """返回输入到隐藏层的权重"""
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    """返回输入层到隐藏层的偏置项"""
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
    X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)
    n_samples = int(mnist.train.num_examples)
    training_epoch = 20
    batch_size = 128
    display_step = 1
    autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input = 784,n_hidden = 200,transfer_function=tf.nn.softplus,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),scale=0.01)


    for epoch in range(training_epoch):
        avg_cost = 0
        total_batch = int(n_samples/batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train,batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost = cost/n_samples * batch_size

        if epoch % display_step == 0:
            print ('Epoch:',epoch,',cost=',avg_cost)

    print ('Total cost:'+str(autoencoder.cal_total_cost(X_test)))













