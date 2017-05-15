"""使用tensorflow实现一个简单的卷积神经网络"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
sess = tf.InteractiveSession()

"""进行权重的初始化"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

"""进行偏置项的初始化"""
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

"""卷积层操作
x ：x是输入，形状类似为[-1(图片数量),28(横向像素数),28(纵向像素数),1(通道数，如果是灰度图片，通道为1，如果是rgb图片，通道为3)]
w : 形状类似为[5,5,1,32]，前两个数字代表卷积核的尺寸，1代表通道数，32代表feture map的数量
[1,1,1,1]：代表卷积模版移动的布长，都是1代表不会遗漏其中任何一个点
padding="SAME"：这样的处理方式代表给边界加上Padding让卷积的输入和输出保持同样的尺寸
"""
def conv2d(x,W):
    return tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')

"""池化操作
ksize代表了池化的大小
strides代表了步长
"""
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

x_image = tf.reshape(x,[-1,28,28,1])

#初始化第一层的参数
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

#经过第一层的卷积和池化，尺寸变为14*14
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#初始化第二层的参数
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

#经过第二层的卷积和池化，尺寸变为7*7，但是有64个feture map
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 将64个feture map上的特征连接一个有1024个隐藏层节点的全链接神经网络中
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

# 需要对此时的输出进行变形
h_poll2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_poll2_flat,w_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))

train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accruacy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))

tf.global_variables_initializer().run()

for i in range(20000):
    batch_xs,batch_ys = mnist.train.next_batch(50)
    if i%100==0:
        train_accruacy = accruacy.eval(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1.0})
        print ("step %d,training accruacy %g" % (i,train_accruacy))
    train_step.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})

print ('accuracy:'.format(accruacy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})))




