import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data  # 导入下载数据集手写体
from PIL import Image as img
import numpy as np
from matplotlib import pyplot as plt
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

class MLPNet:  # 创建一个MLPNet类
    def __init__(self):
        #定义参数
        self.keep_prob=0.9
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input_x') 
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input_label')  

        self.w1 = tf.Variable(tf.truncated_normal(shape=[784, 100], dtype=tf.float32, stddev=tf.sqrt(1 / 100))) 
        self.b1 = tf.Variable(tf.zeros([100], dtype=tf.float32)) 
 
        self.w2 = tf.Variable(tf.truncated_normal(shape=[100, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))  
        self.b2 = tf.Variable(tf.zeros([10], dtype=tf.float32)) 

	# 前向计算
    def forward(self):
        self.forward_1 = tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1)  # 全链接第一层
        self.forward_2 = tf.nn.relu(tf.matmul(self.forward_1, self.w2) + self.b2)  # 全链接第二层
        self.forward_3 = tf.nn.dropout(self.forward_2,self.keep_prob)     #dropout
        self.output = tf.nn.softmax(self.forward_3)  # softmax分类器分类
	
	# 后向计算
    def backward(self):
        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.output))
        # self.cost = tf.reduce_mean(tf.square(self.output - self.y))  # 定义均方差损失
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)      # 使用AdamOptimizer优化器 优化 self.cost损失函数

	# 计算识别精度
    def acc(self):
        self.z = tf.equal(tf.argmax(self.output, 1, name='output_max'), tf.argmax(self.y, 1, name='y_max'))
        self.accaracy = tf.reduce_mean(tf.cast(self.z, tf.float32))


if __name__ == '__main__':
    net = MLPNet() 
    net.forward()   
    net.backward() 
    net.acc()      
    losses=0
    train_loss=[]
    init = tf.global_variables_initializer()  # 定义初始化
    with tf.Session() as sess:               
        sess.run(init)                      
        print('训练集长度：%d 测试集长度：%d'%(len(mnist.train.images),len(mnist.test.images)))
        for i in range(10000):                # 训练10000次
            ax, ay = mnist.train.next_batch(100,shuffle=True)  # 从mnist数据集中取数据出来 ax接收图片 ay接收标签
            loss, accaracy, _ = sess.run(fetches=[net.cost, net.accaracy, net.opt], feed_dict={net.x: ax, net.y: ay})  # 将数据喂进神经网络(以字典的方式传入) 接收loss返回值
            train_loss.append(loss)
            if i % 1000 == 0:  # 每训练1000次
                test_ax, test_ay = mnist.test.next_batch(100)  # 则使用测试集对当前网络进行测试
                test_output = sess.run(net.output, feed_dict={net.x: test_ax})  # 将测试数据喂进网络 接收一个output值
                z = tf.equal(tf.argmax(test_output, 1, name='output_max'), tf.argmax(test_ay, 1, name='test_y_max'))  # 对output值和标签y值进行求比较运算
                test_loss=sess.run(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=test_ay,logits=test_output)))
                accaracy2 = sess.run(tf.reduce_mean(tf.cast(z, tf.float32)))  # 求出精度的准确率进行打印
                print("测试集：准确度 %f,损失函数 %f"%(accaracy2,test_loss))  # 打印当前测试集的精度 
        print("代价函数：%f"%(np.mean(train_loss)))
        plt.plot(train_loss)   #绘制图片
        plt.savefig("MLP曲线.png")
        plt.show()


