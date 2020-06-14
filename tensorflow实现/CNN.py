import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
# 获取数据集
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

class CNN:
    def __init__(self):
        self.keep_prob=0.9
        self.x = tf.placeholder(dtype=tf.float32, shape=[None,28,28,1], name='input_x')  # 创建一个tensorflow占位符(稍后传入图片数据),定义数据类型为tf.float32,形状shape为 None为批次 784为数据集撑开的 28*28的手写体图片 name可选参数
        self.y = tf.placeholder(dtype=tf.float32, shape=[None,10], name='input_label')  # 创建一个tensorflow占位符(稍后传入图片标签), name可选参数
        self.filter1 = tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([16]))
        
        self.filter2 = tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([32]))
        
        self.w3 = tf.Variable(tf.truncated_normal([14*14*32,128], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([128]))
        
        self.w4 = tf.Variable(tf.truncated_normal([128,10], stddev=0.1))
        self.b4 = tf.Variable(tf.zeros([10]))

        self.output=self.forward()
        self.loss=self.backward()
        self.acc=self.acc()
    def forward(self):
        conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(self.x,self.filter1, [1, 1, 1, 1],padding='SAME'), self.b1))
        conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(conv1,self.filter2,[1, 1, 1, 1],padding='SAME'), self.b2))
        conv2=tf.nn.dropout(conv2,self.keep_prob)
        pool=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        flat = tf.reshape(pool, [-1, 14*14*32])
        fc1 = tf.nn.relu(tf.matmul(flat, self.w3) + self.b3)
        fc1=tf.nn.dropout(fc1,self.keep_prob)
        fc2 = tf.nn.softmax(tf.matmul(fc1, self.w4) + self.b4)
        return fc2

    def backward(self):
        loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.output))
        self.opt = tf.train.AdamOptimizer().minimize(loss)
        return loss

    	# 计算识别精度
    def acc(self):
		# 将预测值 output 和 标签值 self.y 进行比较
        self.z = tf.equal(tf.argmax(self.output, 1, name='output_max'), tf.argmax(self.y, 1, name='y_max'))
        # 最后对比较出来的bool值 转换为float32类型后 求均值就可以看到满值为 1的精度显示
        accaracy = tf.reduce_mean(tf.cast(self.z, tf.float32))
        return accaracy

if __name__=='__main__':
    cnn=CNN()
    init = tf.global_variables_initializer()
    losses=[]
    with tf.Session() as sess:
        sess.run(init) 
        for i in range(5000):
            ax, ay = mnist.train.next_batch(100,shuffle=True) 
            train_loss,train_acc,_=sess.run(fetches=[cnn.loss,cnn.acc,cnn.opt],feed_dict={cnn.x: np.reshape(ax,[100,28,28,1]),cnn.y:ay})
            losses.append(train_loss)
            if i%500==0:
                test_ax, test_ay = mnist.test.next_batch(100)  # 则使用测试集对当前网络进行测试
                test_output = sess.run(cnn.output, feed_dict={cnn.x: np.reshape(test_ax,[100,28,28,1])})  # 将测试数据喂进网络 接收一个output值
                z = tf.equal(tf.argmax(test_output, 1, name='output_max'), tf.argmax(test_ay, 1, name='test_y_max'))  # 对output值和标签y值进行求比较运算
                test_loss=sess.run(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=test_ay,logits=test_output)))
                accaracy2 = sess.run(tf.reduce_mean(tf.cast(z, tf.float32))) 
                print("(%d/5000)准确度 %f,损失函数 %f"%(i,accaracy2,test_loss))  # 打印
        print("代价函数：%f"%(np.mean(losses)))
        plt.plot(losses)   #绘制图片
        plt.savefig("CNN曲线.png")
        plt.show()