import tensorflow as tf
import numpy as np
from numba import cuda
import time, os
# from layers import *

# Globals
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Helper Functions
def shuffle_arrays(a, b):
    assert(len(a) == len(b))
    mask = np.random.permutation(len(a))
    return a[mask], b[mask]

def next_batch(X, size=1):
    beg = 0
    N = len(X)
    while (True):
        if (beg+size <= N):
            yield X[beg:beg+size]
            beg += size
        else:
            yield np.vstack((X[beg:], X[:beg+size-N]))
            beg += size-N

# Classes
class Network(object):
    # Super Class for Networks
    def __init__(self, name = "Network"):
        self.name = name
        self.network = self.build_architecture()

    def build_architecture(self):
        network = {}
        return network

    def train(self, x_train, y_train):
        loss = 0
        return loss

    def test(self, x_test):
        y_test = None
        return y_test

class CNN_tf(Network):
    def __init__(self, name = "Network", use_gpu = False):
        self.name = name
        self.network = self.build_architecture(use_gpu)

    def build_architecture(self, use_gpu = False):
        # Adapted from tensorflow tutorial
        # https://www.tensorflow.org/tutorials/estimators/cnn
        if (use_gpu):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            # dev_name = "device:GPU:0"
        else:
            sess = tf.Session()
            # dev_name = "CPU:0"
        network = {}
        # Input Layer
        # with tf.device(dev_name):
        x_train = tf.placeholder(tf.float64, shape=(None, 28, 28))
        y_train = tf.placeholder(tf.float64, shape=(None, 10))
        data = tf.reshape(x_train, shape=(-1, 28, 28, 1))
        # Convolution 1
        conv1 = tf.layers.conv2d(
            inputs = data,
            filters = 32,
            kernel_size = [5, 5],
            padding = "same",
            activation = tf.nn.relu)
        # Pool 1
        pool1 = tf.layers.max_pooling2d(
            inputs = conv1,
            pool_size = [2, 2],
            strides = 2)
        # Convolution 2
        conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = [5, 5],
            padding = "same",
            activation = tf.nn.relu)
        # Pool 2
        pool2 = tf.layers.max_pooling2d(
            inputs = conv2,
            pool_size = [2, 2],
            strides = 2)
        # Dense
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        # Logits Layer
        logits = tf.layers.dense(inputs=dense, units=10)
        y_pred = tf.argmax(input=logits, axis = 1)
        # Loss and Train Step
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y_train, logits=logits)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss)
        # Save useful tensors
        network["x_train"] = x_train
        network["y_train"] = y_train
        network["y_pred"]  = y_pred
        network["train_op"]= train_op
        network["loss"] = loss
        # Initialize all paramters
        sess.run(tf.global_variables_initializer())
        self.sess = sess
        return network

    def train(self, x_train, y_train):
        network = self.network
        sess = self.sess
        x_train_t = network["x_train"]
        y_train_t = network["y_train"]
        train_op  = network["train_op"]
        loss_t    =  network["loss"]
        _, loss = sess.run([train_op, loss_t], feed_dict={x_train_t: x_train, y_train_t: y_train})
        return loss

    def test(self, x_test):
        network = self.network
        sess = self.sess
        x_train_t = network["x_train"]
        y_pred_t  = network["y_pred"]
        y_pred = sess.run([y_pred_t], feed_dict={x_train_t: x_test})
        return y_pred

# class CNN_np(Network):
#     def __call__(self, x):
#         return self.forward(x)

#     def build_architecture(self):
#         network = [ Conv2D(BATCH_SIZE, 28, 5, 1),
#                     ReLU(),
#                     Conv2D(32, 24, 5, 1),
#                     ReLU(),
#                     Conv2D(16, 4, 1, 1),
#                     Flatten() ]
#         return network
    
#     def init_weights(self, weights):
#         # Load the weights for your CNN from the MLP Weights given
#         self.layers[0].W = np.array([weights[0].T[i].reshape(8, 24).T\
#                                 for i in range(len(weights[0].T))])
#         self.layers[2].W = weights[1].T.reshape(16, 8, 1)
#         self.layers[4].W = weights[2].T.reshape(4, 16, 1)
        

#     def forward(self, x):
#         # You do not need to modify this method
#         out = x
#         for layer in self.layers:
#             out = layer(out)
#         return out

#     def backward(self, delta):
#         # You do not need to modify this method
#         for layer in self.layers[::-1]:
#             delta = layer.backward(delta)
#         return delta

#     def train(self, x_train, y_train):
#         y_pred = self.forward(x_train)
#         delta = (y_pred-y_train)**2
#         return self.backward(delta)
        
#     def test(self, x_test):
#         return self.forward(x_test)

class CNN_cuda(Network):
    def build_architecture(self):
        network = {}
        return network

    def train(self, x_train, y_train):
        loss = 0
        return loss

    def test(self, x_test):
        y_test = None
        return y_test

def main():
    # Load Data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]

    # Preprocess Data
    x_train, y_train = shuffle_arrays(x_train, y_train)
    x_test, y_test = shuffle_arrays(x_test, y_test)
    x_gen, y_gen = next_batch(x_train, BATCH_SIZE), next_batch(y_train, BATCH_SIZE)

    # Construct Different Network Models
    cnn_tf_cpu = CNN_tf(name="cnn_tf_cpu")
    cnn_tf_gpu = CNN_tf(name="cnn_tf_gpu", use_gpu=True)
    # cnn_np = CNN_np(name="cnn_np_cpu")
    cnn_cuda = CNN_cuda(name="cnn_cuda")
    # nets = [cnn_tf_gpu, cnn_np, cnn_cuda, cnn_tf_cpu]

    # Start Training
    net = cnn_tf_gpu
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        for iteration in range(len(x_train)//BATCH_SIZE+1):
            x_batch, y_batch = next(x_gen), next(y_gen)
            loss = net.train(x_batch, y_batch)
            if (iteration%20==0):
                total_time = time.time()-start_time
                print("Name: %s\nEpoch: %d\nStep: %d\nTime: %6f(s)\nLoss: %6f\n"%\
                    (net.name, epoch, iteration, total_time, loss))            
                start_time = time.time()
           
           
        

if __name__ == '__main__':
    main()
            