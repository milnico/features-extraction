from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import shutil
from random_play import rndMovement



class supervised_net(object):
    def __init__(self,store_inputs,store_outputs):



        # Training Parameters
        self.store_inputs = store_inputs
        self.store_outputs = store_outputs

        self.learning_rate = 0.001
        self.training_steps = 20000
        self.batch_size = 50
        self.display_step = 200
        self.timestep = 5
        self.index =np.arange(len(store_inputs))
        #self.img_size = 28
        #self.img_shape = (self.img_size, self.img_size)
        #plot_images(images=images,cls_true=cls_true)

        # Network Parameters
        self.num_input = np.shape(store_inputs[0])[1]
        self.num_hidden = 50 # hidden layer num of features
        self.num_output = np.shape(store_outputs[0])[1]
        self.init_state = []

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.timestep, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_output])

        self.logits, self.hidden = self.create_net()
        self.loss_op = tf.reduce_mean(tf.square(self.logits-self.Y))#tf.nn.softmax(self.logits)
        # Define loss and optimizer
        #self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=self.logits, labels=self.Y))
        self.optimizer =  tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(tf.global_variables_initializer())
        self.parameters = tf.trainable_variables("supervised")

        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        self.parameter_shapes = [shape2int(p) for p in self.parameters]

        # Operations to assign new values to the parameters.
        self.parameters_placeholders = [tf.placeholder(dtype=tf.float32, shape=s) for s in self.parameter_shapes]
        self.set_parameters_ops = [par.assign(placeholder) for par, placeholder in
                                   zip(self.parameters, self.parameters_placeholders)]


    def avg_std(self):
        tmp_inputs = np.concatenate(list(self.store_inputs.values()), axis=0)
        tmp_outputs = np.concatenate(list(self.store_outputs.values()), axis=0)
        self.averagex = np.mean(tmp_inputs, axis=0)
        self.stdx = np.std(tmp_inputs, axis=0)
        self.averagey = np.mean(tmp_outputs, axis=0)
        self.stdy = np.std(tmp_outputs, axis=0)

    def get_params(self):


        p = self.sess.run(self.parameters)

        flat_list = [item for sublist in p for item in sublist]

        return np.hstack(flat_list)

    def set_parameters(self, parameters):
        # Sets network parameters from flat 1D array with parameter values.
        feed_dict = {}
        current_position = 0
        for parameter_placeholder, shape in zip(self.parameters_placeholders, self.parameter_shapes):
            length = np.prod(shape)
            feed_dict[parameter_placeholder] = parameters[current_position:current_position+length].reshape(shape)
            current_position += length
        self.sess.run(self.set_parameters_ops, feed_dict=feed_dict)

    def create_net(self):

        x = self.X

        #x = tf.unstack(x,self.timestep,1)
        with tf.variable_scope('supervised', reuse=tf.AUTO_REUSE):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden,activation=tf.nn.tanh)
            #self.init_state = tf.Variable(rnn_cell.zero_state(self.batch_size, tf.float32), trainable=False)

            #rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.num_hidden), tf.contrib.rnn.BasicLSTMCell(self.num_hidden)])
            # generate prediction
            outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)

            outputs = tf.reshape(outputs,[-1,self.num_hidden])

            logits = tf.layers.dense(outputs, units=self.num_output, use_bias=True, activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(0.1))


        '''
        with tf.variable_scope('supervised', reuse=tf.AUTO_REUSE):
            c_i = tf.layers.dense(x, units=self.num_hidden, use_bias=True, activation=tf.nn.relu,name="h0")
            c_i = tf.layers.dense(c_i, units=self.num_hidden, use_bias=True, activation=tf.nn.relu,name="h1")
            logit = tf.layers.dense(c_i, units=self.num_output, use_bias=True, activation=None,name="out")
        return logit,c_i
        '''


        return logits, outputs

    def create_batch(self,keys):
        batch_x = []
        batch_y =[]
        for i in keys:
            ind = np.random.randint(0, len(self.store_inputs[i]) - self.timestep)
            batch_x.append((self.store_inputs[i][ind:ind + self.timestep]))

            batch_y.append((self.store_outputs[i][ind:ind + self.timestep]))

        batch_y = np.reshape(batch_y,[-1,np.shape(batch_y)[-1]])
        #print(np.shape(batch_x))

        #print((batch_x - self.averagex)/(self.stdx+1e-5))
        #print((batch_y - self.averagey) / (self.stdy + 1e-5))
        xx = batch_x
        yy = batch_y
        return xx, yy

    def training(self,steps):
        # Start training
        #with tf.Session() as sess:
        self.training_steps=steps
        
        # Run the initializer
        #self.sess.run(tf.global_variables_initializer())
        #print(self.sess.run(self.parameters))
        #input("dd")
        key = list(self.store_inputs.keys())
        for step in range(1, self.training_steps+1):

            np.random.shuffle(key)

            batch_x, batch_y = self.create_batch(key[:self.batch_size])

            #batch_x, batch_y = self.store_inputs[key[0]],self.store_outputs[key[0]]

            # Run optimization op (backprop)
            self.sess.run(self.train_op, feed_dict={self.X: batch_x,self.Y: batch_y})

            if step % self.display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                logits, loss, acc,hd = self.sess.run([self.logits,self.loss_op, self.accuracy,self.hidden], feed_dict={self.X: batch_x,
                                                                                         self.Y: batch_y})

                #print(" net prediction ",logits[0])
                #print(" expected values ", batch_y[0])
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss))

        print("Optimization Finished!")

        #print(self.sess.run(self.parameters))
        #input("dd")
        #print("value ",np.reshape(self.store_inputs[12,:],(1,len(self.store_inputs[12,:]))))
        #print(sess.run(self.logits, feed_dict={self.X: np.reshape(self.store_inputs[12,:],(1,len(self.store_inputs[12,:])))}))
        #input("ee")
        #print("ttt ", np.reshape(self.store_outputs[12, :], (1, len(self.store_outputs[12, :]))))
        #input("www")
        # Calculate accuracy for 128 mnist test images
        #batch_x, batch_y = test_inputs, test_outputs
        #loss = self.sess.run([self.loss_op], feed_dict={self.X: batch_x, self.Y: batch_y})
        #print(loss)
        #input("test")
        #print("logits:", self.sess.run(self.logits, feed_dict={self.X: batch_x, self.Y: batch_y}))
        #print("batch_y :",batch_y)
        #print("batch_x :", batch_x)
        #input("ee")
        return self.parameters

    def prediction(self,values):
        values = np.expand_dims(values,axis=0)


        v,h = (self.sess.run([self.logits,self.hidden], feed_dict={self.X: values}))

        return v[-1],h[-1]

    def test(self):
        key = list(self.store_inputs.keys())
        np.random.shuffle(key)
        batch_x, batch_y = self.create_batch(key[:1])


        logits = self.sess.run(self.logits,feed_dict={self.X: batch_x})
        print("prediction ",logits)
        print(" true ",batch_y)
        input("test")


    def save_model(self,seed):
        scriptdirname = os.path.dirname(os.path.realpath(__file__))
        p = scriptdirname + '/predictorS' +str(seed)
        if os.path.exists(p):
            shutil.rmtree(p)
            os.makedirs(p)
        else:
            os.makedirs(p)
        saver = tf.train.Saver(self.parameters)
        save_path = saver.save(self.sess, p+"/model.ckpt")
        print("Model saved in path: %s" % save_path)
        #print(self.sess.run(self.parameters))

    def restore_model(self,seed):
        scriptdirname = os.path.dirname(os.path.realpath(__file__))
        p = scriptdirname + '/predictorS' + str(seed)

        saver = tf.train.Saver(self.parameters)
        save_path = saver.restore(self.sess, p + "/model.ckpt")
        print("Model restored in path: %s" % save_path)
        #print(self.sess.run(self.parameters))



def shape2int(x):
    s = x.get_shape()
    return [int(si) for si in s]
'''
play_random = rndMovement()
store_inputs, store_outputs = play_random.run(10000,render=False)


s_net = supervised_net(store_inputs, store_outputs)
s_net.training(20000)
'''
