from models import  mlp
import tensorflow as tf
import numpy as np
import pybullet_envs
import pybullet as p
import os
import shutil
import gym
import time
from gym import spaces
from pickle import dumps,loads
#import renderWorld
nonlin_dict = {
    'elu': tf.nn.elu,
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh
}


# If you add a new network you should add "string --> class" mapping here.
network_dict = {
    "mlp" : mlp
}


class Policy(object):
    def __init__(self, env, network, hiddens):
        # Maximum length of each episode (in steps)
        self.max_episode_len = 250000
        self.timestep = 5
        # Nonlinearity used in the network

        # Apply standard reinforcement learning preprocessing pipeline.
        # to the input frames
        self.env = env
        self.s_hiddens = hiddens
        # Shapes of the input and output of the network.
        if isinstance(self.env.action_space , spaces.Discrete):
            self.out_num = self.env.action_space.n
            print(self.out_num)
        else:
            self.out_num = len(self.env.action_space.sample())

        in_ = (hiddens,)#(self.env.observation_space.shape[0],)
        self.in_shape = list(in_)#list(self.env.observation_space.shape)


        # Placeholder for the input state
        self.input_placeholder = tf.placeholder(tf.float32, [None] + self.in_shape, name='Input')

        # Create session for 1 CPU
        #gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        #tf.set_random_seed(123)
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = tf.Session(config=tf_config)

        # Output from the network, computed values for each action.
        # In each step we will execute action with the maximum value.
        NetworkClass = network_dict[network]
        self.action_op = NetworkClass(self.input_placeholder, self.out_num)

        self.sess.run(tf.global_variables_initializer())

        # Those variables will be updated using ES algorithm.
        self.parameters = tf.trainable_variables("evo")

        # We need to save parameter shapes. Those are used when extracting parameters from flat array.
        self.parameter_shapes = [Policy.shape2int(p) for p in self.parameters]

        # Operations to assign new values to the parameters.
        self.parameters_placeholders = [tf.placeholder(dtype=tf.float32, shape=s) for s in self.parameter_shapes]
        self.set_parameters_ops = [par.assign(placeholder) for par, placeholder in
                                   zip(self.parameters, self.parameters_placeholders)]

    @staticmethod
    def shape2int(x):
        s = x.get_shape()
        return [int(si) for si in s]

    def get_parameters(self):
        # Extracts parameters from the network and returns flat 1D array with parameter values.
        parameters = self.sess.run(self.parameters)
        return np.concatenate([p.flatten() for p in parameters])

    def set_parameters(self, parameters):
        # Sets network parameters from flat 1D array with parameter values.
        feed_dict = {}
        current_position = 0
        for parameter_placeholder, shape in zip(self.parameters_placeholders, self.parameter_shapes):
            length = np.prod(shape)
            feed_dict[parameter_placeholder] = parameters[current_position:current_position+length].reshape(shape)
            current_position += length
        self.sess.run(self.set_parameters_ops, feed_dict=feed_dict)


    def rollout(self,s_net= None,seed=None, render=False):
        # Evaluates the policy for up to max_episode_len steps.

        if(render==True):
            self.env.render(mode="human")
        self.env.seed(seed)
        ob = self.env.reset()
        inp = list(ob)
        pred_inp = np.tile(inp, (self.timestep, 1))
        predicted_ob, hidden_state = s_net.prediction(pred_inp)

        #h=np.zeros(self.s_hiddens,)#s_net.prediction(list(ob)+list(first))

        ob = np.asarray(ob)

        net_input = (list(predicted_ob))#+ list(np.squeeze(h.T,1)))

        t = 0
        rew_sum = 0
        for _ in range(self.max_episode_len):

            inp = list(ob)
            if t==0:
                pred_inp = np.tile(inp,(self.timestep,1))
            else:
                pred_inp = np.roll(pred_inp,-1,axis=0)
                pred_inp[-1] = inp

            v, hidden_state = s_net.prediction(pred_inp)

            
            ac = self.sess.run(self.action_op, feed_dict={self.input_placeholder: [hidden_state*5.0]})
            predicted_ob = v
            #print("predicted ",v)
            ob, rew, done, _ = self.env.step(ac[0])#np.argmax(ac)ac[0]
            #print("true ", ob)
            #print("predicted ",np.mean(np.square(v-ob)))
            #print("just ", np.mean(np.square(fob - ob)))
            #input("pause")
            ob = np.asarray(ob)
            rew_sum += rew
            t += 1
            #input("paud")
            if render:
                self.env.render()
                #print("pred ",predicted_ob)
                #print("true ",ob)
                #time.sleep(.05)

            if done:
                break
        return rew_sum, t

    def rollout_experience(self, s_net=None, seed=None, render=False, ntrials=10):
        # Evaluates the policy for up to max_episode_len steps.
        actions = {}
        outputs = {}
        if (render == True):
            self.env.render(mode="human")
        self.env.seed(seed)

        runs = 0
        failed = 0
        while(runs < ntrials and failed<10):
            t = 0
            tmp_actions = []
            tmp_outputs = []
            ob = self.env.reset()

            h = np.zeros(self.s_hiddens, )  # s_net.prediction(list(ob)+list(first))
            
            ob = np.asarray(ob)
            predicted_ob = ob
            net_input = (list(predicted_ob) )  # + list(np.squeeze(h.T,1)))

            rew_sum = 0
            while (runs < ntrials):
                ob_0 = ob

                #ac = ac+np.random.randn(len(ac))*0.1
                inp = list(ob)
                if t == 0:
                    pred_inp = np.tile(inp, (self.timestep, 1))
                else:
                    pred_inp = np.roll(pred_inp, -1, axis=0)
                    pred_inp[-1] = inp

                v, hidden_state = s_net.prediction(pred_inp)
                ac = self.sess.run(self.action_op, feed_dict={self.input_placeholder: [hidden_state*5.0]})
                predicted_ob = v
                tmp_actions.append(inp)
                ob, rew, done, _ = self.env.step(ac[0])  # np.argmax(ac)ac[0]

                tmp_outputs.append(list(ob_0))
                ob = np.asarray(ob)
                net_input = predicted_ob
                rew_sum += rew
                t += 1
                #input("paud")
                if render:
                    self.env.render()
                if done or t==999:
                    print(len(tmp_actions))
                    print(" faile ",failed)
                    if len(tmp_actions) > self.timestep:
                        if len(tmp_actions) > 101:
                            rnd = np.random.randint(0,len(tmp_actions)-101)
                            tmp_actions = np.asarray(tmp_actions)
                            tmp_outputs = np.asarray(tmp_outputs)
                            actions[runs] = tmp_actions[rnd:rnd+100,:]
                            outputs[runs] = tmp_outputs[rnd:rnd+100,:]

                            runs+=1
                        else:
                            actions[runs] = tmp_actions
                            outputs[runs] = tmp_outputs
                            runs += 1
                    else:
                        failed+=1
                    break
        return actions,outputs
