import numpy as np
import gym

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NN:
    def __init__(self,inputs,outputs,hiddens):
        self.inputs = inputs.shape[0]
        if list(outputs.shape)==[]:
            self.outputs = 1
        else:
            self.outputs = outputs.shape[0]
        self.hidden = hiddens #hidden neurons number
        self.input_size = self.inputs#len(inputs)
        self.output_size = self.outputs
        #value to split the genotype into matrix
        self.first_layer = self.input_size * self.hidden
        self.recurrent_layer = self.first_layer + self.hidden * self.hidden
        #weight matrix initialization
        self.input_weight = np.zeros((self.input_size, self.hidden))
        self.recurrent_weight = np.zeros((self.hidden, self.hidden))
        self.output_weight = np.zeros((self.hidden, self.output_size))
        #store the state of hidden neurons
        self.hiddenState = np.zeros(self.hidden)

    def netParameter(self):
        return self.input_size * self.hidden + self.hidden * self.hidden + self.hidden * self.output_size

    def setInput(self, inp):
        self.inputs = inp

    def rollout(self, env, genotype, render=False, seed=None):

        t = 0
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            env.seed(seed)
        ob=env.reset()
        done = False
        rews=0
        while not done:
            ac = self.updateNet(ob,genotype)
            # Perform a step
            ob, rew, done, _ = env.step(ac) # mujoco internally scales actions in the proper ranges!!!
            # Append the reward
            rews += rew
            t += 1
            if render:
                env.render()
            if done:
                break
        # Transform the list of rewards into an array

        return rews, t

    def updateNet(self,observation, genotype):
        self.inputs = observation

        self.input_weight = genotype[:self.first_layer].reshape(self.input_size, self.hidden)

        self.recurrent_weight = genotype[self.first_layer:self.recurrent_layer].reshape(self.hidden, self.hidden)

        self.output_weight = genotype[self.recurrent_layer:].reshape(self.hidden, self.output_size)
        lfirst = np.dot(self.inputs, self.input_weight)
        lrecurrent = np.dot(self.hiddenState, self.recurrent_weight)

        lhidden = np.tanh(lfirst+lrecurrent)
        #store the hidden state
        self.hiddenState = lhidden
        #output calculation
        lout = np.tanh(np.dot(lhidden, self.output_weight))
        if lout<0:
            return 0
        else:
            return 1
        #return lout

    def reset_net(self):
        self.input_weight = np.zeros((self.input_size, self.hidden))
        self.recurrent_weight = np.zeros((self.hidden, self.hidden))
        self.output_weight = np.zeros((self.hidden, self.output_size))
        self.hiddenState = np.zeros(self.hidden)