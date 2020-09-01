import pybullet_envs
import pybullet
import gym
import numpy as np
import policy
#import os
#scriptdirname = os.path.dirname(os.path.realpath(__file__))



#net = Policy(env,'mlp','elu')
#fname = scriptdirname + "/bestgS" + str(1) + ".npy"
#bestgeno = np.load(fname)

#scriptdirname = os.path.dirname(os.path.realpath(__file__))


class rndMovement(object):
	def __init__(self,env):
		self.actions = []
		self.outputs = []
		self.env = env#gym.make("HopperBulletEnv-v0")

	def run(self,val):
		self.env.render(mode="human")
		t=0
		while t <val:
			ob = self.env.reset()
			#input("ee")
			for i in range(500):#while True:
				ac = self.env.action_space.sample()

				self.actions.append(list(ob)+list(ac))
				ob, rew, done, _ =self.env.step(ac)
				self.outputs.append(list(ob))
				self.env.render()
				if done:
					self.env.reset()
				t+=1

		#print(len(self.outputs))
		return self.actions,self.outputs

