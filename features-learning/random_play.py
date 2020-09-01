import pybullet_envs
import pybullet
import gym
import numpy as np



class rndMovement(object):
	def __init__(self,env):
		self.actions = {}
		self.outputs = {}
		self.timestep = 5
		self.env = env#gym.make("HopperBulletEnv-v0")

	def run(self,val,render=False):
		if(render):
			self.env.render(mode="human")
		t=0
		runs = 0

		while runs <val:
			ob = self.env.reset()
			tmp_actions = []
			tmp_outputs = []
			#input("ee")
			while runs <val:
				ob_0 = ob
				ac = self.env.action_space.sample()

				tmp_actions.append(list(ob_0))
				ob, rew, done, _ =self.env.step(ac)

				tmp_outputs.append(list(ob_0))
				t += 1
				if(render):
					self.env.render()
				if done:
					if len(tmp_actions) > self.timestep:

						self.actions[runs]= tmp_actions
						self.outputs[runs] = tmp_outputs
						runs += 1
					break


		return self.actions,self.outputs

#r= rndMovement([])
#r.run(1999999)
