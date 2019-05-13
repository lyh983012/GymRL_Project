from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
import gym_super_mario_bros.actions
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import pickle 

class Qtable(object):
	def __init__(self,actions):
		self.actions = actions
		self.table = {}

	def checkState(self,state):
		return state in self.table

	def addNewState(self,state):
		self.table[state] = np.zeros(len(self.actions))

	def returnMax(self,state):
		actionList = self.table[state]
		maxQ=max(actionList)-0.00001
		maxListArg = np.where(actionList>maxQ)		
		return np.random.choice(maxListArg[0], size=None, replace=True, p=None)

	def returnQ(self,state,action):
		return self.table[state][action]

	def returnMaxQ(self,state):
		return max(self.table[state])

	def renewQ(self,state,action,Q):
		self.table[state][action]+=Q

class RL(object):

	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		self.actions = actions 	# 是一个合法动作的list
		self.lr = learning_rate # 学习率
		self.gamma = reward_decay # 奖励衰减率
		self.epsilon = e_greedy # 贪婪度
		self.qTable = Qtable(actions)

	def chooseAction(self, observation):
		self.checkStateExist(observation) 
		if np.random.uniform() < self.epsilon:
			action = self.qTable.returnMax(observation)
		else: # 随机选择 action
			action = np.random.choice(self.actions)
		return action

	def learn(self, oldState, action, reward, newState):
		self.checkStateExist(newState) 
		Predict = self.qTable.returnQ(oldState, action) #返回得分
		if not newState[2] :
			Target = reward + self.gamma * self.qTable.returnMaxQ(newState)
		else:
			Target = reward #
		self.qTable.renewQ(oldState,action,self.lr * (Target - Predict)) 
	def checkStateExist(self, state):
		if not self.qTable.checkState(state):
			self.qTable.addNewState(state)



env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

actions = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]

actions= [ 0,1,2,3,4,5]

myRL = RL(actions)
oldObse = [-1,400,0]
for step in range(5000):
	env = gym_super_mario_bros.make('SuperMarioBros-v0')
	env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
	done = True
	for step2 in range(200):
		if done:
			env.reset()
			oldObse = (-1,400,0)
		action = myRL.chooseAction(oldObse)
		newObse, reward, done, info = env.step(action)
		print(info)
		newObse= (info['x_pos'],info['time'],done)
		myRL.learn(oldObse, action, reward, newObse)
		oldObse = newObse
		env.render()
	env.close()


file = open('saved_model/myRL.pickle','w')
pickle.dump(myRL,file)
file.close()
