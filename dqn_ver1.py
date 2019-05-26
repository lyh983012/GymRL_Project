from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
import gym_super_mario_bros.actions
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import pickle

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD , Adam
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from collections import deque

EPISODES = 5000 # episode to train the agent
ITERATION = 5000
GAMMA = 0.95 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
INITIAL_EPSILON = 1 # starting value of epsilon
DECAY_EPSILON = 0.95 # decay rate of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-5

actions= [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left']
]
STATE_SIZE = (240, 256, 3)
IMG_ROWS = STATE_SIZE[0]
IMG_COLS = STATE_SIZE[1]
CHANNELS = 1
actions = [0,1,2,3,4,5,6]

class DQNAgent(object):

    def __init__(self, gamma, initial_epsilon, final_epsilon, decay_epsilon, lr):
        # self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma    # discount rate
        self.epsilon = initial_epsilon  # exploration rate
        self.epsilon_min = final_epsilon
        self.epsilon_decay = decay_epsilon
        self.learning_rate = lr
        self.model = self._build_model()
        # self.target_model = self._build_model()
        # self.update_target_model()


    def pre_process(self, x_t):
        """
        input: image of time step t
        output: image preprocessed
        """
        x_t = skimage.color.rgb2gray(x_t)  # colored to gray
        x_t = skimage.transform.resize(x_t,(80,80))  # down_size
        x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))  # rescale the intensity
        x_t = x_t / 255.0  # normalization
        x_t = x_t[np.newaxis, :, :, np.newaxis]  # 80*80*1
        return x_t


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add( Conv2D(16, (8, 8),
                   strides=4,
                   kernel_initializer='random_normal',
                   input_shape=(80, 80, 1)) )
        model.add(Activation('relu'))

        model.add( Conv2D(32, (4, 4),
                   strides=2,
                   kernel_initializer='random_normal') )
        model.add(Activation('relu'))
        model.add(Flatten())

        model.add( Dense(256, kernel_initializer='random_normal') )
        model.add(Activation('relu'))
        model.add( Dense(len(actions), kernel_initializer='random_normal') )

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return model


    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def chooseAction(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(np.arange(len(actions)), 1)[0]

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def learn_from_replay(self, batch_size):
        batch_mask = np.random.choice(np.arange(len(self.memory)), batch_size)
        minibatch = np.array(self.memory)[batch_mask]

        for state, action, reward, next_state, done in minibatch:
            target_q = (self.model.predict(state))

            if done:
                target_q[0][action] = reward
            else:
                q_next = (self.model.predict(next_state))[0]
                target_q[0][action] = reward + self.gamma * np.max(q_next)
                
            self.model.fit(state, target_q, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

    myRL = DQNAgent(gamma=GAMMA, initial_epsilon=INITIAL_EPSILON,
                    final_epsilon=FINAL_EPSILON,
                    decay_epsilon=DECAY_EPSILON,
                    lr=LEARNING_RATE)

    for episode in range(EPISODES):
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
        done = True
        action = 0  # action 'NOOP'

        for itera in range(ITERATION):
            if done:
                env.reset()
                oldObse, _, _, _ = env.step(0)  # get the initial state
                oldObse = myRL.pre_process(oldObse)
                # oldObse = (-1, 'NOOP', 0)

            action = myRL.chooseAction(oldObse)
            
            newObse, reward, done, info = env.step(action)
            newObse = myRL.pre_process(newObse)

            myRL.remember(state=oldObse, action=action, reward=reward, next_state=newObse, done=done)
            oldObse = newObse

            if len(myRL.memory) > BATCH:
                myRL.learn_from_replay(BATCH)
            env.render()
        env.close()

        print('Episode %d done.' % (episode+1))


    # file = open('saved_model/myRL.pickle','w')
    # pickle.dump(myRL,file)
    # file.close()

