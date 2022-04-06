from pickletools import optimize
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.optimizer_v1 import adam
#from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import random
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('_mpl-gallery')
from PIL import Image
import cv2

#shit
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


class DQNAgent:
    
    def __init__(self):
         # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(1,1,)),#wrong shape or sum
            tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)#maby not right...
            ])
        model.compile(loss="mse", optimizer=adam(lr=.001), metrics=['accuracy'])

        return model
    
    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
            # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


class Process():
    def __init__(self):
        #params
        self.volume = 0#liters
        self.pv = 0 #liters
        self.sp = 50 #liters
        self.u = False #on off. pump

        #variables
        self.waterOut = 0 #l/s
        self.waterIn = 0 #l/s
        self.e = 0
        self.state = np.array([0], dtype = np.float32)
        self.action = np.array([0], dtype = np.bool)
        self.df = pd.DataFrame()
        self.reward = 0
        self.done = False
        self.score = 0
        self.y = []
        with open('data.txt', 'w') as f:
                f.write('')
                print("wrote data")

        #system
        self.timeLast = 0
        self.time = 0 
        self.mafsTime = 0
        self.runTime = 0
        self.renderTime = 0
        self.renderTimeLast = 0

    def step(self):
        #self.controller()
        self.mafs()

        #reward
        self.reward = 0
        if (self.e < 3) and (self.e > -3):
            self.reward = 1  

        self.score += self.reward

        if self.runTime > 10:
            with open('data.txt', 'a') as f:
                f.write(str(self.y))
                f.write('\n')
                print("wrote data")
            self.y = []
            self.done = True


    def mafs(self):
        #mafstime
        self.time = time.time()#set time to actual time
        if (self.timeLast == 0):#on first cycle
            self.timeLast = self.time-.2
        self.mafsTime = self.time - self.timeLast #calc mafstime. basically cycletime
        self.timeLast = self.time#uptdate last time
        self.runTime += self.mafsTime#update runtime

        #assign action
        self.u = self.action

        #innløp
        self.waterIn = int(self.u) * (10)# * self.mafsTime#10 l/s

        #utløp
        self.waterOut = 1# * self.mafsTime

        #calc pv
        self.pv += self.waterIn - self.waterOut
        self.e = self.sp - self.pv
        
        self.state = np.array([self.pv])


    def reset(self):
        self.pv = 0
        self.reward = 0
        self.runTime = 0 
        self.score = 0 
        self.done = False

        self.state = np.array([self.pv])
    
    def render(self):
        #timethings
        time.sleep(.02)#to actually see shit
        self.renderTime = time.time()#set time to actual time
        if (self.renderTime - self.renderTimeLast) > .5: #only show image ever .5 sec. kinda dudu
            plt.fill_between(np.arange(0,11),[0]*11, [self.pv]*11, color='blue')
            plt.plot([0,10,10,0,0],[0,0,100,100,0], color='black')#frame
            plt.savefig('gui.png')
            cv2.destroyAllWindows() 
            cv2.imshow('Process', cv2.imread('gui.png'))
            cv2.waitKey(1)
            self.renderTimeLast = self.renderTime#uptdate last time

sim = Process()
while not sim.done:
    sim.step()
    print(sim.reward)
    sim.render()

    