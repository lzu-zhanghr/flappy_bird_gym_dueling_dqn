

import torch
import random
from collections import deque

class Memory(object):

    memory_size = 50000
    def __init__(self):

        self.memory = deque()

    def save_memory(self, state, action, reward, done, next_state):

        self.memory.append((state, action, reward, done, next_state))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def load_memory(self, batch_size):

        batch_memory = random.sample(self.memory, batch_size)
        states = torch.empty((batch_size, 8, 64, 64))
        next_states = torch.empty((batch_size, 8, 64, 64))
        actions = []
        rewards = []
        dones = []

        for i in range(batch_size):
            states[i] = batch_memory[i][0]
            actions.append(batch_memory[i][1])
            rewards.append(batch_memory[i][2])
            dones.append(batch_memory[i][3])
            next_states[i] = batch_memory[i][4]

        return states, actions, rewards, dones, next_states

class ReFrame(object):

    frames_size = 7

    def __init__(self):
        self.queue = []

    def pop_frame(self):
        self.queue.pop(0)

    def push_frame(self, frame):

        self.queue.append(frame)
        if len(self.queue) > self.frames_size:
            self.pop_frame()

    def get_reframe(self):

        frames = self.queue[0]

        for i in range(1, len(self.queue)):
            frames = torch.cat([frames, self.queue[i]], 1)

        return frames




