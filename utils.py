
import cv2
import torch

def print_info(time_step, state, epsilon, action, reward):
    print("TIMESTEP", time_step, " STATE", state,
          " EPSILON", epsilon, " ACTION", action,
          " REWARD", reward)

def preprocess(observation):

    observation = cv2.cvtColor(cv2.resize(
        observation, (64, 64)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return torch.torch.FloatTensor(observation).resize_((1, 1, 64, 64))


