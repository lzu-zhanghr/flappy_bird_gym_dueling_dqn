

import time
import flappy_bird_gym

flappybird = flappy_bird_gym.make("FlappyBird-rgb-v0")
init_action = [0, 1]

obs = flappybird.reset()


init_obs, _, _, _ = flappybird.step(init_action)
