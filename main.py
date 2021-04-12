
from utils import preprocess, print_info
import random
import torch
import flappy_bird_gym
from agent import DDqn

train = True
observe = 1000
explore = 200000
episode = 1000000

def init_game():

    flappybird = flappy_bird_gym.make("FlappyBird-rgb-v0")
    agent = DDqn(mode = train)
    time_step = 0
    obs = flappybird.reset()
    for i in range(7):
        agent.reframe.push_frame(preprocess(obs))

    return obs, agent, time_step, flappybird

def paly_game():

    obs, agent, time_step, flappybird = init_game()
    for i in range(episode):

        state = torch.cat([agent.reframe.get_reframe(), preprocess(obs)], 1)

        while True:

            if random.random() <= agent.epsilon and train:
                print("----------Random Action----------")
                action = flappybird.action_space.sample()
            else:
                q_eval = agent.eval_net(state.to(agent.device))
                action = torch.argmax(q_eval, -1)

            next_obs, reward, done, info = flappybird.step(action)
            # flappybird.render()
            # time.sleep(1 / 30)

            score = info['score']
            if score:
                reward *= (score * 0.1)
            else:
                reward *=0.1
            next_state = torch.cat([agent.reframe.get_reframe(), preprocess(next_obs)], 1)
            agent.reframe.push_frame(preprocess(next_obs))
            agent.memory.save_memory(state, action, reward, done, next_state)

            if train and time_step > observe:
                states, actions, rewards, dones, next_states = \
                    agent.memory.load_memory(agent.batch_size)
                states = states.to(device=agent.device)
                next_states = next_states.to(device=agent.device)
                q_eval = agent.eval_net(states)
                q_target = q_eval.clone()
                q_eval_next = agent.eval_net(next_states)
                q_target_next = agent.target_net(next_states)

                for i in range(0, agent.batch_size):
                    if dones[i]:
                        # y[i] = rewards[i]
                        q_target[i][actions[i]] = rewards[i]
                    else:
                        action_index = torch.argmax(q_eval_next[i])
                        q_target[i][actions[i]] = rewards[i] + agent.gamma * q_target_next[i][action_index]

                agent.optimizer.zero_grad()
                loss = agent.loss(q_target, q_eval)
                loss.backward()
                agent.optimizer.step()
                agent._save_model(time_step)

            if agent.epsilon > agent.final_epsilon and time_step > observe:
                agent.epsilon -= (agent.init_epsilon - agent.final_epsilon) / (observe + explore)

            if time_step % agent.replace_interval == 0:
                agent.target_net.load_state_dict(agent.eval_net.state_dict())
                print('target_params_replaced')

            if time_step < observe:
                print_info(time_step, 'observe', agent.epsilon, action, reward)
            elif observe <= time_step and time_step < explore:
                print_info(time_step, 'explore', agent.epsilon, action, reward)
            else:
                print_info(time_step, 'train', agent.epsilon, action, reward)

            if done:
                print('game over with score:', score)
                obs = flappybird.reset()
            state = next_state
            time_step += 1


if __name__ == '__main__':
    paly_game()
























