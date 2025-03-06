import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    

class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        # initalise the network
        self.layers = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 4)),
            nn.Tanh(),
            layer_init(nn.Linear(4, 4)),
            nn.Tanh(),
            layer_init(nn.Linear(4, 4)),
            nn.Tanh(),
            layer_init(nn.Linear(4, env.action_space.n)))

    def forward_pass(self, x):
        logits = self.layers(x)
        probs = Categorical(logits=logits) # stochastic model
        action = probs.sample() # sample action space

        return action, probs.log_prob(action)

def reinforce(reward, log_prob):
    return reward * (-log_prob)

    
if "__main__" == __name__:
    # init env & agent
    env = gym.make("CartPole-v0")
    network = Agent(env)

    num_steps = 256
    num_updates = 1000
    learning_rate = 1e-3
    gamma = 0.98

    actions = torch.zeros(num_steps)
    logprobs = torch.zeros(num_steps)

    optimizer = optim.Adam(network.parameters(), learning_rate)

    # for episode
    for episode in range(num_updates):

        # anneal learning rate
        frac = 1.0 - (episode - 1.0) / num_updates
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        # init env
        next_obs = env.reset()
        log_probs = [] # need to make numpy arrays
        rewards = []
        episodic_return = 0

        # for step
        for step in range(num_steps):
            # calc prob & reward
            action, log_prob = network.forward_pass(torch.tensor(next_obs))

            # step env
            next_obs, reward, done, _ = env.step(np.array(action))

            # prob & reward
            log_probs.append(log_prob)
            rewards.append(reward)
            episodic_return += reward

            if done:
                break

        print(episodic_return)

        # calc discounted return
        returns = []
        for step in range(len(rewards)):
            if len(returns) > 0:
                discounted_return = returns[0] 
            else:
                discounted_return = 0
            returns.insert(0, gamma * discounted_return + rewards[step])

        # normalise returns
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6) # to avoid dividing by 0

        # policy loss
        loss = []
        for log_prob, disc_return in zip(log_probs, returns):
            loss.append(-log_prob * disc_return)
        #loss = torch.tensor(loss).sum()
        loss = sum(loss)

        # optimise network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()










    






    


