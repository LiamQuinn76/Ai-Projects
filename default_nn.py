import torch
import numpy as np
from backpropagation import OptimiserLinearLayer, MiniBatch
import random

class Env(object):
    # temp env
    def __init__(self, obs_len: int):
        self.observation_space = obs_len
        self.action_space = 1

    def step(self):
        # Decide whether to generate positive or negative numbers
        #sign = random.choice([1, -1])
        sign = 1
        
        # Generate a list of 5 integers (you can change the length if needed) with values between 1 and 9
        int_list = [random.randint(1, 9) * sign for _ in range(self.observation_space)]
        
        # Calculate the sum of the list
        sum_of_list = sum(int_list)

        # format input
        obs = []
        for item in int_list:
            obs.append([item])
        
        # Normalize the sum to be between 0 and 1
        normalized_sum = sign * (sum_of_list / (9 * self.observation_space))

        return np.transpose(obs), normalized_sum




class Layer(object):
    def __init__(self, activations: int, neurons: int):
        # (no. neuron, no. activations)
        self.weight = np.random.randint(0, 2, (neurons, activations))
        # (no. neurons, 1)
        self.bias = np.zeros((neurons, 1))
        self.z = None
        
    def activate(self, activations: int) -> np.ndarray:
        # z = [weights][activations] + bias
        z = np.dot(self.weight, activations)
        z = np.add(z, self.bias)
        self.z = z
        return z

class NeuralNetwork(object):
    def __init__(self, env: Env):
        # initalise the network
        self.layers = (Layer(env.observation_space, 4),
                       Layer(4, 4),
                       Layer(4, env.action_space))

    def forward_pass(self, obs: list) -> list:
        # tracking var
        activations = []

        # for layer in network
        y = np.transpose(obs) # vertical -> horizontal nodes
        for layer in self.layers:
            x = layer.activate(np.array(y)) # calc z
            y = torch.sigmoid(torch.tensor(x)) # o(z) 

            activations.append(y)
        
        return activations
    


    
if "__main__" == __name__:
    # init traiing params
    episode_length = 8
    training_eps = 256

    # init game objects
    env = Env(obs_len=1)
    nn = NeuralNetwork(env)
    optimiser = OptimiserLinearLayer()
    mb = MiniBatch(episode_length=episode_length, num_layer=len(nn.layers))

    # training loop
    for training_episode in range(training_eps):
        for step in range(mb.episode_length):
            obs, label = env.step()
            activations = nn.forward_pass(obs)
            y = optimiser.calc_cost(activations[-1], label)

            for layer in reversed(nn.layers):
                z = layer.z
                weight = layer.weight
                bias = layer.bias
                layer_num = nn.layers.index(layer)
                current_activation = activations[layer_num]

                if layer_num > 0:
                    previous_activation = activations[layer_num - 1]
                    db, dw = optimiser.relevent_derivative(z, current_activation, previous_activation, torch.tensor(y))
                    y = optimiser.calc_next_cost(db, weight)
                
                else: 
                    db, dw = optimiser.relevent_derivative(z, current_activation, np.transpose(obs), torch.tensor(y))

                mb.running_total(db, dw, layer_num)
        nn = optimiser.update_network(nn, mb.average)









    


