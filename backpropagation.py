
import numpy as np
import torch


# activation function
def sigmoind_function(x: torch.Tensor) -> np.ndarray:
    output = torch.sigmoid(x.clone().detach())
    return np.array(output)


def d_sigmoind_function(x: torch.Tensor) -> np.ndarray:
    o = sigmoind_function(x)
    o = np.array(o)
    ones = np.ones(o.shape) 
    return o * (ones - o)
    
# cost function
def squared_difference(x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    return np.array((x - y)**2)

def d_squared_difference(x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    return np.array(2 * (x - y))



class OptimiserLinearLayer():
    def __init__(self):
        # initalise exact functions used
        self.activation_function = sigmoind_function
        self.d_activation_function = d_sigmoind_function
        self.cost = squared_difference
        self.d_cost = d_squared_difference


    def relevent_derivative(self, z, current_activation, previous_activation, y) -> np.ndarray:
        # dc_db = 1 * da_dz * dc_da
        da_dz = self.d_activation_function(torch.tensor(z))
        dc_da = -1 *(self.d_cost(current_activation, y))
        dc_db = da_dz * dc_da

        # dc_db = a(L-1) * da_dz * dc_da
        dc_dw = np.dot(dc_db, np.transpose(previous_activation))

        return dc_db, dc_dw
    
    def calc_cost(self, x, y):
        return self.cost(x, y)
    
    def calc_next_cost(self, db, weight):
        # dc_a(L-1)
        sum_weight = np.array([np.sum(weight, 0)])
        return np.transpose(sum_weight) * db
    
    @staticmethod
    def update_network(network, average):
        for layer, nudge in zip(network.layers, average):
            layer.bias = np.add(layer.bias, nudge[0])
            layer.weight = np.add(layer.weight, nudge[1])
        return network



class MiniBatch():
    def __init__(self, episode_length, num_layer):
        self.episode_length = episode_length
        self.average = [[] for _ in range(num_layer)] # [[db, dw], [L - 1], ...]

    def running_total(self, db, dw, index):
        # for layer add db, dw to running total for layer
        try:
            db_runnning, dw_runnning = self.average[index]
            db_runnning += db
            db_runnning = db_runnning / 2

            dw_runnning += dw 
            dw_runnning = dw_runnning / 2
            
            self.average[index] = [db_runnning, dw_runnning]

        except:
            # first entry at index
            self.average[index] = [db, dw]

