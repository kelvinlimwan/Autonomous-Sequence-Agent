from template import Agent
import random
import torch
from torch import nn
import copy
import torch.optim as optim

from collections import defaultdict

BOARD = [['jk','2s','3s','4s','5s','6s','7s','8s','9s','jk'],
         ['6c','5c','4c','3c','2c','ah','kh','qh','th','ts'],
         ['7c','as','2d','3d','4d','5d','6d','7d','9h','qs'],
         ['8c','ks','6c','5c','4c','3c','2c','8d','8h','ks'],
         ['9c','qs','7c','6h','5h','4h','ah','9d','7h','as'],
         ['tc','ts','8c','7h','2h','3h','kh','td','6h','2d'],
         ['qc','9s','9c','8h','9h','th','qh','qd','5h','3d'],
         ['kc','8s','tc','qc','kc','ac','ad','kd','4h','4d'],
         ['ac','7s','6s','5s','4s','3s','2s','2h','3h','5d'],
         ['jk','ad','kd','qd','td','9d','8d','7d','6d','jk']]

#Store dict of cards and their coordinates for fast lookup.
COORDS = defaultdict(list)
for row in range(10):
    for col in range(10):
        COORDS[BOARD[row][col]].append((row,col))

class ReplayMemory(object):
    """docstring for ReplayMemory"""
    def __init__(self, size):
        self.size = size
        self.memory = []
        self.position = 0

    def push(self, state, reward):
        if len(self.memory) < self.size:
            self.memory.append(None)

        self.memory[self.position] = (state, reward)
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
class Net(nn.Module):
    def __init__(self, _id):
        super(Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(100, 50),
            nn.Linear(50, 1)
        )
        self.load(_id)
        self.criterion = nn.MSELoss()
        self.opti = optim.Adam(self.network.parameters(), lr = 2e-5)
        

    def forward(self, state, opp_colour):
        input = self.to_numbers(state, opp_colour)
        # input_t = torch.tensor(input).float()
        # print(input_t.shape)
        return self.network(torch.tensor([input]).float())

    def train(self, xs, ys, opp_colour):
        xs = [self.to_numbers(x, opp_colour) for x in xs]

        xs_t = torch.tensor(xs).float()
        ys_t = torch.tensor(ys)

        self.opti.zero_grad()
        logits = self.network(xs_t)
        loss = self.criterion(logits.squeeze(-1), ys_t.float())
        loss.backward()
        self.opti.step()

    
    def save(self, id):
        torch.save(self.state_dict(), 'agent_{}.dat'.format(id))

    def load(self, id):
        self.load_state_dict(torch.load('agent_{}.dat'.format(id)))

    def to_numbers(self, state, opp_colour):
        output = []
        for col in state:
            for row in col:
                if row == "_":
                    output.append(0)
                    continue
                if row == opp_colour:
                    output.append(-1)
                    continue
                
                output.append(1) 

        return output

    def get_future_max(self, state, opp_colour):
        state = self.to_numbers(state, opp_colour)

        successors = []
        for i in range(len(state)):
            state_copy = copy.deepcopy(state)
            if state[i] == -1:
                state_copy[i] = 0
            
            if state[i] == 0:
                state_copy[i] = 1
            successors.append(state_copy)
        

        successors.sort(key=lambda s: self.network(torch.tensor([s]).float()), reverse = True)

        
        return self.network( torch.tensor([successors[0]]).float())


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.policy_net = Net(_id)

        self.memory = ReplayMemory(32)
        self.batch_size = 8

        self.alpha = 0.2
        self.gemma = 0.9

    def SelectAction(self, actions, game_state):
        self.last_score = game_state.agents[self.id].score
        self.opp_colour = game_state.agents[self.id].opp_colour
        gs_copy = copy.deepcopy(game_state)

        best_action = actions[0]
        best_score = self.policy_net.forward(self.update_state(best_action, gs_copy), self.opp_colour)

        for a in actions[1:]:
            gs_copy = copy.deepcopy(game_state)
            score = self.policy_net.forward(self.update_state(a, gs_copy), self.opp_colour)

            if score > best_score:
                score = best_score
                best_action = a

        # return random.choice(best_action)
        return best_action
    
    
    def update_model(self, game_state):
        
        old_q = self.policy_net(game_state.board.chips, self.opp_colour)
        reward = game_state.agents[self.id].score - self.last_score
        future_max = self.policy_net.get_future_max(game_state.board.chips, self.opp_colour)
        score = old_q + self.alpha * (reward + self.gemma * future_max - old_q)

        self.memory.push(game_state.board.chips, score)

        if len(self.memory.memory) >= self.batch_size:
            batch = self.memory.sample(self.batch_size)

            xs, ys = self.make_x_y(batch)

            self.policy_net.train(xs, ys, self.opp_colour)

    def update_state(self, action, gs):
        agent = gs.agents[self.id]
        chips = gs.board.chips

        if action['type'] == 'trade':
            return chips

        r,c = action['coords']
        if action['type']=='place':
            chips[r][c] = agent.colour
            
        elif action['type']=='remove':
            chips[r][c] = '_'

        return chips

    def make_x_y(self, batch):
        xs = []
        ys = []

        for state, score in batch:
            xs.append(state)
            ys.append(score)
        return xs, ys

    def save_net(self):
        self.policy_net.save(self.id)