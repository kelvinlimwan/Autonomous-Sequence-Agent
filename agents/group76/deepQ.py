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

# class written for storing training examples
# if overflow, new ones replace the old ones
# First in First out
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

        # the neural network structure
        self.network = nn.Sequential(
            nn.Linear(100, 50),
            nn.Linear(50, 25),
            nn.Linear(25, 1),

        )

        # if start from 0, comment below out
        self.load(_id)
        #########################################

        self.criterion = nn.MSELoss()
        self.opti = optim.Adam(self.network.parameters(), lr = 2e-5)
        


    def forward(self, state, opp_colour):
        input = self.to_numbers(state, opp_colour)

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

    # save parameters
    def save(self, id):
        torch.save(self.state_dict(), 'agent_{}.dat'.format(id))

    # load parameters
    def load(self, id):
        self.load_state_dict(torch.load('agent_{}.dat'.format(id)))

    # convert colour symbols to numbers
    # self_colour and jokers -> 1
    # empty -> 0
    # opp_clour -> -1
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

    # generate all possible future states
    # empty -> self_colour
    # opp -> empty
    # return the one with the highest q-value
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
        # the neural network evaluating chips returns a q-value for this state
        self.policy_net = Net(_id)

        # replay memory storing the (state, reward) pairs
        self.memory = ReplayMemory(32)

        # start training at NO.32 move 
        self.batch_size = 8

        self.alpha = 0.1
        self.gemma = 0.9

    def SelectAction(self, actions, game_state):
        self.last_score = game_state.agents[self.id].score
        self.colour = game_state.agents[self.id].colour
        self.opp_colour = game_state.agents[self.id].opp_colour

        # exploration vs exploitation for better training result
        # otherwise, it is a greedy agent
        # if random.random() < 0.3:
        #     return random.choice(actions)


        # for each action in the action list
        # find the state with the highest q-value 
        gs_copy = copy.deepcopy(game_state)
        best_action = actions[0]

        action_state = self.update_action(best_action, gs_copy)
        # evluating the draft card by generating all possible states 
        # after playing this draft card
        # best_score = self.policy_net(action_state, self.opp_colour)

        draft_states = self.update_draft(best_action, action_state)
        best_score =  max([self.policy_net(s, self.opp_colour) for s in draft_states])
        
        # find the best action
        for a in actions[1:]:
            gs_copy = copy.deepcopy(game_state)

            action_state = self.update_action(best_action, gs_copy)
            # score = self.policy_net(action_state, self.opp_colour)

            draft_states = self.update_draft(best_action, action_state)
            score = max([self.policy_net(s, self.opp_colour) for s in draft_states]) 

            # update if there is a better action
            if score > best_score:
                score = best_score
                best_action = a

        # return the best action
        return best_action
    
    # function updates the network
    def update_model(self, game_state):
        
        # derive the old_q
        old_q = self.policy_net(game_state.board.chips, self.opp_colour)
        # reward is the score difference of before and action
        reward = game_state.agents[self.id].score - self.last_score
        # find future maximum in the successors of the resulting state
        future_max = self.policy_net.get_future_max(game_state.board.chips, self.opp_colour)

        # q-learning formula
        score = old_q + self.alpha * (reward + self.gemma * future_max - old_q)

        # push it to the replaymemory
        self.memory.push(game_state.board.chips, score)

        # check if there are enough training examples
        if len(self.memory.memory) >= self.batch_size:
            batch = self.memory.sample(self.batch_size)

            xs, ys = self.make_x_y(batch)

            self.policy_net.train(xs, ys, self.opp_colour)

    # function returns the chips of the resulting state after the action
    def update_action(self, action, gs):
        chips = gs.board.chips

        if action['type'] == 'trade':
            return chips

        r,c = action['coords']
        if action['type']=='place':
            chips[r][c] = self.colour
            
        elif action['type']=='remove':
            chips[r][c] = '_'

        return chips
    
    # function finds all possible states after playing the draft card in the action
    def update_draft(self, action, chips):
        possible_states = []

        if action['draft_card'] == None:
            possible_states.append(chips)
            return possible_states
        
        draft_card = action['draft_card']

        if draft_card in COORDS:
            for r, c in COORDS[draft_card]:
                chips_copy = copy.deepcopy(chips)
                if chips[r][c] == '_':
                    chips_copy[r][c] = self.colour
                    possible_states.append(chips_copy)

        elif draft_card in ['jd','jc']: #two-eyed jacks
            for r in range(10):
                for c in range(10):
                    if chips[r][c]=='_':
                        chips_copy = copy.deepcopy(chips)
                        chips_copy[r][c] = self.colour
                        possible_states.append(chips_copy)
        
        elif draft_card in ['jh','js']: #one-eyed jacks
            for r in range(10):
                for c in range(10):
                    if chips[r][c]==self.opp_colour:
                        chips_copy = copy.deepcopy(chips)
                        chips_copy[r][c] = '_'
                        possible_states.append(chips_copy)

        if len(possible_states) == 0:
            possible_states.append(chips)
            return possible_states
        return possible_states

    # helper function makes x_batch and y_batch
    def make_x_y(self, batch):
        xs = []
        ys = []

        for state, score in batch:
            xs.append(state)
            ys.append(score)
        return xs, ys

    def save_net(self):
        self.policy_net.save(self.id)