# CONSTANTS ----------------------------------------------------------------------------------------------------------#

from collections import defaultdict
import copy
from template import Agent
import random

import numpy as np


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
MAX = 99999
RED     = 'r'
BLU     = 'b'

class searchNode():
    def __init__(self, state, action, depth):
        (hand, draft, chips, colour, opp_colour) = state
        self.action = action
        self.hand = hand
        self.draft = draft
        self.chips = chips
        self.colour = colour
        self.opp_colour = opp_colour
        self.depth = depth

    def generate_successor(self):
        successors = []
        for play_card in self.hand:
            hand_copy = copy.deepcopy(self.hand)
            hand_copy.remove(play_card)
            for draft_card in self.draft:
                draft_copy = copy.deepcopy(self.draft)
                draft_copy.remove(draft_card)
                hand_copy.append(draft_card)

                if play_card in ['jd','jc']: #two-eyed jacks
                    # continue
                    for r in range(10):
                        for c in range(10):
                            if self.chips[r][c]=='_':
                                chips_copy = copy.deepcopy(self.chips)
                                chips_copy[r][c] = self.colour
                                state = (hand_copy, draft_copy, chips_copy, self.colour, self.opp_colour)
                                node = searchNode(state,self.action, self.depth+1)
                                successors.append(node)
               
                elif play_card in ['jh','js']: #one-eyed jacks
                    # continue
                    for r in range(10):
                        for c in range(10):
                            if self.chips[r][c] == self.opp_colour:
                                chips_copy = copy.deepcopy(self.chips)
                                chips_copy[r][c] = '_'
                                state = (hand_copy, draft_copy, chips_copy, self.colour, self.opp_colour)
                                node = searchNode(state, self.action, self.depth+1)
                                successors.append(node)

                                
                else: #regular cards
                    for r,c in COORDS[play_card]:
                        if self.chips[r][c]=='_':
                            chips_copy = copy.deepcopy(self.chips)
                            chips_copy[r][c] = self.colour
                            state = (hand_copy, draft_copy, chips_copy, self.colour, self.opp_colour)
                            node = searchNode(state, self.action, self.depth+1)
                            successors.append(node)
        return successors

    def get_h_value(self):
        lines = self.create_lines(self.chips)
        seq_candidates = self.create_sequence_candidates(lines, self.opp_colour)

        steps_to_complete_sep = [self.to_complete_seq(i) for i in seq_candidates]
        steps_to_complete_sep.sort()
        step_to_complete_2_seq = steps_to_complete_sep[0] + steps_to_complete_sep[1]

        mean_step_to_complete = sum(steps_to_complete_sep)/len(steps_to_complete_sep)

        step_to_occupy_heart = self.to_occupy_heart(self.chips, self.opp_colour)

        h_value = min(step_to_occupy_heart, step_to_complete_2_seq) + mean_step_to_complete
        return h_value
    
    def create_lines(self, chips):
        chips = np.asarray(chips)
        lines = []

        for i in chips:
            lines.append(i)
        
        for i in np.transpose(chips):
            lines.append(i)
        
        d1 = [(0, i) for i in range(10)] + [(i, 0) for i in range(10)]
        d2 = [(9, i) for i in range(10)] + [(i, 0) for i in range(10)]

        for (x, y) in d1:
            temp = []
            while (x < 10 and y < 10):
                temp.append(chips[y][x])
                x += 1
                y += 1
            if len(temp) >= 5:
                lines.append(temp)
        
        for (x, y) in d2:
            temp = []
            while (x < 10 and y < 10):
                temp.append(chips[y][x])
                x += 1
                y -= 1
            if len(temp) >= 5:
                lines.append(temp)
        return lines
    
    def create_sequence_candidates(self, lines, opp_colour):
        candidates = []

        for line in lines:
            opps = np.where(line == opp_colour)[0]
            if len(opps) != 0:
                start = 0
                for opp_index in opps:
                    if start == 0:
                        temp = line[start: opp_index]
                    else:
                        temp = line[start+1: opp_index]
                    start = opp_index

                    if len(temp) >= 5:
                        candidates.append(temp)
                temp = line[start: len(line)]
                if len(temp)>=5:
                    candidates.append(temp)
            else:
                candidates.append(line)
        return candidates

    def to_complete_seq(self, seq_candidate):
        step = 5
        for i in range(len(seq_candidate) - 4):
            five_chips = seq_candidate[i:i+5]

            if self.num_space(five_chips) < step:
                step = self.num_space(five_chips)
        return step

    def num_space(self, seq):
        num_space = 0
        for i in seq:
            if i == '_':
                num_space += 1
        return num_space

    def to_occupy_heart(self, chips, opp_coulor):
        heart = [chips[4][4], chips[4][5], chips[5][4], chips[5][5]]
        if opp_coulor in heart:
            return MAX
        
        return self.num_space(heart)

    def is_goal_state(self): #Game ends if a team has formed at least 2 sequences, or if the deck is empty.
        lines = self.create_lines(self.chips)
        if lines == 2 or self.draft == 0:
            return True
        else:
            return False


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
    
    def SelectAction(self,actions,game_state):
        temp = [(self.update_state(a, copy.deepcopy(game_state), self.id),a) for a in actions ]
        node_list = [searchNode(state, a, 1) for (state, a) in temp]
        h_value = 0
        if len(node_list)>0:
            current_node = random.choice(node_list)
            h_value = current_node.get_h_value()
            current_node = self.hill_climbing(current_node, h_value)
        return current_node.action

    def hill_climbing(self, initial, h_value):
        best_state = initial        
        while not best_state is None and best_state.is_goal_state():
            best_state, h_value = self.improve(best_state, h_value)
        return best_state
    
    def improve(self, state, h_value):
        queue = []
        queue.append(state)
        closed = set()
        while len(queue) > 0:
            new_sate = queue.pop()
            if new_sate not in closed:
                closed.add(new_sate)
                new_h_value = new_sate.get_h_value()
                if new_h_value < h_value:                    
                   return new_sate, new_h_value
                node_list = state.generate_successor() 
                for node in node_list:
                    queue.append(node)
        # while len(closed) > 0:
        #     new_sate = closed.pop()
        #     new_h_value = new_sate.get_h_value()
        #     if new_h_value < h_value:                    
        #            return new_sate, new_h_value
        return 

    def update_state(self, action, gs, agent_id):
        agent = gs.agents[agent_id]
        hand = agent.hand
        draft = gs.board.draft
        chips = gs.board.chips

        if action['type']=='trade':
            if action['play_card'] == None:
                return hand, draft, chips, agent.colour, agent.opp_colour
            else:
                play_card, draft_card = action['play_card'], action['draft_card']

                hand.remove(play_card)
                hand.append(draft_card)
                draft.remove(draft_card)
                return hand, draft, chips, agent.colour, agent.opp_colour

        r,c = action['coords']
        if action['type']=='place':
            chips[r][c] = agent.colour

            play_card, draft_card = action['play_card'], action['draft_card']

            hand.remove(play_card)
            hand.append(draft_card)
            draft.remove(draft_card)
            return hand, draft, chips, agent.colour, agent.opp_colour
            
        elif action['type']=='remove':
            chips[r][c] = '_'
            
            play_card, draft_card = action['play_card'], action['draft_card']

            hand.remove(play_card)
            hand.append(draft_card)
            draft.remove(draft_card)
            return hand, draft, chips, agent.colour, agent.opp_colour