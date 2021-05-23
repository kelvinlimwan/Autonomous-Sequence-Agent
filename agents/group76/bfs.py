from template import Agent
import random
import numpy as np
import copy

from collections import defaultdict

# CONSTANTS ----------------------------------------------------------------------------------------------------------#

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

# class represents a search node for BFS
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
    
    # function returns the sucessors of the current node
    def generate_successor(self):
        successors = []

        # choose a card to play
        for play_card in self.hand:
            hand_copy = copy.deepcopy(self.hand)
            hand_copy.remove(play_card)

            # choose a card to draft
            for draft_card in self.draft:
                draft_copy = copy.deepcopy(self.draft)
                draft_copy.remove(draft_card)
                hand_copy.append(draft_card)

                # play the chosen play_card

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
 

    # count the number of moves to occupy the heart of board
    def to_occupy_heart(self, chips, opp_coulor):
        heart = [chips[4][4], chips[4][5], chips[5][4], chips[5][5]]

        # if there is a opp_colour in heart
        # impossible to do so
        if opp_coulor in heart:
            return MAX
        return self.num_space(heart)

    # given a sequence candidate
    # count the number of moves to complete it
    def to_complete_seq(self, seq_candidate):
        step = 5
        for i in range(len(seq_candidate) - 4):
            five_chips = seq_candidate[i:i+5]

            if self.num_space(five_chips) < step:
                step = self.num_space(five_chips)
        return step

    # count the number of space in the give sequence candidate
    def num_space(self, seq):
        num_space = 0
        for i in seq:
            if i == '_':
                num_space += 1
        return num_space

    # create lines vertical, horizontal, plus 2 diagnals
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
    
    # go through each line, create candidates where we can 
    # complete a sequence
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

    # return reward for this state and h_value 
    def evaluate(self):
        lines = self.create_lines(self.chips)
        seq_candidates = self.create_sequence_candidates(lines, self.opp_colour)
        

        steps_to_complete_sep = [self.to_complete_seq(i) for i in seq_candidates]
        steps_to_complete_sep.sort()
        # the min number of steps to complete 2 sequences
        step_to_complete_2_seq = steps_to_complete_sep[0] + steps_to_complete_sep[1]
        # mean number of steps to complete a sequence
        mean_step_to_complete = sum(steps_to_complete_sep)/len(steps_to_complete_sep)

        step_to_occupy_heart = self.to_occupy_heart(self.chips, self.opp_colour)

        # h_value =  min step to win + mean step to complete a sequence
        h_value = min(step_to_occupy_heart, step_to_complete_2_seq) + mean_step_to_complete
        # one sequence is 1 point, heart board worth 3 points by rules
        reward = self.get_reward(step_to_occupy_heart, steps_to_complete_sep)

        return  reward, h_value

    def get_reward(self, step_to_occupy_heart, step_to_complete_seq):
        return 3*int(step_to_occupy_heart == 0) + sum([int(i==0) for i in step_to_complete_seq])


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
    
    def SelectAction(self,actions,game_state):
        
        # breadth first search through the leaf nodes
        # depth is limited to 2
        DEPTH_LIMIT = 2

        # convert actions to search nodes
        temp = [(self.update_state(a, copy.deepcopy(game_state), self.id),a) for a in actions ]
        open_list = [searchNode(state, a, 1) for (state, a) in temp]


        # find the best search node in the open_list
        # s1 is strictly better than s2 if s1 has more rewards
        # s1 is better than s2 if they have a same reward, but s1 has a lower h_value
        best_reward, best_h_value = open_list[0].evaluate()
        best_node = open_list[0]
        visited = [best_node]

        while len(open_list)>0:
            node = open_list[0]
            open_list = open_list[1:]

            if node.chips in visited:
                continue
            
            visited.append(node.chips)

            if node.depth != DEPTH_LIMIT:
                open_list += node.generate_successor()
            
            n_reward, n_h_value = node.evaluate()
            if (n_reward > best_reward) or (n_reward == best_reward and n_h_value < best_h_value):
                best_reward = n_reward
                best_h_value = n_h_value
                best_node = node


        return best_node.action
    
    # function execute the action and derive the resulting state
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



    

