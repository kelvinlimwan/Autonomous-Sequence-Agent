from template import Agent
import random
import numpy as np
import copy

from Sequence.sequence_model import SequenceGameRule as GameRule


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        

    
    def SelectAction(self,actions,game_state):
        self.game_state = game_state
        self.game_state.deck.cards = []
        actions.sort(key=self.evaluate_action)
        return actions[0]
    
    def evaluate_action(self, action):
        self_colour = self.game_state.agents[self.id].colour
        opp_colour = self.game_state.agents[self.id].opp_colour

        copy_state = copy.deepcopy(self.game_state)

        game_rule = GameRule(4)
        new_state = game_rule.generateSuccessor(copy_state, action, self.id)

        state_score = self.evaluate_state(self_colour, opp_colour, new_state.board.chips)
        return 0

    
    def evaluate_state(self, self_colour, opp_colour, chips):
        return self.heuristic(chips, opp_colour) - self.heuristic(chips, self_colour)*0.5

    def heuristic(self, chips, opp_colour):

        seq_candidates = []

        for line in  self.create_lines(chips):
            opps = np.where(line == opp_colour)[0]
            if len(opps) != 0:
                start = 0
                for opp_index in opps:
                    if start == 0:
                        temp = line[start: int(opp_index)]
                    else:
                        temp = line[start+1: opp_index]
                    start = opp_index

                    if len(temp) >= 5:
                        seq_candidates.append(temp)
                temp = line[start: len(line)]
                if len(temp)>=5:
                    seq_candidates.append(temp)
            else:
                seq_candidates.append(line)

        
        num_candidate = len(seq_candidates)

        num_step_to_complete = [self.step_to_complete(i) for i in seq_candidates]
        num_step_to_complete.sort()
        num_step_to_win = num_step_to_complete[0] + num_step_to_complete[1]

        if self.heart_board(chips, opp_colour) == None:
            return num_step_to_win + 1/(1+num_candidate)
        return min(num_step_to_win, self.heart_board(chips, opp_colour)) + 1/(1+num_candidate)

    def heart_board(self, chips, opp_coulor):
        heart = [chips[4][4], chips[4][5], chips[5][4], chips[5][5]]
        if opp_coulor in heart:
            return None
        
        return self.num_space(heart)

    def step_to_complete(self, seq_candidate):
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
        

