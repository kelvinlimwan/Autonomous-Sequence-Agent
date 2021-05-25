from template import Agent
import random
import numpy as np
import copy

from Sequence.sequence_model import SequenceGameRule as GameRule

from collections import defaultdict

# CONSTANTS ----------------------------------------------------------------------------------------------------------#

BOARD = [['jk', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'jk'],
         ['6c', '5c', '4c', '3c', '2c', 'ah', 'kh', 'qh', 'th', 'ts'],
         ['7c', 'as', '2d', '3d', '4d', '5d', '6d', '7d', '9h', 'qs'],
         ['8c', 'ks', '6c', '5c', '4c', '3c', '2c', '8d', '8h', 'ks'],
         ['9c', 'qs', '7c', '6h', '5h', '4h', 'ah', '9d', '7h', 'as'],
         ['tc', 'ts', '8c', '7h', '2h', '3h', 'kh', 'td', '6h', '2d'],
         ['qc', '9s', '9c', '8h', '9h', 'th', 'qh', 'qd', '5h', '3d'],
         ['kc', '8s', 'tc', 'qc', 'kc', 'ac', 'ad', 'kd', '4h', '4d'],
         ['ac', '7s', '6s', '5s', '4s', '3s', '2s', '2h', '3h', '5d'],
         ['jk', 'ad', 'kd', 'qd', 'td', '9d', '8d', '7d', '6d', 'jk']]

# Store dict of cards and their coordinates for fast lookup.
COORDS = defaultdict(list)
for row in range(10):
    for col in range(10):
        COORDS[BOARD[row][col]].append((row, col))


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectAction(self, actions, game_state):

        best_action = actions[0]
        best_reward, best_h_value = self.evaluate_action(best_action, game_state)

        for a in actions[1:]:
            reward, h_value = self.evaluate_action(a, game_state)
            if reward > best_reward:
                best_action = a
                best_reward = reward
                best_h_value = h_value
                continue
            if reward == best_reward and h_value < best_h_value:
                best_action = a
                best_reward = reward
                best_h_value = h_value
                continue

        return best_action

    def evaluate_action(self, action, gs):
        self_colour = gs.agents[self.id].colour
        opp_colour = gs.agents[self.id].opp_colour

        s0 = copy.deepcopy(gs)
        chips = s0.board.chips

        if action['type'] == 'place':
            r, c = action['coords']
            chips[r][c] = self_colour
            draft_card = action['draft_card']

            score = self.evaluate_draft(chips, draft_card, self_colour, opp_colour)

        if action['type'] == 'remove':
            r, c = action['coords']
            chips[r][c] = '_'

            draft_card = action['draft_card']
            score = self.evaluate_draft(chips, draft_card, self_colour, opp_colour)

        if action['type'] == 'trade':
            if action['play_card'] == None:
                score = self.evaluate_state(chips, self_colour, opp_colour)
            else:
                draft_card = action['draft_card']
                score = self.evaluate_draft(chips, draft_card, self_colour, opp_colour)

        return score

    def evaluate_draft(self, chips, card, self_colour, opp_colour):
        results = []

        for r, c in COORDS[card]:
            if chips[r][c] == '_':
                chips[r][c] = self_colour
            results.append(self.evaluate_state(chips, self_colour, opp_colour))

        if len(results) == 0:
            return self.evaluate_state(chips, self_colour, opp_colour)

        best_reward, best_h_value = results[0]
        for reward, h_value in results[1:]:
            if reward > best_reward:
                best_reward = reward
                best_h_value = h_value
                continue
            if reward == best_reward and h_value < best_h_value:
                best_reward = reward
                best_h_value = h_value
                continue

        return best_reward, best_h_value

    def evaluate_state(self, chips, self_colour, opp_colour):
        self_reward, self_h = self.heuristic(chips, opp_colour)

        opp_reward, opp_h = self.heuristic(chips, self_colour)
        return self_reward - opp_reward, self_h - opp_h

    def heuristic(self, chips, opp_colour):
        lines = self.create_lines(chips)
        seq_candidates = self.create_sequence_candidates(lines, opp_colour)

        steps_to_complete = [self.to_complete_seq(i) for i in seq_candidates]
        steps_to_complete.sort()
        step_to_win = steps_to_complete[0] + steps_to_complete[1]

        num_complete_seq = sum([int(i == 0) for i in steps_to_complete])
        mean_step_to_complete = sum(steps_to_complete) / len(steps_to_complete)

        step_to_occupt_heart = self.to_occupy_heart(chips, opp_colour)
        if step_to_occupt_heart == None:  # impossible to occupy heart
            return num_complete_seq, step_to_win + mean_step_to_complete
        elif step_to_occupt_heart == 0:
            return 3 + num_complete_seq, mean_step_to_complete
        else:
            return num_complete_seq, min(step_to_occupt_heart, step_to_win) + mean_step_to_complete

    def to_occupy_heart(self, chips, opp_coulor):
        heart = [chips[4][4], chips[4][5], chips[5][4], chips[5][5]]
        if opp_coulor in heart:
            return None

        return self.num_space(heart)

    def to_complete_seq(self, seq_candidate):
        step = 5
        for i in range(len(seq_candidate) - 4):
            five_chips = seq_candidate[i:i + 5]

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
                        temp = line[start + 1: opp_index]
                    start = opp_index

                    if len(temp) >= 5:
                        candidates.append(temp)
                temp = line[start: len(line)]
                if len(temp) >= 5:
                    candidates.append(temp)
            else:
                candidates.append(line)
        return candidates

