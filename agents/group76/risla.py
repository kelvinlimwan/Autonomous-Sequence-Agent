import copy
import math
from template import Agent
import random
from Sequence.sequence_model import COORDS, SequenceGameRule as GameRule


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectAction(self, actions, game_state):
        agent = game_state.agents[self.id]
        # print(self.id)
        # print(game_state.agents[self.id])
        s0 = copy.deepcopy(game_state)
        root = MCTSNode(s0, agent)   
        for i in range(50):
            node = root
            while (not node.can_expand()) and (not node.is_terminal()):
                node = self.select_child(node)   

            if node.can_expand():
                node = node.add_child(agent)

            reward = self.simulate(node, self.id)

            while node is not None:
                node.backpropagate(reward)

        best_move = None
        best_value = -1.0
        for child in root.children:
            child_value = child.value
            if child_value > best_value:
                best_value = child_value
                best_move = child.move
        return best_move
        # return actions[3]
        # return root.unvisited_moves[3]

    def simulate(self, node, agentId):
        reward = 0
        while not node.is_terminal():
            #choose an action to execute
            action = random.choice(node.unvisited_moves)
            
            # execute the action
            node.game_rule.update(node.game_rule, action)
            
            # get the score
            agent = node.game_rule.current_game_state.agents[agentId]
            print(agent.score)
            reward = agent.score
            
        return reward

    def select_child(self, node):
        print('inside select_child')
        total_visits = sum(child.visits for child in node.children)

        best_score = -1
        best_child = None
        for child in node.children:
            score = self.uct_score(
                total_visits,
                child.visits,
                child.score,
                self.exploration_weight)
            if score > best_score:
                best_score = self.uct_score
                best_child = child
        return best_child

    def uct_score(total_visits, visits, score, exploration_weight):
        exploration = math.sqrt(math.log(total_visits) / visits)
        return score + exploration_weight * exploration

class MCTSNode(object):
    def __init__(self, game_state, agent, parent=None, move=None, exploration_weight=1):
        self.exploration_weight = exploration_weight
        self.game_state = game_state        
        self.parent = parent
        self.move = move
        self.children = []  # children of each node
        self.visits = 0
        self.value = 0.00
        self.game_rule = GameRule(4)
        self.unvisited_moves = self.game_rule.getLegalActions(self.game_rule, self.game_state, agent)

    def add_child(self, agent):
        index = random.randint(0, len(self.unvisited_moves) - 1) 
        new_move = self.unvisited_moves.pop(index)        
        self.game_rule.update(self.game_rule, new_move)
        new_game_state = self.game_rule.current_game_state
        new_node = MCTSNode(new_game_state, self, new_move)

        self.children.append(new_node)
        return new_node

    def can_expand(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return self.game_rule.gameEnds()

    def backpropagate(self, reward):
        self.visits += 1.
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    

    