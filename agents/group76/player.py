from template import Agent
import random


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    # Given a set of available actions for the agent to execute, and
    # a copy of the current game state (including that of the agent),
    # select one of the actions to execute.
    def SelectAction(self, actions, game_state):
        return random.choice(actions)