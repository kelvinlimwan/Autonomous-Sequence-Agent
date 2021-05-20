from template import Agent
from copy import deepcopy as dc

TRADE = 'trade'
PLACE = 'place'
REMOVE = 'remove'
EMPTY = '_'

class myAgent(Agent):

    def __init__(self, _id):
        super().__init__(_id)
        # self.id = _id


    def SelectAction(self, actions, game_state):
        '''
        Given a set of available actions for the agent to execute, and
        a copy of the current game state (including that of the agent),
        select one of the actions to execute.
        actions: actions that you are allowed to take this round- list of dictionaries
        game_state: sequence state object in sequence_model
        '''

        chips = game_state.board.chips
        draft = game_state.board.draft
        player = game_state.agents[self.id]
        hand = player.hand
        colour = player.colour
        state = (chips, hand, colour)

class MDP():

    def isTerminal(self, state):
        # returns True if state is terminal

    def execute(self, state, action):
        # returns (newState, reward)

    def getReward(self, state, action):
        # returns reward of performing action in state

        chips = dc(state[0])
        hand = dc(state[1])
        colour = state(2)

        # placing a chip in heart of board gets reward 1
        heart_coords = [(4, 4), (4, 5), (5, 4), (5, 5)]
        if action['type'] == PLACE and action['coord'] in heart_coords:
            return 1

        


    def getNextState(self, state, action):

        nextChips = dc(state[0])
        nextHand = dc(state[1])
        colour = state[2]

        if action['type'] == PLACE:
            row = action['coords'][0]
            col = action['coords'][1]
            nextChips[row][col] = colour
        elif action['type'] == REMOVE:
            row = action['coords'][0]
            col = action['coords'][1]
            nextChips[row][col] = EMPTY

        nextHand.remove(action['play_card'])
        nextHand.append(action['draft_card'])

        return (nextChips, nextHand, colour)



class Node():

    def __init__(self, chips, hand, draft, value, ):







def SelectAction(self, actions, game_state):
    player = game_state.agents[self.id]
    hand = player.hand
    colour = player.colour
    draft = game_state.board.draft
    chips = game_state.board.chips
    # state = (hand, chips)

    # when the player hasn't traded yet in this round, check if there any trade actions;
    # if so, choose the one that has the lowest  trade heuristic
    if not player.trade:
        trade_actions = []
        for action in actions:
            if action['type'] == TRADE:
                trade_actions.append(action)
        if trade_actions:  # when list is not empty- when we have at least one dead card
            trade_actions_heuristics = {}
            for trade_action in trade_actions:
                trade_actions_heuristics[trade_action] = TradeHeuristic()

def getReward(self, state, action, player):

    clr, sclr = plr_state.colour, plr_state.seq_colour
    oc, os = plr_state.opp_colour, plr_state.opp_seq_colour

    heart_coords = [(4, 4), (4, 5), (5, 4), (5, 5)]
    # if placing in heart of board, reward is 5
    if action['type'] == PLACE and action['coord'] in heart_coords:
        return 5

