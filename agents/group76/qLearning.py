from template import Agent
from copy import deepcopy as dc
from random import choice
from collections import defaultdict

TRADE = 'trade'
PLACE = 'place'
REMOVE = 'remove'
JOKER   = '#'
EMPTY = '_'
TRADSEQ = 1
HOTBSEQ = 2
MULTSEQ = 3
TIMEOUT = 1
ALPHA = 0.4
GAMMA  = 0.9
EPSILON = 0.05

HEART_COORDS = [(4,4),(4,5),(5,4),(5,5)]
VERTICAL_COORDS = [(-4,0),(-3,0),(-2,0),(-1,0),(0,0),(1,0),(2,0),(3,0),(4,0)]
HORIZONTAL_COORDS = [(0,-4),(0,-3),(0,-2),(0,-1),(0,0),(0,1),(0,2),(0,3),(0,4)]
DIAGONAL1_COORDS = [(-4,-4),(-3,-3),(-2,-2),(-1,-1),(0,0),(1,1),(2,2),(3,3),(4,4)]
DIAGONAL2_COORDS = [(-4,4),(-3,3),(-2,2),(-1,1),(0,0),(1,-1),(2,-2),(3,-3),(4,-4)]

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

class myAgent(Agent):

    def __init__(self, _id):
        super().__init__(_id)
        # self.id = _id
        self.qValues = {}
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON

    def SelectAction(self, actions, game_state):
        '''
        Given a set of available actions for the agent to execute, and
        a copy of the current game state (including that of the agent),
        select one of the actions to execute.
        actions: actions that you are allowed to take this round- list of dictionaries
        game_state: sequence state object in sequence_model
        '''

        temp_dict = {}

        bestQ = float('-inf')
        bestActions = []

        for action in actions:
            print("before reward")
            reward = self.getReward(game_state, action)
            print("after reward")
            print("before next state")
            nextState = self.getNextState(dc(game_state), action)
            print("after next state")
            newQ = self.update(game_state, action, reward, nextState)
            if newQ != 0:
                temp_dict[(game_state, action)] = newQ

            if newQ > bestQ:
                bestQ = newQ
                bestActions = [action]
            elif newQ == bestQ:
                bestActions.append(action)

        for key, val in temp_dict.items():
            self.qValues[key] = val

        return choice(bestActions)

    def update(self, state, action, reward, nextState):
        '''
        Return an update Q value for (state, action) pair
        '''
        maxFutureQ = self.computeValueFromQValues(nextState)

        return self.getQValue(state, action) + self.alpha * (reward + self.gamma * maxFutureQ - self.getQValue(state, action))

    def getReward(self, state, action):
        '''
        Return reward received by executing action on state
        '''

        copyState = dc(state)
        chips = copyState.board.chips
        colour = copyState.agents[self.id].colour
        oppColour = copyState.agents[self.id].opp_colour

        # All joker spaces become player chips for the purposes of reward allocation.
        for r, c in COORDS['jk']:
            chips[r][c] = colour

        if action['type'] == PLACE:
            r, c = action['coords']

            # reward of 5 if we place a chip in heart of board
            if (r, c) in HEART_COORDS:
                return 6

            # reward on number of own chips - number of opp chips around (r, c) by 5 rows and columns either sides
            count = 0
            for row in range(max(0, r-5), min(9, r+5)):
                for col in range(max(0, c-5), min(9, c+5)):
                    if chips[row][col] == colour:
                        count += 1
                    elif chips[row][col] == oppColour:
                        count -= 1
            return count

        elif action['type'] == REMOVE:
            r, c = action['coords']

            # reward of 5 if we remove an opp chip in heart of board
            if (r, c) in HEART_COORDS:
                return 2

            # reward on number of opp chips - number of own chips around (r, c) by 5 rows and columns either sides
            count = 0
            for row in range(max(0, r - 5), min(9, r + 5)):
                for col in range(max(0, c - 5), min(9, c + 5)):
                    if chips[row][col] == colour:
                        count -= 1
                    elif chips[row][col] == oppColour:
                        count += 1
            return count/3

        else:   #action['type'] == TRADE
            # reward of 10 if draft card is wild card
            if action['draft_card'][0] == 'j':
                return 10


    def getQValue(self, state, action):
        '''
        Return the qValue of the (state, action) pair
        '''
        if (state, action) in self.qValues:
            return self.qValues[(state, action)]
        else:
            return 0

    def computeValueFromQValues(self, state):
        '''
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        '''
        possibleActions = self.getPossibleActions(state)
        if possibleActions:
            maxQ = float('-inf')
            for action in possibleActions:
                q = self.getQValue(state, action)
                maxQ = max(maxQ, q)
            return maxQ
        return 0.0

    def getNextState(self, state, action):
        '''
        Return the state we are at when executing action on state
        '''
        nextState = dc(state)
        nextState.board.new_seq = False
        #print(f"agent id {self.id}")
        plr_state = nextState.agents[self.id]
        plr_state.last_action = action  # Record last action such that other agents can make use of this information.
        reward = 0

        # Update agent state. Take the card in play from the agent, discard, draw the selected draft, deal a new draft.
        # If agent was allowed to trade but chose not to, there is no card played, and hand remains the same.
        card = action['play_card']
        draft = action['draft_card']
        if card:
            plr_state.hand.remove(card)  # Remove card from hand.
            plr_state.discard = card  # Add card to discard pile.
            nextState.deck.discards.append(card)  # Add card to global list of discards (some agents might find tracking this helpful).
            nextState.board.draft.remove(draft)  # Remove draft from draft selection.
            plr_state.hand.append(draft)  # Add draft to player hand.
            nextState.board.draft.extend(nextState.deck.deal())  # Replenish draft selection.

        # If action was to trade in a dead card, action is complete, and agent gets to play another card.
        if action['type'] == TRADE:
            plr_state.trade = True  # Switch trade flag to prohibit agent performing a second trade this turn.
            return nextState

        # Update Sequence board. If action was to place/remove a marker, add/subtract it from the board.
        r, c = action['coords']
        if action['type'] == PLACE:
            nextState.board.chips[r][c] = plr_state.colour
            nextState.board.empty_coords.remove(action['coords'])
            nextState.board.plr_coords[plr_state.colour].append(action['coords'])
        elif action['type'] == REMOVE:
            nextState.board.chips[r][c] = EMPTY
            nextState.board.empty_coords.append(action['coords'])
        else:
            print("Action unrecognised.")

        # Check if a sequence has just been completed. If so, upgrade chips to special sequence chips.
        if action['type'] == PLACE:
            seq, seq_type = self.checkSeq(nextState.board.chips, plr_state, (r, c))
            if seq:
                reward += seq['num_seq']
                nextState.board.new_seq = seq_type
                for sequence in seq['coords']:
                    for r, c in sequence:
                        if nextState.board.chips[r][c] != JOKER:  # Joker spaces stay jokers.
                            nextState.board.chips[r][c] = plr_state.seq_colour
                            try:
                                nextState.board.plr_coords[plr_state.colour].remove(action['coords'])
                            except:  # Chip coords were already removed with the first sequence.
                                pass
                plr_state.completed_seqs += seq['num_seq']
                plr_state.seq_orientations.extend(seq['orientation'])

        plr_state.trade = False  # Reset trade flag if agent has completed a full turn.
        plr_state.agent_trace.action_reward.append((action, reward))  # Log this turn's action and any resultant score.
        plr_state.score += reward
        return nextState

    def getPossibleActions(self, state):
        actions = []
        agent_state = state.agents[self.id]

        # First, give the agent the option to trade a dead card, if they haven't just done so.
        if not agent_state.trade:
            for card in agent_state.hand:
                if card[0] != 'j':
                    free_spaces = 0
                    for r, c in COORDS[card]:
                        if state.board.chips[r][c] == EMPTY:
                            free_spaces += 1
                    if not free_spaces:  # No option to place, so card is considered dead and can be traded.
                        for draft in state.board.draft:
                            actions.append({'play_card': card, 'draft_card': draft, 'type': 'trade', 'coords': None})

            if len(actions):  # If trade actions available, return those, along with the option to forego the trade.
                actions.append({'play_card': None, 'draft_card': None, 'type': 'trade', 'coords': None})
                return actions

        # If trade is prohibited, or no trades available, add action/s for each card in player's hand.
        # For each action, add copies corresponding to the various draft cards that could be selected at end of turn.
        for card in agent_state.hand:
            if card in ['jd', 'jc']:  # two-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if state.board.chips[r][c] == EMPTY:
                            for draft in state.board.draft:
                                actions.append(
                                    {'play_card': card, 'draft_card': draft, 'type': 'place', 'coords': (r, c)})

            elif card in ['jh', 'js']:  # one-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if state.board.chips[r][c] == agent_state.opp_colour:
                            for draft in state.board.draft:
                                actions.append(
                                    {'play_card': card, 'draft_card': draft, 'type': 'remove', 'coords': (r, c)})

            else:  # regular cards
                for r, c in COORDS[card]:
                    if state.board.chips[r][c] == EMPTY:
                        for draft in state.board.draft:
                            actions.append({'play_card': card, 'draft_card': draft, 'type': 'place', 'coords': (r, c)})

        return actions
