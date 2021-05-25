from template import Agent
from copy import deepcopy as dc
from random import choice
from collections import defaultdict

TRADE = 'trade'
PLACE = 'place'
REMOVE = 'remove'
EMPTY = '_'
RED = 'r'
BLU = 'b'
ALPHA = 0.4
GAMMA = 0.9

HEART_COORDS = [(4,4),(4,5),(5,4),(5,5)]

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

CARDS = ['as', 'ah', 'ac', 'ad', '2s', '2h', '2c', '2d', '3s', '3h', '3c', '3d', '4s', '4h', '4c', '4d',
         '5s', '5h', '5c', '5d', '6s', '6h', '6c', '6d', '7s', '7h', '7c', '7d', '8s', '8h', '8c', '8d',
         '9s', '9h', '9c', '9d', 'ts', 'th', 'tc', 'td', 'js', 'jh', 'jc', 'jd', 'qs', 'qh', 'qc', 'qd',
         'ks', 'kh', 'kc', 'kd']

#Store dict of cards and their coordinates for fast lookup.
COORDS = defaultdict(list)
for row in range(10):
    for col in range(10):
        COORDS[BOARD[row][col]].append((row,col))

class State:
    def __init__(self, chips, hand, draft):
        self.chips = chips
        self.hand = hand
        self.draft = draft

class myAgent(Agent):

    def __init__(self, _id):
        super().__init__(_id)
        # self.id = _id
        self.colour = BLU if _id % 2 else RED
        self.oppColour = RED if _id % 2 else BLU
        self.qValues = {}
        self.alpha = ALPHA
        self.gamma = GAMMA

    def SelectAction(self, actions, game_state):
        '''
        Given a set of available actions for the agent to execute, and
        a copy of the current game state (including that of the agent),
        select one of the actions to execute.
        actions: actions that you are allowed to take this round- list of dictionaries
        game_state: sequence state object in sequence_model
        '''

        chips = game_state.board.chips
        hand = game_state.agents[self.id].hand
        draft = game_state.board.draft
        state = State(chips, hand, draft)

        temp_dict = {}

        bestQ = float('-inf')
        bestActions = []

        for action in actions:
            reward = self.getReward(state, action)
            nextState = self.getNextState(state, action)
            newQ = self.update(state, action, reward, nextState)
            if newQ != 0:
                temp_dict[(state, tuple(action))] = newQ

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
        copyChips = dc(state.chips)

        # All joker spaces become player chips for the purposes of reward allocation.
        for r, c in COORDS['jk']:
            copyChips[r][c] = self.colour

        if action['type'] == PLACE:
            r, c = action['coords']

            # reward of 5 if we place a chip in heart of board
            if (r, c) in HEART_COORDS:
                return 10

            # reward on number of own chips - number of opp chips around (r, c) by 5 rows and columns either sides
            count = 0
            for row in range(max(0, r - 5), min(9, r + 5)):
                for col in range(max(0, c - 5), min(9, c + 5)):
                    if copyChips[row][col] == self.colour:
                        count += 1
                    elif copyChips[row][col] == self.oppColour:
                        count -= 0.5

            return count

        elif action['type'] == REMOVE:
            r, c = action['coords']

            # reward of 5 if we remove an opp chip in heart of board and there is at least 3 of opp colour in heart
            if (r, c) in HEART_COORDS:
                heart_count = 0
                for coord in HEART_COORDS:
                    for pos in copyChips:
                        if pos == self.oppColour:
                            heart_count += 1
                if heart_count >= 3:
                    return 10

            # reward on number of opp chips - number of own chips around (r, c) by 5 rows and columns either sides
            count = 0
            for row in range(max(0, r - 5), min(9, r + 5)):
                for col in range(max(0, c - 5), min(9, c + 5)):
                    if copyChips[row][col] == self.colour:
                        count -= 1
                    elif copyChips[row][col] == self.oppColour:
                        count += 0.5
            print("REMOVE REWARD:" + str(count))
            return count

        else:   #action['type'] == TRADE
            # reward of 10 if draft card is wild card
            if action['draft_card'] is not None and action['draft_card'][0] == 'j':
                return 10
            else:
                return 0

    def getQValue(self, state, action):
        '''
        Return the qValue of the (state, action) pair
        '''
        if (state, tuple(action)) in self.qValues:
            return self.qValues[(state, tuple(action))]
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
        nextChips = dc(state.chips)
        nextHand = dc(state.hand)
        nextDraft = dc(state.draft)

        if action['type'] == PLACE:
            r, c = action['coords']
            play_card = action['play_card']
            draft_card = action['draft_card']
            nextChips[r][c] = self.colour
            nextHand.remove(play_card)
            nextHand.append(draft_card)
            nextDraft.remove(draft_card)
            nextDraft.append(choice(CARDS)) # add a random card (could be already used)

        elif action['type'] == REMOVE:
            r, c = action['coords']
            play_card = action['play_card']
            draft_card = action['draft_card']
            nextChips[r][c] = EMPTY
            nextHand.remove(play_card)
            nextHand.append(draft_card)
            nextDraft.remove(draft_card)
            nextDraft.append(choice(CARDS))  # add a random card (could be already used)

        else: # action['type'] == TRADE
            play_card = action['play_card']
            draft_card = action['draft_card']
            if play_card is not None and draft_card is not None:
                nextHand.remove(play_card)
                nextHand.append(draft_card)
                nextDraft.remove(draft_card)
                nextDraft.append(choice(CARDS))  # add a random card (could be already used)

        return State(nextChips, nextHand, nextDraft)

    def getPossibleActions(self, state):
        chips = dc(state.chips)
        hand = dc(state.hand)
        draft = dc(state.draft)

        actions = []

        # add trade actions
        for handCard in hand:
            if handCard[0] != 'j':  # can't trade a jack
                freeSpaces = 0  # counts number of free spaces in board for handCard
                for r,c in COORDS[handCard]:
                    if chips[r][c] == EMPTY:
                        freeSpaces += 1
                if not freeSpaces:  # if handCard has no space in board (dead)
                    for draftCard in draft:
                        actions.append({'play_card': handCard, 'draft_card': draftCard, 'type': TRADE, 'coords': None})
        if len(actions):    # if can trade, give option to forego trade
            actions.append({'play_card': None, 'draft_card': None, 'type': TRADE, 'coords': None})

        # add play and remove actions
        for handCard in hand:
            if handCard in ['jc', 'jd']:  # two-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if chips[r][c] == EMPTY:
                            for draftCard in draft:
                                actions.append({'play_card': handCard, 'draft_card': draftCard, 'type': PLACE, 'coords': (r, c)})


            elif handCard in ['js', 'jh']:  # one-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if chips[r][c] == self.oppColour:
                            for draftCard in draft:
                                actions.append({'play_card': handCard, 'draft_card': draftCard, 'type': REMOVE, 'coords': (r, c)})

            else: #regular cards
                for r,c in COORDS[handCard]:
                    if chips[r][c] == EMPTY:
                        for draftCard in draft:
                            actions.append({'play_card': handCard, 'draft_card': draftCard, 'type': PLACE, 'coords':(r,c)})

        return actions