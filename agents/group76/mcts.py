from Sequence.sequence_utils import BLU, EMPTY, HOTBSEQ, JOKER, MULTSEQ, RED, TRADSEQ
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
        root = MCTSNode(s0, self.id)   
        for i in range(50):
            node = root
            while (not node.can_expand()) and (not node.is_terminal()):
                node = self.select_child(node)   

            if node.can_expand():
                node = node.add_child(self.id)

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
    def __init__(self, game_state, agentId, parent=None, move=None, exploration_weight=1):
        self.exploration_weight = exploration_weight
        self.game_state = game_state        
        self.parent = parent
        self.move = move
        self.children = []  # children of each node
        self.visits = 0
        self.value = 0.00
        self.game_rule = GameRule(4)
        self.unvisited_moves = self.getLegalActions(self.game_state, agentId)
        # self.unvisited_moves = self.game_rule.getLegalActions(self.game_state, agent)

    def add_child(self, agentId):
        index = random.randint(0, len(self.unvisited_moves) - 1) 
        new_move = self.unvisited_moves.pop(index)     
        self.update(new_move)
        new_game_state = self.game_rule.current_game_state
        new_node = MCTSNode(new_game_state, agentId, self, new_move)

        self.children.append(new_node)
        return new_node

    def can_expand(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return self.game_rule.gameEnds()

    def getNextAgentIndex(self):
        return (self.game_rule.current_agent_index + 1) % self.game_rule.num_of_agent

    def backpropagate(self, reward):
        self.visits += 1.
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def getLegalActions(self, game_state, agent_id):
        actions = []
        agent_state = game_state.agents[agent_id]
        
        #First, give the agent the option to trade a dead card, if they haven't just done so.
        if not agent_state.trade:
            for card in agent_state.hand:
                if card[0]!='j':
                    free_spaces = 0
                    for r,c in COORDS[card]:
                        if game_state.board.chips[r][c]==EMPTY:
                            free_spaces+=1
                    if not free_spaces: #No option to place, so card is considered dead and can be traded.
                        for draft in game_state.board.draft:
                            actions.append({'play_card':card, 'draft_card':draft, 'type':'trade', 'coords':None})
                        
            if len(actions): #If trade actions available, return those, along with the option to forego the trade.
                actions.append({'play_card':None, 'draft_card':None, 'type':'trade', 'coords':None})
                return actions
                
        #If trade is prohibited, or no trades available, add action/s for each card in player's hand.
        #For each action, add copies corresponding to the various draft cards that could be selected at end of turn.
        for card in agent_state.hand:
            if card in ['jd','jc']: #two-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if game_state.board.chips[r][c]==EMPTY:
                            for draft in game_state.board.draft:
                                actions.append({'play_card':card, 'draft_card':draft, 'type':'place', 'coords':(r,c)})
                            
            elif card in ['jh','js']: #one-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if game_state.board.chips[r][c]==agent_state.opp_colour:
                            for draft in game_state.board.draft:
                                actions.append({'play_card':card, 'draft_card':draft, 'type':'remove', 'coords':(r,c)})
            
            else: #regular cards
                for r,c in COORDS[card]:
                    if game_state.board.chips[r][c]==EMPTY:
                        for draft in game_state.board.draft:
                            actions.append({'play_card':card, 'draft_card':draft, 'type':'place', 'coords':(r,c)})
                    
        return actions

    def update(self, action):
        temp_state = self.game_rule.current_game_state
        self.game_rule.current_game_state = self.generateSuccessor(temp_state, action, self.game_rule.current_agent_index)
        #If current action is to trade, agent in play continues their turn.
        self.game_rule.current_agent_index = self.getNextAgentIndex() if action['type']!='trade' else self.game_rule.current_agent_index
        self.game_rule.action_counter += 1

    def generateSuccessor(self, state, action, agent_id):
        state.board.new_seq = False
        print(f"agent id {agent_id}")
        plr_state = state.agents[agent_id]
        plr_state.last_action = action #Record last action such that other agents can make use of this information.
        reward = 0
 
        #Update agent state. Take the card in play from the agent, discard, draw the selected draft, deal a new draft.
        #If agent was allowed to trade but chose not to, there is no card played, and hand remains the same.
        card  = action['play_card']
        draft = action['draft_card']
        if card:
            print(plr_state.hand)
            print(card)
            plr_state.hand.remove(card)                 #Remove card from hand.
            print('generateSuccessor debug')
            plr_state.discard = card                    #Add card to discard pile.
            state.deck.discards.append(card)            #Add card to global list of discards (some agents might find tracking this helpful).
            state.board.draft.remove(draft)             #Remove draft from draft selection.
            plr_state.hand.append(draft)                #Add draft to player hand.
            state.board.draft.extend(state.deck.deal()) #Replenish draft selection.
        
        #If action was to trade in a dead card, action is complete, and agent gets to play another card.
        if action['type']=='trade':
            plr_state.trade = True #Switch trade flag to prohibit agent performing a second trade this turn.
            return state

        #Update Sequence board. If action was to place/remove a marker, add/subtract it from the board.
        r,c = action['coords']
        if action['type']=='place':
            state.board.chips[r][c] = plr_state.colour
            state.board.empty_coords.remove(action['coords'])
            state.board.plr_coords[plr_state.colour].append(action['coords'])            
        elif action['type']=='remove':
            state.board.chips[r][c] = EMPTY
            state.board.empty_coords.append(action['coords'])
        else:
            print("Action unrecognised.")
        
        #Check if a sequence has just been completed. If so, upgrade chips to special sequence chips.
        if action['type']=='place':
            seq,seq_type = self.checkSeq(state.board.chips, plr_state, (r,c))
            if seq:
                reward += seq['num_seq']
                state.board.new_seq = seq_type
                for sequence in seq['coords']:
                    for r,c in sequence:
                        if state.board.chips[r][c] != JOKER: #Joker spaces stay jokers.
                            state.board.chips[r][c] = plr_state.seq_colour
                            try:
                                state.board.plr_coords[plr_state.colour].remove(action['coords'])
                            except: #Chip coords were already removed with the first sequence.
                                pass
                plr_state.completed_seqs += seq['num_seq']
                plr_state.seq_orientations.extend(seq['orientation'])
        
        plr_state.trade = False #Reset trade flag if agent has completed a full turn.
        plr_state.agent_trace.action_reward.append((action,reward)) #Log this turn's action and any resultant score.
        plr_state.score += reward
        return state

    def gameEnds(self): #Game ends if a team has formed at least 2 sequences, or if the deck is empty.
        scores = {RED:0, BLU:0}
        for plr_state in self.game_rule.current_game_state.agents:
            scores[plr_state.colour] += plr_state.completed_seqs
        return scores[RED]>=2 or scores[BLU]>=2 or len(self.game_rule.current_game_state.board.draft)==0

    def checkSeq(self, chips, plr_state, last_coords):
        clr,sclr   = plr_state.colour, plr_state.seq_colour
        oc,os      = plr_state.opp_colour, plr_state.opp_seq_colour
        seq_type   = TRADSEQ
        seq_coords = []
        seq_found  = {'vr':0, 'hz':0, 'd1':0, 'd2':0, 'hb':0}
        found      = False
        nine_chip  = lambda x,clr : len(x)==9 and len(set(x))==1 and clr in x
        lr,lc      = last_coords
        
        #All joker spaces become player chips for the purposes of sequence checking.
        for r,c in COORDS['jk']:
            chips[r][c] = clr
        
        #First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
        coord_list = [(4,4),(4,5),(5,4),(5,5)]
        heart_chips = [chips[y][x] for x,y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (oc in heart_chips or os in heart_chips):
            seq_type = HOTBSEQ
            seq_found['hb']+=2
            seq_coords.append(coord_list)
            
        #Search vertical, horizontal, and both diagonals.
        vr = [(-4,0),(-3,0),(-2,0),(-1,0),(0,0),(1,0),(2,0),(3,0),(4,0)]
        hz = [(0,-4),(0,-3),(0,-2),(0,-1),(0,0),(0,1),(0,2),(0,3),(0,4)]
        d1 = [(-4,-4),(-3,-3),(-2,-2),(-1,-1),(0,0),(1,1),(2,2),(3,3),(4,4)]
        d2 = [(-4,4),(-3,3),(-2,2),(-1,1),(0,0),(1,-1),(2,-2),(3,-3),(4,-4)]
        for seq,seq_name in [(vr,'vr'), (hz,'hz'), (d1,'d1'), (d2,'d2')]:
            coord_list = [(r+lr, c+lc) for r,c in seq]
            coord_list = [i for i in coord_list if 0<=min(i) and 9>=max(i)] #Sequences must stay on the board.
            chip_str   = ''.join([chips[r][c] for r,c in coord_list])
            #Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
            if nine_chip(chip_str, clr):
                seq_found[seq_name]+=2
                seq_coords.append(coord_list)
            #If this potential sequence doesn't overlap an established sequence, do fast check.
            if sclr not in chip_str:
                sequence_len = 0
                start_idx    = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == clr:
                        sequence_len += 1
                    else:
                        start_idx = i+1
                        sequence_len = 0
                    if sequence_len >= 5:
                        seq_found[seq_name] += 1
                        seq_coords.append(coord_list[start_idx:start_idx+5])    
                        break
            else: #Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
                for pattern in [clr*5, clr*4+sclr, clr*3+sclr+clr, clr*2+sclr+clr*2, clr+sclr+clr*3, sclr+clr*4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx+5] == pattern:
                            seq_found[seq_name]+=1
                            seq_coords.append(coord_list[start_idx:start_idx+5])
                            found = True
                            break
                    if found:
                        break
        
        for r,c in COORDS['jk']:
            chips[r][c] = JOKER #Joker spaces reset after sequence checking.
        
        num_seq = sum(seq_found.values())
        if num_seq > 1 and seq_type != HOTBSEQ:
            seq_type = MULTSEQ
        return ({'num_seq':num_seq, 'orientation':[k for k,v in seq_found.items() if v], 'coords':seq_coords}, seq_type) if num_seq else (None,None)