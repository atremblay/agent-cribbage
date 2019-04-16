#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:45:28 2019


Play Cribbage


@author: nicholas
"""


import gym 
import gym_cribbage
import random
import copy
import numpy as np
import torch
import torch.nn as nn

env = gym.make('cribbage-v0')




# model architechture

class DeeperFF(nn.Module):
    def __init__(self, num_features=156, n_hid1=128, n_hid2=64, n_hid3=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, n_hid1),
            nn.ReLU(),
        
            nn.Linear(n_hid1, n_hid2),
            nn.ReLU(),
            
            nn.Linear(n_hid2, n_hid3),
            nn.ReLU(),

            nn.Linear(n_hid3, 1),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  
                nn.init.constant_(m.bias, 0)
                
    def forward(self, input_tensor):
        return self.model(input_tensor)




def skip_start(env):
    """ 
    Remove 2 cards from each hands. 
    Remove 1st card, but random since Deck shuffled
    Returns the first hand to play (playable cards only)
    """
    for i in range(4):
        # Get active player
        player = env.player
        # Remove a card for that player
        state, _, _, _ = env.step(env.hands[player][0])
    return state



def feature_representation_HTA(hand, table, action):
    """
    Turns a hand, table, action (HTA) in it's 
    feature represnatation for Q, i.e. a 
    vectors of 12*13 long, where bloc
        0-4/12 = hand. Sorted (to reduce variability)
        5-11/12 = table
        12 = action
    Use only rank of cards. 
    """
    fr = torch.zeros(13*12)
    # Hand
    hand = [c.rank_value for c in hand]
    hand.sort()   
    for i, c in enumerate(hand):
        fr[i*13+c-1] = 1
    # Table
    for i, c in enumerate(table):
        i += 4
        fr[i*13+c.rank_value-1] = 1
    # Action
    fr[11*13+action.rank_value-1] = 1
        
    return fr




def Policy(VF, S, epsilon):
    """ 
    S is tuple (hand, table)
    Returns the card with the highest Q(S,card) value, 1-epsilon times
    """
    hand = S[0]
    table = S[1]
    tests = np.zeros(len(hand))

    for i, card in enumerate(hand):
        # Make copies to not change the env
        test_hand = copy.deepcopy(hand)
        test_table = copy.deepcopy(table)
        # Play the card
   #     test_hand.play(card)
   #     test_table = test_table.add(card)
        # In feature represenation for Q
        test_fr = feature_representation_HTA(test_hand, test_table, card)
        tests[i] = VF(test_fr)
        
    return hand[int(tests.argmax())]



def id_best_card_to_play(env):
    """
    Get the best card to play according to active player 
    plus reward associate
    """
    best_reward = 0
    best_card = random.choice(env.state.hand)
    env_start = copy.deepcopy(env)
    for card in (env_start.state.hand):
        _,reward,_,_ = env_start.step(card)
        if reward > best_reward:
            best_reward = reward
            best_card = card
        # Come back to initial env
        env_start = copy.deepcopy(env)
    return best_card



def PlayRandom(env, VF):
    env.reset()
    S_partial = skip_start(env)     

    R1 = 0
    R2 = 0
    
    while env.phase == 1:
        #print(env.player, S_partial.hand)

        # who is playing
        if env.player == 0: 
            # What to play
            A = Policy(VF, (S_partial.hand, env.table), 0)
        else:
            # Random player
            A = random.choice(S_partial.hand)
            
       # print(env.player, A)
        # Play it
        
        S_partial, R, _, _ = env.step(A)
        
        if S_partial.reward_id == 0: R1 += R
        else: R2 += R
        
    #print(R1, R2)
    
    if R1 > R2: value = 1
    elif R1 == R2: value = 0.5
    else: value =0 
        
        
    return value





def PlayDeterministic(env, VF):
    env.reset()
    S_partial = skip_start(env)     

    R1 = 0
    R2 = 0
    
    while env.phase == 1:
        #print(env.player, S_partial.hand)

        # who is playing
        if env.player == 0: 
            # What to play
            A = Policy(VF, (S_partial.hand, env.table), 0)
        else:
            # Deterministic player
            A = id_best_card_to_play(env)
            
       # print(env.player, A)
        # Play it
        
        S_partial, R, _, _ = env.step(A)
        
        if S_partial.reward_id == 0: R1 += R
        else: R2 += R
        
    #print(R1, R2)
    
    if R1 > R2: value = 1
    elif R1 == R2: value = 0.5
    else: value =0 
        
        
    return value
        




        
#%%
        
Q = DeeperFF()

#%%

sd = torch.load('/Users/nicholas/Desktop/COMP767/Cribbage/Results/GameReward.pth')

#%%

Q.load_state_dict(sd['state_dict'])        

#%%
print('Deterministic')
ratios = np.zeros(1)

for tries in range(1):
    
    plays=1000
    l_win = np.zeros(plays)  
    
    for i in range(plays):
        win = PlayDeterministic(env, Q)
        l_win[i] = win
        
    print(l_win.mean())
    ratios[tries] = l_win.mean()
    
print(ratios.mean())    
    
        
#%%

print('Random')
plays=1000
l_win = np.zeros(plays)  

for i in range(plays):
    win = PlayRandom(env, Q)
    l_win[i] = win
    
print(l_win.mean())
        
    
    
    
    
    
    
    
    
    
    
    
