#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:01:24 2019

@author: nicholas
"""

import gym 
import gym_cribbage
import random
import copy
import numpy as np


env = gym.make('cribbage-v0')


#%%
s1,r1,d1,_ = env.reset()

print(s1.hand)
i = int(input('P1- Discard which card? '))


s2,r2,d2,_ = env.step(s1.hand[i])
print(s2.hand)
i = int(input('P2- Discard which card? '))

s1,r1,d1,_ = env.step(s2.hand[i])
print(s1.hand)
i = int(input('P1- Discard which card? '))

s2,r2,d2,_ = env.step(s1.hand[i])
print(s2.hand)
i = int(input('P2- Discard which card? '))

a,b,c,d = env.step(s2.hand[i])

#%%



def is_sequence(cards):
    # Need at least 3 cards
    if len(cards) < 3:
        return False

    rank_values = list(sorted([c.rank_value for c in cards]))
    for i, val in enumerate(rank_values[:-1], 1):
        if (val + 1) != rank_values[i]:
            return False
    return True



def evaluate_table(cards):
    points = 0

    if sum(c.value for c in cards) == 15:
        points += 2

    # Pair points
    pair_point = 0
    for i in range(-2, -min(len(cards)+1, 5), -1):
        if cards[-1].rank == cards[i].rank:
            if i == -2:
                pair_point = 2
            elif i == -3:
                pair_point = 6
            elif i == -4:
                pair_point = 12
        else:
            break

    points += pair_point

    # Run points
    for i in reversed(range(-3, -len(cards)-1, -1)):
        if is_sequence(cards[i:]):
            points += len(cards[i:])
            break

    return points



def skip_start(env):
    """ 
    Remove 2 cards from each hands. 
    Remove 1st card, but random since shuffle
    """
    for i in range(4):
        # Get active player
        player = env.player
        # Remove a card for that player
        env.step(env.hands[player][0])
    return env
        


def id_best_card_to_play(env):
    """
    Get the best card to play according to active player 
    plus reward associate
    """
    best_reward = 0
    best_card = random.choice(env.state.hand)
    env_start = copy.deepcopy(env)
    for card in (env.state.hand):
        _,reward,_,_ = env.step(card)
        if reward > best_reward:
            best_reward = reward
            best_card = card
        # Come back to initial env
        env = copy.deepcopy(env_start)
    return best_card, best_reward, env


def make_data(env, qt=1):
    """
    Takes an env and create list of qt x (hand, table, best_card, reward)
    """
    data = []
    for d in range(qt):
        if d%1000 == 0: print(d)
        env.reset()
        skip_start(env)
        for i in range(8): 
            c, r, env = id_best_card_to_play(env)
            data.append((env.state.hand, copy.deepcopy(env.table), c, r))
            env.step(c)
            
    return data
    
#%%
    
def make_data_eval_play(env, qt=1):
    """
    Takes an env and create list of qt x (table, reward).
    Reward is given by last card added, played randomly.
    """
    data = []
    for d in range(qt):
        if d%1000 == 0: print('qt=',d)
        env.reset()
        skip_start(env)
        for i in range(8): 
            # Copy table BEFORE taking step.
            table = copy.deepcopy(env.table)
            card = copy.deepcopy(random.choice(env.state.hand))
            _, r, _, _ = env.step(card)
            data.append((table.add(card), r))
            
    return data

#%%

def feature_rep(data):
    """
    Turns list of (hand, table, best_card, reward) in 
    1- matrix item 
        size = vectors 13*12 x length of data
        Use only rank of cards. 
        0-4/12 = hand. Sorted (to reduce variability)
        5-12/12 = table
    2- vector of targets ()
        size = vector length of data
        Max reward when thinking only one step
    """
    data_mat = np.zeros((13*12, len(data)))
    target = np.zeros(len(data))
    for j,(h,t,_,r) in enumerate(data):
        hand = [c.rank_value for c in h]
        hand.sort()   
        for i, c in enumerate(hand):
            data_mat[i*13+c-1,j] = 1
        for i, c in enumerate(t):
            i += 4
            data_mat[i*13+c.rank_value-1,j] = 1
            
        target[j] = r
        
    return data_mat, target

#%%
    
" Function just to see human way of hand "
def to_card(data_mat):
    cards = []
    for i in range(12):
        c = (data_mat[i*13:i*13+13]).nonzero()+1
        if c.nelement() != 0: c = c[0].item()
        else: c = 0
        cards.append(c)
        
    return cards

#%%

# Next:
# Create data for 100K or 1M. Split train, valid test.
# Get averege reward for this
# Build FF model to try to learn this surpervised. Get Avrg reward
# Same for LSTM.
# Compare FF and LSTM in avrg reward found.
# Use the best feature representation to do TD(lambda)

#%%

def feature_rep_eval_play(data):
    """
    Turns list of (table, reward) in 
    1- matrix item 
        size = vectors 13*8 x length of data
        Use only rank of cards. 
    2- vector of targets ()
        size = vector length of data
        Value of reward because last card added
    """
    data_mat = np.zeros((13*8, len(data)))
    target = np.zeros(len(data))
    for j,(t,r) in enumerate(data):
        for i, c in enumerate(t):
            data_mat[i*13+c.rank_value-1,j] = 1
            
        target[j] = r
        
    return data_mat, target



#%%

def Splitting(l_items, ratio_1, ratio_2, ratio_3):
    """
    Splitting a list of items randowly, into sublists, according to ratios.
    Returns the 3 sublist (2 could be empty)
    """
    # Make sure ratios make sense
    if ratio_1 + ratio_2 + ratio_3 != 1:
        raise Exception("Total of ratios need to be 1, got {}".format(ratio_1 + ratio_2 + ratio_3))
    size_1 = round(ratio_1 * len(l_items))
    size_2 = round(ratio_2 * len(l_items))
    np.random.shuffle(l_items)
    sub_1 = l_items[:size_1]
    sub_2 = l_items[size_1:size_1+size_2]
    sub_3 = l_items[size_1+size_2:]

    return sub_1, sub_2, sub_3     
    


#%%%
    



















#%%


dTr = np.load('dTrFeatureRepData800K.npy')
tTr = np.load('tTrFeatureRepData800K.npy')

train_dataset = FeatRepDataset(dTr, tTr)
    
#%%
full_tr_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=True, **kwargs)    
    
#%%

same =[]
for i, (inputs, targets) in enumerate(full_tr_loader):
    d = torch.Tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).double()
    if torch.all(torch.eq(inputs, d)):
        same.append(targets)
        
#%%

st = torch.zeros(33)
for i, t in enumerate(same):
    st[i] = t
    
    
    
    
    
    


