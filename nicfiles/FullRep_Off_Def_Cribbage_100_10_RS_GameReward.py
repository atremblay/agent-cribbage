#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 08:20:52 2019


Offense Defense Cribbage


@author: nicholas
"""



import gym 
import gym_cribbage
import random
import copy
import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
import argparse
# To use Orion
from orion.client import report_results
import sys
# Deterministic start
import itertools
from gym_cribbage.envs.cribbage_env import evaluate_cards, Stack

print(sys.argv)

env = gym.make('cribbage-v0')

#%%

######## Setting args
parser = argparse.ArgumentParser(description='Cribbage Trainning with Full representation, Reward Offensive-Defensive')
parser.add_argument('--id', type=str, metavar='', required=True, help='ID of experience. Will be used when saving file.')
parser.add_argument('--lr', type=float, metavar='', default=0.01, help='Initial learning rate')
parser.add_argument('--lrdecay', type=float, metavar='', default=1, help='Amount of decay per boucle')
parser.add_argument('--epsilon', type=float, metavar='', default=0.1, help='Initial epsilon-greedy policy')
parser.add_argument('--epsidecay', type=float, metavar='', default=1, help='Amount of decay per boucle')
parser.add_argument('--initdata', type=int, metavar='', default='1', help='Categorical indicate: Number of data initialy used')
parser.add_argument('--replacedata', type=int, metavar='', default='1', help='Categorical indicate: Ratio of data replacement by boucle')
parser.add_argument('--batch', type=int, metavar='', default=64, help='Batch size')
parser.add_argument('--boucle', type=int, metavar='', default=100, help='Number of boucle')

args = parser.parse_args()

####### ADJUST agrs for data (because Orions' choice are not working)
if args.initdata == 1:
    args.initdata = 10000
elif args.initdata == 2:
    args.initdata = 45000
else: args.initdata = 90000

if args.replacedata == 1:
    args.replacedata = 0.1
elif args.replacedata == 2:
    args.replacedata = 0.2
else: args.replacedata = 0.5




######## CUDA AVAILABILITY CHECK
if torch.cuda.is_available(): DEVICE = 'cuda'
else: DEVICE = 'cpu'


#%%

       

class FeatRepDataset(data.Dataset):
    """
    This is a class working with Feature Representation of cards.
    Inputs is 13*12, reprensenting 12 cards (1 to 13). 0-4 Hand, 5-11 Table, 12 Action 
    Target is approximate value for state (Hand + Hable) and Actions
    
    Returns a sample (input) and a targeted value (target).
    """
    
    def __init__(self, l_d):
        self.l_d = l_d
        
    def __len__(self):
        "Total number of samples."
        return len(self.l_d)

    def __getitem__(self, index):
        "Generate one sample of data."
        data, target = self.l_d[index]
        return data, target

 
    
    
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
            nn.Sigmoid()
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  
                nn.init.constant_(m.bias, 0)
                
    def forward(self, input_tensor):
        return self.model(input_tensor)







def Train(train_loader, model, criterion, optimizer, DEVICE):
    model.train()
    train_loss = 0
    nb_batch = len(train_loader) 
        
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      #  if batch_idx % 5000 == 0:
      #      print('Batch {} out of {}.  Loss:{}'\
      #            .format(batch_idx, nb_batch, train_loss/(batch_idx+1)))  

        inputs = inputs.to(DEVICE).float()
        targets = targets.to(DEVICE).float().view(-1,1)
        # re-initialize the gradient computation
        optimizer.zero_grad()   
        pred = model(inputs)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss
        
    train_loss /= nb_batch
        
    return train_loss 



#def Eval(valid_loader, model, criterion, DEVICE):
#    model.eval
#    eval_loss = 0
#    nb_batch = len(valid_loader)
#    
#    with torch.no_grad():
#        for batch_idx, (inputs, targets) in enumerate(valid_loader):
#           # if batch_idx % 300000 == 0:
#           #     print('**VALID** Batch {} out of {}.  Loss:{}'\
#           #           .format(batch_idx, nb_batch, eval_loss/(batch_idx+1)))
#
#            inputs = inputs.to(DEVICE).float()
#            targets = targets.to(DEVICE).float().view(-1,1)
#            pred = model(inputs)  
#            loss = criterion(pred, targets)
#            eval_loss += loss
#    
#    eval_loss /= nb_batch 
#    
#    return eval_loss






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



def deterministic_start(env):
    """
    Removes 2 cards from each hands. 
    Choices are made in a greedy myopic way
    """
    
    val0 = np.zeros(15)
    val1 = np.zeros(15)
    
    l_comb0 = list(itertools.combinations(env.hands[0], 2))
    l_comb1 = list(itertools.combinations(env.hands[1], 2))
    
    for i, (a,b) in enumerate(l_comb0):
        h = copy.deepcopy(env.hands[0])
        test_h = Stack(cards=[c for c in h if c != a and c != b])
        val0[i] = evaluate_cards(test_h)

    for i, (a,b) in enumerate(l_comb1):
        h = copy.deepcopy(env.hands[1])
        test_h = Stack(cards=[c for c in h if c != a and c != b])
        val1[i] = evaluate_cards(test_h)
        
    a0, b0 = l_comb0[np.argmax(val0)]
    a1, b1 = l_comb1[np.argmax(val1)]
    
    if env.player == 0:
        env.step(a0)
        env.step(a1)
        env.step(b0)
        state, _, _, _ = env.step(b1)
    else:
        env.step(a1)
        env.step(a0)
        env.step(b1)
        state, _, _, _ = env.step(b0)
        
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
        
    return fr.to(DEVICE)





#%%

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
        
    return np.random.choice([ hand[int(tests.argmax())] , np.random.choice(hand) ], \
                                  p=[1-epsilon, epsilon])
    

#%%




def make_data_Off_Def(VF, env, epsilon=0.1, qt=1):
   
    data = []
    
    
    for d in range(qt):
        if d%10000 == 0: print(d)
       
        """ Initialisation """
        points = np.zeros(2)
        env.reset()
        S_partial = skip_start(env)        
        
        # Ajust S according to active player (good hand + env.table)
        S = copy.deepcopy((S_partial.hand, env.table))
        
        # Get A from Policy(S)
        A = Policy(VF, S, epsilon)
        
        # Play A to get the AfterState (AS) and add points to the appropriate player
        AS_partial, R, _, _ = env.step(A)
        points[AS_partial.reward_id] += R
        
        # Ajust AS according to active player (good hand + env.table)
        AS = copy.deepcopy((AS_partial.hand, env.table))
                
        # Get AfterAction (AA) from Policy(AS)
        AA = Policy(VF, AS, epsilon)
        
        """ Loop for complete episode (one The Play) """
        while env.phase == 1:
            R_opp = 0
            
            # In case same players has to play turns in a row
            player = env.player
            while env.player == player and env.phase == 1:
                # Play to get SPrime
                SPrime_partial, R_opp_i, _, _ = env.step(AA)    
                R_opp += R_opp_i
                if env.player == player and env.phase == 1:
                    AS_partial = SPrime_partial
                    # Get AS according to active player (good hand + env.table)
                    AS = copy.deepcopy((AS_partial.hand, env.table))
                    # Get AfterAction (AA) from Policy(AS)
                    AA = Policy(VF, AS, epsilon)    
            points[SPrime_partial.reward_id] += R_opp
            
            if env.phase == 1:
                # Ajust SPrime according to active player    
                SPrime = copy.deepcopy((SPrime_partial.hand, env.table))
                # Get APrime from Policy(SPrime)
                APrime = Policy(VF, SPrime, epsilon)
                
                # Update Dataset (in feature representation)
                fr_Prime = feature_representation_HTA(SPrime[0],SPrime[1], APrime)
                target = float(VF(fr_Prime)) #+ R - R_opp 
                fr = feature_representation_HTA(S[0], S[1], A)
                data.append((fr, target))
                
                # Prepare for next iteration 
                S = AS
                A = AA
                AS = SPrime
                AA = APrime
                R = R_opp
                
            else:
                # Check for equality
                if points[0] == points[1]:
                    target_previous = 0.5
                else: 
                    winner = points.argmax()
                    if SPrime_partial.reward_id == winner:
                        target_previous = 0
                    else:
                        target_previous = 1
#                # Update Dataset for terminal state
#                target = R - R_opp
                # in feature representation
                fr = feature_representation_HTA(S[0], S[1], A)
                data.append((fr, target_previous))
                # The last one of the episode, in feature representation
                fr_Prime = feature_representation_HTA(SPrime[0],SPrime[1], APrime)
                data.append((fr_Prime, 1 - target_previous))
            
    return data




"""
TO PLAY
"""


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



def PlayDeterministic(env, VF):
    env.reset()
    S_partial = deterministic_start(env)     

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



"""
OTHERS
"""


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


""""""""""""""""""""""""""""""""""""""""""""""""""""""

def to_card(data):
    " Function just to see human way of hand "

    cards_data = []
    for d,t in data:
        
        cards = []
        for i in range(12):
            
            c = (d[i*13:i*13+13]).nonzero()+1
            if c.nelement() != 0: c = c[0].item()
            else: c = 0
            cards.append(c)
            
        cards_data.append((cards, t))
        
    return cards_data
""""""""""""""""""""""""""""""""""""""""""""""""""""""


#%%
    
Q = DeeperFF().to(DEVICE)

#%%
ddd = make_data_Off_Def(Q, env, epsilon=args.epsilon, qt=args.initdata)

#%%%
 
#import pickle
#
#with open('OffDef100kdata.pickle', 'wb') as f:
#    pickle.dump(ddd, f)
#
#
#
#
#





#%%

nb_wins = np.zeros(args.boucle)
        
for boucle in range(args.boucle):
        
    # Create new data
    ddd_new = make_data_Off_Def(Q, env, epsilon=(args.epsidecay**boucle)*args.epsilon, \
                                qt=int(args.replacedata*args.initdata))
  
    # Put new data at the end
    new_idx = len(ddd) - len(ddd_new)
    ddd[0:new_idx] = ddd[len(ddd_new):]
    ddd[new_idx:] = ddd_new
    
         
    print('########################### boucle ', boucle, ' ###################################\n')
  
    
    # Split train / valid
    (Tr, V, _) = Splitting(ddd, 0.9, 0.1, 0)
       
    
    ######## CREATING DATASET ListRatingDataset 
    train_dataset = FeatRepDataset(Tr)
    valid_dataset = FeatRepDataset(V)
    
    
    ######## CREATE DATALOADER
    kwargs = {}
    #if(args.DEVICE == "cuda"):
    #    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
    valid_loader = data.DataLoader(valid_dataset, batch_size=64, shuffle=True, **kwargs)    
    
    
    ######## CREATE MODEL
    optimizer = torch.optim.Adam(Q.parameters(), lr = (args.lrdecay**boucle)*args.lr)#, momentum=0.9, nesterov=True)
    criterion = nn.BCELoss()
    
    ######## RUN TRAINING AND VALIDATION 
    
   # train_losses = []
   # valid_losses = []
    
   # for epoch in range(1):
        
    train_loss = Train(train_loader, Q, criterion, optimizer, DEVICE)
    
    plays=1000
    l_win = np.zeros(plays)  
    
    for i in range(plays):
        win = PlayDeterministic(env, Q)
        l_win[i] = win
        
    nb_wins[boucle] = l_win.sum()
     #  eval_loss = Eval(valid_loader, Q, criterion, DEVICE)
       
      #  train_losses.append(train_loss)
      #  valid_losses.append(eval_loss)
      #  losses = [train_losses, valid_losses]  
        
      #  print('\n==> Epoch: {} \n======> Train loss: {:.4f}\n======> Valid loss: {:.4f}'.format(
      #        epoch, train_loss, eval_loss))
        
        # Patience - Stop if the Model didn't improve in the last 'patience' epochs
      #  patience = 1  # args.patience

      #  if len(valid_losses) - valid_losses.index(min(valid_losses)) > patience:
      #      print('--------------------------------------------------------------------------------')
      #      print('-                               STOPPED TRAINING                               -')
      #      print('-  Recent valid losses:', valid_losses[-patience:])
      #      print('--------------------------------------------------------------------------------')
      #      break
    
        
        # Save fisrt model and only if it improves on valid data after   
      #  precedent_losses = valid_losses[:-1]
      #  if precedent_losses == []: precedent_losses = [0]     # Cover 1st epoch for min([])'s error
      #  if epoch == 0 or eval_loss < min(precedent_losses):
  #  print('Saving...')
    state = {
            'boucle': boucle,
            'state_dict': Q.state_dict(),
            'optimizer': optimizer.state_dict(),
            'nb_wins': nb_wins
            }
    torch.save(state, './Results/FullRepOffDef_'+args.id+'.pth')
  #  print('...saved\n\n')
                
    

    
    
#%%
avrg_wins = float(nb_wins[-10:].mean())
    
# For Orion, print results (MongoDB,...)
report_results([dict(
    name='NEG_Avrg_wins_last_10_boucles_on_1000',
    type='objective',
    value=-avrg_wins)])

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




