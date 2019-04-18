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
parser.add_argument('--boucle', type=int, metavar='', default=200, help='Number of boucle')

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
    def __init__(self, num_features=143, n_hid1=128, n_hid2=64, n_hid3=32):
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



def Eval(valid_loader, model, criterion, DEVICE):
    model.eval
    eval_loss = 0
    nb_batch = len(valid_loader)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
           # if batch_idx % 300000 == 0:
           #     print('**VALID** Batch {} out of {}.  Loss:{}'\
           #           .format(batch_idx, nb_batch, eval_loss/(batch_idx+1)))

            inputs = inputs.to(DEVICE).float()
            targets = targets.to(DEVICE).float().view(-1,1)
            pred = model(inputs)  
            loss = criterion(pred, targets)
            eval_loss += loss
    
    eval_loss /= nb_batch 
    
    return eval_loss






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




def feature_representation_Full(state, env, action):
    """
    Turns a hand, env, action in it's 
    Full feature represnatation for Q, i.e. a vector of (143):
        13 (Hand) 
        + 13 (player's cards played, starter, player's cards in crib)
        + 13 (cards played by opponent)
        + 7*13 (91) for ordered table 
        + 13 Action, card to play
    """
    
    fr = torch.zeros(11*13)
    
    # Hand
    hand = [c.rank_value for c in state.hand]
    for c in (hand):
        fr[c-1] += 1
        
    # Played by opponent 
    played_opp = [c.rank_value for c in env.played[int(not state.hand_id)]]
    for c in (played_opp):
        fr[13+c-1] += 1
        
    # Cards not available anymore
    # Get crib of player
    if state.hand_id == env.dealer:
        crib = [env.crib[0]] + [env.crib[2]]
    else:
        crib = [env.crib[1]] + [env.crib[3]]
    not_avail = [c.rank_value for c in env.played[state.hand_id]] + \
                [c.rank_value for c in env.starter] + \
                [c.rank_value for c in crib]    
    for c in (not_avail):
        fr[2*13+c-1] += 1   

    # Table
    for i, c in enumerate(env.table):
        i += 3
        fr[i*13+c.rank_value-1] = 1
                 
    # Action
    fr[10*13+action.rank_value-1] = 1
        
    return fr.to(DEVICE)


#%%

def Policy(VF, S, epsilon):
    """ 
    S is tuple (hand, table)
    Returns the card with the highest Q(S,card) value, 1-epsilon times
    """
    state = S[0]
    env = S[1]
    tests = np.zeros(len(state.hand))

    for i, card in enumerate(state.hand):
        # Make copies to not change the env
        test_state = copy.deepcopy(state)
        test_env = copy.deepcopy(env)
        # Play the card
   #     test_hand.play(card)
   #     test_table = test_table.add(card)
        # In feature represenation for Q
        test_fr = feature_representation_Full(test_state, test_env, card)
        tests[i] = VF(test_fr)
        
  #  return hand[int(tests.argmax())]

    return np.random.choice([ state.hand[int(tests.argmax())] , np.random.choice(state.hand) ], \
                                  p=[1-epsilon, epsilon])

#%%



def make_data_Off_Def(VF, env, epsilon=0.1, qt=1):
   
    data = []
    
    for d in range(qt):
        if d%1000 == 0: print(d)
       
        """ Initialisation """
        env.reset()
        S_partial = skip_start(env)        
        
        # Ajust S according to active player (good hand + env.table)
        S = copy.deepcopy((S_partial, env))
        
        # Get A from Policy(S)
        A = Policy(VF, S, epsilon)
        
        # Play A to get the AfterState (AS)
        AS_partial, R, _, _ = env.step(A)
        
        # Ajust AS according to active player (good hand + env.table)
        AS = copy.deepcopy((AS_partial, env))
                
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
                    AS = copy.deepcopy((AS_partial, env))
                    # Get AfterAction (AA) from Policy(AS)
                    AA = Policy(VF, AS, epsilon)
            
            if env.phase == 1:
                # Ajust SPrime according to active player    
                SPrime = copy.deepcopy((SPrime_partial, env))
                # Get APrime from Policy(SPrime)
                APrime = Policy(VF, SPrime, epsilon)
                
                # Update Dataset (in feature representation)
                fr_Prime = feature_representation_Full(SPrime[0],SPrime[1], APrime)
                target = float(R - R_opp + VF(fr_Prime))
                fr = feature_representation_Full(S[0], S[1], A)
                data.append((fr, target))
                
                # Prepare for next iteration 
                S = AS
                A = AA
                AS = SPrime
                AA = APrime
                R = R_opp
                
            else:
                # Update Dataset for terminal state
                target = R - R_opp
                # in feature representation
                fr = feature_representation_Full(S[0], S[1], A)
                data.append((fr, target))
                # The last one of the episode, in feature representation
                fr_Prime = feature_representation_Full(SPrime[0],SPrime[1], APrime)
                data.append((fr_Prime, R_opp))

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
    S_partial = skip_start(env)     

    R1 = 0
    R2 = 0
    
    while env.phase == 1:
        #print(env.player, S_partial.hand)

        # who is playing
        if env.player == 0: 
            # What to play
            A = Policy(VF, (S_partial, env), 0)
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
    
         
    print('\n########################### boucle ', boucle, ' ###################################\n')
  
    
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
    criterion = nn.MSELoss()
    
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
    print('Saving...')
    state = {
            'boucle': boucle,
            'state_dict': Q.state_dict(),
            'optimizer': optimizer.state_dict(),
            'nb_wins': nb_wins
            }
    torch.save(state, './Results/FullRepOffDef_'+args.id+'.pth')
    print('...saved\n\n')
                
    

    
    
#%%
avrg_wins = float(nb_wins[-50:].mean())
    
# For Orion, print results (MongoDB,...)
report_results([dict(
    name='NEG_Avrg_wins_last_50_boucles_on_1000',
    type='objective',
    value=-avrg_wins)])

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




