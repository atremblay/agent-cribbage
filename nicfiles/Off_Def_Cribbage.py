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
from tqdm import tqdm
import numpy as np
import torch
from torch.utils import data
import torch.nn as nn

env = gym.make('cribbage-v0')

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
        
    return hand[int(tests.argmax())]


#%%



def make_data_Off_Def(VF, env, epsilon=0.1, qt=1):
   
    data = []
    
    for d in range(qt):
        if d%1000 == 0: print(d)
       
        """ Initialisation """
        env.reset()
        S_partial = skip_start(env)        
        
        # Ajust S according to active player (good hand + env.table)
        S = copy.deepcopy((S_partial.hand, env.table))
        
        # Get A from Policy(S)
        A = Policy(VF, S, epsilon)
        
        # Play A to get the AfterState (AS)
        AS_partial, R, _, _ = env.step(A)
        
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
            
            if env.phase == 1:
                # Ajust SPrime according to active player    
                SPrime = copy.deepcopy((SPrime_partial.hand, env.table))
                # Get APrime from Policy(SPrime)
                APrime = Policy(VF, SPrime, epsilon)
                
                # Update Dataset (in feature representation)
                fr_Prime = feature_representation_HTA(SPrime[0],SPrime[1], APrime)
                target = float(R - R_opp + VF(fr_Prime))
                fr = feature_representation_HTA(S[0], S[1], A)
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
                fr = feature_representation_HTA(S[0], S[1], A)
                data.append((fr, target))
                # The last one of the episode, in feature representation
                fr_Prime = feature_representation_HTA(SPrime[0],SPrime[1], APrime)
                data.append((fr_Prime, R_opp))

    return data






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
    
Q = DeeperFF()

#%%
ddd = make_data_Off_Def(Q, env, epsilon=0.1, qt=100000)

#%%%
 
import pickle

with open('OffDef100kdata.pickle', 'wb') as f:
    pickle.dump(ddd, f)














#%%
        
for boucle in range(50):
        
    # Create new data
    ddd_new = make_data_Off_Def(Q, env, epsilon=0.05, qt=20000)
  
    
    # Put new data at the end
    new_idx = len(ddd) - len(ddd_new)
    ddd[0:new_idx] = ddd[len(ddd_new):]
    ddd[new_idx:] = ddd_new
    
         
    print('\n########################### boucle ', boucle, ' ###################################/n')
  
    
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
    optimizer = torch.optim.Adam(Q.parameters(), lr = 0.01)#, momentum=0.9, nesterov=True)
    criterion = nn.MSELoss()
    
    ######## RUN TRAINING AND VALIDATION 
    
    train_losses = []
    valid_losses = []
    
    for epoch in range(50):
        
        train_loss = Train(train_loader, Q, criterion, optimizer, 'cpu') #args.DEVICE)
        eval_loss = Eval(valid_loader, Q, criterion, 'cpu') #args.DEVICE)
       
        train_losses.append(train_loss)
        valid_losses.append(eval_loss)
        losses = [train_losses, valid_losses]  
        
        print('\n==> Epoch: {} \n======> Train loss: {:.4f}\n======> Valid loss: {:.4f}'.format(
              epoch, train_loss, eval_loss))
        
        # Patience - Stop if the Model didn't improve in the last 'patience' epochs
        patience = 3  # args.patience

        if len(valid_losses) - valid_losses.index(min(valid_losses)) > patience:
            print('--------------------------------------------------------------------------------')
            print('-                               STOPPED TRAINING                               -')
            print('-  Recent valid losses:', valid_losses[-patience:])
            print('--------------------------------------------------------------------------------')
            break
    
        
        # Save fisrt model and only if it improves on valid data after   
        precedent_losses = valid_losses[:-1]
        if precedent_losses == []: precedent_losses = [0]     # Cover 1st epoch for min([])'s error
        if epoch == 0 or eval_loss < min(precedent_losses):
            print('Saving...')
            state = {
                    'epoch': epoch,
                    'eval_loss': eval_loss,
                    'state_dict': Q.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'losses': losses
                    }
            torch.save(state, './Results/Off_Def_B_lr_NEWNEWDATA.pth')
            print('...saved\n\n')
                
    

#%%

i=0            

for j in range(10):
                
    print(to_card([ddd[i]]))
    print(Q(ddd[i][0]))    
        
    i = int(input('next?'))    
    

#%%
    
d99104 = ddd[99104][0]

#%%

d99104[-2] = 0
d99104[-9] = 0 
d99104[-10] = 1  

#%%
c1 = gym_cribbage.envs.cribbage_env.Card(4,"♦︎")
c2 = gym_cribbage.envs.cribbage_env.Card('Q','♣︎')
ct1 = gym_cribbage.envs.cribbage_env.Card('K',"♥︎")
ct2 = gym_cribbage.envs.cribbage_env.Card(4,"♣︎")

hand = gym_cribbage.envs.cribbage_env.Stack([c1, c2])
table = gym_cribbage.envs.cribbage_env.Stack([ct1, ct2])



#%%



Policy(VF, S, epsilon)
    
#%%


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




