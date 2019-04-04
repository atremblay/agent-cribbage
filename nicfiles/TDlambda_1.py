#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:15:02 2019


TD Lambda


@author: nicholas
"""



import torch
import numpy as np
from torch.utils import data
import torch.nn as nn
import argparse
import sys



print(sys.argv)


######## Setting args
parser = argparse.ArgumentParser(description='TD lambda for Cribbage')
parser.add_argument('--id', type=str, metavar='', required=True, help='ID of experience. Will be used when saving file.')
parser.add_argument('--lr', type=float, metavar='', default=0.00001, help='Learning rate')
parser.add_argument('--gamma', type=float, metavar='', default=0, help='gamma - importance of future rewards')
parser.add_argument('--lamb', type=float, metavar='', default=0, help='lambda - Fading paste of eligibility trace')
parser.add_argument('--batch', type=int, metavar='', default=1, help='Batch size')                      # AT 1 for first test on TD
parser.add_argument('--epoch', type=int, metavar='', default=1000, help='Number of epoch')
parser.add_argument('--hidden1', type=int, metavar='', default=128, \
                    help='Integers corresponding to the hidden layers size')
parser.add_argument('--hidden2', type=int, metavar='', default=64, \
                    help='Integers corresponding to the hidden layers size')
parser.add_argument('--hidden3', type=int, metavar='', default=32, \
                    help='Integers corresponding to the hidden layers size')
parser.add_argument('--patience', type=int, metavar='', default=5, \
                    help='number of epoch to wait without improvement in valid_loss before ending training')
parser.add_argument('--ORION', type=bool, metavar='', default=False, \
                    help="Using ORION for hyper-parameter search or not")
parser.add_argument('--DEVICE', type=str, metavar='', default='cuda', choices=['cuda', 'cpu'], \
                    help="Type of machine to run on")
args = parser.parse_args()

# To use Orion
if args.ORION:
    from orion.client import report_results



class FeatRepDataset(data.Dataset):
    """
    This is a class working with Feature Representation of cards.
    Inputs is 13*12, reprensenting 12 cards (1 to 13). 0-4 Hand, 5-12 Table 
    Target is the max reward in this setting
    
    Returns a sample (input) and a targeted value (target).
    """
    
    def __init__(self, d, t):
        self.d = d
        self.t = t
        
    def __len__(self):
        "Total number of samples."
        return len(self.t)

    def __getitem__(self, index):
        "Generate one sample of data."
        return self.d[:,index], self.t[index]



class DeeperFF(nn.Module):
    def __init__(self, num_features, dropout=0.25, n_hid1=10, n_hid2=5, n_hid3=2):
        super(DeeperFF, self).__init__()
        self.model = nn.Sequential(
            #nn.Dropout(dropout),
            nn.Linear(num_features, n_hid1),
            nn.ReLU(),
            #nn.BatchNorm1d(n_hid1),
            #nn.Dropout(dropout),            
            nn.Linear(n_hid1, n_hid2),
            nn.ReLU(),
            #nn.BatchNorm1d(n_hid2),
            #nn.Dropout(dropout),
            nn.Linear(n_hid2, n_hid3),
            nn.ReLU(),
            #nn.BatchNorm1d(n_hid2),
            #nn.Dropout(dropout),
            nn.Linear(n_hid3, 1),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Only good for visual??
                nn.init.constant_(m.bias, 0)
                
    def forward(self, input_tensor):
        return self.model(input_tensor)



" Function just to see human way of hand "
def to_card(data_mat):
    cards = []
    for i in range(12):
        c = (data_mat[i*13:i*13+13]).nonzero()+1
        if c.nelement() != 0: c = c[0].item()
        else: c = 0
        cards.append(c)
        
    return cards


#
#def Train(train_loader, model, criterion, optimizer, DEVICE):
#    model.train()
#    train_loss = 0
#    nb_batch = len(train_loader) 
#        
#    for batch_idx, (inputs, targets) in enumerate(train_loader):
#        if batch_idx % 5000 == 0:
#            print('Batch {} out of {}.  Loss:{}'\
#                  .format(batch_idx, nb_batch, train_loss/(batch_idx+1)))  
#
#        inputs = inputs.to(DEVICE).float()
#        targets = targets.to(DEVICE).float().view(-1,1)
#        # re-initialize the gradient computation
#        optimizer.zero_grad()   
#        pred = model(inputs)
#        loss = criterion(pred, targets)
#        loss.backward()
#        optimizer.step()
#        train_loss += loss
#        
#    train_loss /= nb_batch
#        
#    return train_loss 
#
#
#def Eval(valid_loader, model, criterion, DEVICE):
#    model.eval
#    eval_loss = 0
#    nb_batch = len(valid_loader)
#    
#    with torch.no_grad():
#        for batch_idx, (inputs, targets) in enumerate(valid_loader):
#            if batch_idx % 5000 == 0:
#                print('**VALID** Batch {} out of {}.  Loss:{}'\
#                      .format(batch_idx, nb_batch, eval_loss/(batch_idx+1)))
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





if __name__ == '__main__':
    ######## LOAD DATA
    dTr = np.load('dTrFeatureRepData800K.npy')
    tTr = np.load('tTrFeatureRepData800K.npy')
    dV = np.load('dVFeatureRepData800K.npy')
    tV = np.load('tVFeatureRepData800K.npy')
    
    ######## CREATING DATASET ListRatingDataset 
    print('******* Creating torch datasets *******')
    train_dataset = FeatRepDataset(dTr, tTr)
    valid_dataset = FeatRepDataset(dV, tV)


    ######## CREATE DATALOADER
    print('******* Creating dataloaders *******\n\n')    
    kwargs = {}
    if(args.DEVICE == "cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, **kwargs)
    valid_loader = data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True, **kwargs)    


    ######## CREATE MODEL
    print('******* Creating Model *******\n\n')  
    V = DeeperFF(156, n_hid1=args.hidden1, n_hid2=args.hidden2, n_hid3=args.hidden3).to(args.DEVICE)
    # Create a z that mimics the model, but we will only use values    
    z = DeeperFF(156, n_hid1=args.hidden1, n_hid2=args.hidden2, n_hid3=args.hidden3).to(args.DEVICE)
    optimizer = torch.optim.SGD(V.parameters(), lr = args.lr, momentum=0.9, nesterov=True)
    criterion = nn.MSELoss()
    
    
    ######## RUN TRAINING AND VALIDATION


    valid_losses = []
    
    
    for epoch in range(args.epoch):
        
        V.train
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
        
            
            optimizer.zero_grad()
            
            
            # Initiialize z (eligibility trace) at 0
            for p in z.parameters():
                nn.init.constant_(p, 0)
                
                
                
            # NO LOOP FOR STEPS SINCE 1 EPISODE == 1 STEP
            Vs = V(inputs.float().to(args.DEVICE))

            # Print values
            if batch_idx % 100000 == 0: 
                print(epoch, ' - ',batch_idx)
                print(to_card(inputs[0]))
                print(Vs)
                print(targets)
                 
            
            # For 1 episode = 1 step, VsPrime = 0    ***********************
            
            VsPrime = 0
            
            # Evaluate delta (TD error) (dit it mow to include in z)
            delta = targets.float().to(args.DEVICE) + args.gamma * VsPrime - Vs 

            
            # Evaluate gradient (will appear in V.model[i].weight.grad)
            Vs.backward()
            
            
            # EVALUATE Z   (inlcude delta here to use optimizer)
    
            # For each layer in the model (REMEMBER: z and V have same structure)
            for i in range(len(V.model)):
                # If it's a Linear one (not a ReLu or Dropout...)
                if isinstance(V.model[i], nn.Linear):
                    z.model[i].weight = torch.nn.Parameter( delta *  \
                           (args.gamma * args.lamb * z.model[i].weight + V.model[i].weight.grad ))
                    z.model[i].bias = torch.nn.Parameter( delta * \
                           (args.gamma * args.lamb * z.model[i].bias + V.model[i].bias.grad ))

                                                    
            
            # Replace gradient in graph with new z values. NEGATIVE BECAUSE OPTIMAZER MIN, WE MAX
            
            # For each layer in the model (REMEMBER: z and V have same structure)
            for i in range(len(V.model)):
                # If it's a Linear one (not a ReLu or Dropout...)
                if isinstance(V.model[i], nn.Linear):
                    V.model[i].weight.grad  = -1 * z.model[i].weight 
                    V.model[i].weight.bias  = -1 * z.model[i].bias

        
            # Update weights 
            optimizer.step()

        
        ##### EVAL AT EACH EPOCH 
        
        V.eval
            
        for batch_idx, (inputs, targets) in enumerate(valid_loader):    
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(valid_loader):
        
                    inputs = inputs.to(args.DEVICE).float()
                    targets = targets.to(args.DEVICE).float().view(-1,1)
                    pred = V(inputs)  
                    loss = criterion(pred, targets)
                    print('Average value of V over valid is:', pred.mean())
                    print('The avrg error upon targets is:', loss, '\n')
            
#            
#            
#
#        
#        train_loss = Train(train_loader, model, criterion, optimizer, args.DEVICE)
#        eval_loss = Eval(valid_loader, model, criterion, args.DEVICE)
#       
#        train_losses.append(train_loss)
#        valid_losses.append(eval_loss)
#        losses = [train_losses, valid_losses]  
#        
#        print('\n==> Epoch: {} \n======> Train loss: {:.4f}\n======> Valid loss: {:.4f}'.format(
#              epoch, train_loss, eval_loss))
#        
#        # Patience - Stop if the Model didn't improve in the last 'patience' epochs
#        patience = args.patience
#        if len(valid_losses) - valid_losses.index(min(valid_losses)) > patience:
#            print('--------------------------------------------------------------------------------')
#            print('-                               STOPPED TRAINING                               -')
#            print('-  Recent valid losses:', valid_losses[-patience:])
#            print('--------------------------------------------------------------------------------')
#            break
#    
#        
#        # Save fisrt model and only if it improves on valid data after   
#        precedent_losses = valid_losses[:-1]
#        if precedent_losses == []: precedent_losses = [0]     # Cover 1st epoch for min([])'s error
#        if epoch == 0 or eval_loss < min(precedent_losses):
#            print('Saving...')
#            state = {
#                    'epoch': epoch,
#                    'eval_loss': eval_loss,
#                    'state_dict': model.state_dict(),
#                    'optimizer': optimizer.state_dict(),
#                    'losses': losses
#                    }
#            torch.save(state, './Results/'+args.id+'.pth')
#            print('...saved\n\n')
#            
#
#    
    
#    ######## CREATING DATASET ListRatingDataset 
#    print('******* Creating torch datasets *******')
#    test_dataset = FeatRepDataset(dTs, tTs)
#
#
#    ######## CREATE DATALOADER
#    print('******* Creating dataloaders *******\n\n')    
#    kwargs = {}
#    if(args.DEVICE == "cuda"):
#        kwargs = {'num_workers': 1, 'pin_memory': True}
#    test_loader = data.DataLoader(test_dataset, batch_size=args.batch, shuffle=True, **kwargs)

#%%

#    ############# MESURES
#
#    mesure_loader = data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True, **kwargs)    
#    
#    for inputs, targets in mesure_loader:
#        inputs = inputs.to(args.DEVICE).float()
#        targets = targets.to(args.DEVICE).float().view(-1,1)
#        pred = model(inputs)
#        joined = torch.cat((pred, targets), 1)
#    
#    
#    #%%
#    joinedExs = joined[0:20]
#    joinedErr = joined[:,0].sub(joined[:,1]).abs()
#    joinedRatio = len(joinedErr.nonzero()) / 80000
#    meanErr = joinedErr[joinedErr.nonzero()].float().mean()
#    varErr = joinedErr[joinedErr.nonzero()].float().var()
#    print('Ratio of error: ', joinedRatio)
#    print("When error, it's mean=", meanErr)
#    print('and variance=', varErr)
#
#
##%%
#    
#    if args.ORION:    
#        # For Orion, print results (MongoDB,...)
#        report_results([dict(
#        name='valid_loss',
#        type='objective',
#        value=eval_loss.item()),
#        dict(
#        name='nb_epoch',
#        type='constraint',
#        value=epoch),
#        dict(
#        name='Ratio_Err',
#        type='constraint',
#        value=joinedRatio),
#        dict(
#        name='Mean_Err',
#        type='constraint',
#        value=meanErr.item()),
#        dict(
#        name='Var_Err',
#        type='constraint',
#        value=varErr.item()),
#        dict(
#        name='Examples',
#        type='constraint',
#        value=joinedExs)])


#%%


        

#%%


#import matplotlib.pyplot as plt 
#
#
#def Plot(l_x, l_y, x_label = 'x -axis', y_label='y-axis', title='Graph'):   
#    """Plot list of curves, with or without standard deviation (e)"""
#   
#    for i in range(len(l_y)):
#        plt.scatter(l_x[i], l_y[i])
#        
#    # naming the x axis 
#    plt.xlabel(x_label) 
#    # naming the y axis 
#    plt.ylabel(y_label) 
#          
#    # giving a title to my graph 
#    plt.title(title) 
#    
#    # print legend
#    plt.legend()
#    
#    # function to show the plot 
#    plt.show() 
#
#
##%%
#pr = pred.view(-1).tolist()
#ta = targets.view(-1).tolist()
##print(pr)
##print(ta)   
#plt.scatter(ta, pr)
        
    
    
    

