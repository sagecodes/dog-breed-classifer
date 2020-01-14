import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim

import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from IPython.display import display, clear_output

# TODO refactor for model type?

class Resnet50:
    
    def __init__(self, num_classes, lr=0.001,use_cuda=False):

        # use pre-trained resnet50 model from pytorch
        self.model = models.resnet50(pretrained=True)
        self.use_cuda = use_cuda

        # freeze model params for features
        for param in self.model.parameters():
            param.requires_grad = False

        # configure output layer for number of classes
        self.model.fc = nn.Linear(2048,
                            num_classes,
                            bias=True)

        # self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.model().fc.parameters(), lr=0.001)

        if self.use_cuda:
            self.model = self.model.cuda()
        


    def train(self, n_epochs, loaders, optimizer,
                        criterion, save_path, verbose=False):
        """returns trained model"""
        
        train_output = []
        
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf 
        
        for epoch in range(1, n_epochs+1):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
        
            # train the model
            self.model.train()
            for batch_idx, (data, target) in enumerate(loaders['train']):
                # move to GPU
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                    
                ## find the loss and update the model parameters accordingly
                ## record the average training loss, using something like
                ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                
                # reset gradient weights to zero
                optimizer.zero_grad()
                
                output = self.model(data)
                
                # calculate loss
                loss = criterion(output, target)
                
                # Compute Gradient
                loss.backward()
                
                # Adjust weights w/ Gradient
                optimizer.step()
                
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                
                # Output info in jupyter notebook
                
                if verbose:
                    print('Epoch #{}, Batch #{} train_loss: {:.6f}'.format(epoch, batch_idx + 1, train_loss))
                else:
                    clear_output(wait=True)
                    display('Epoch #{}, Batch #{} train_loss: {:.6f}'.format(epoch, batch_idx + 1, train_loss))
                

            ######################    
            # validate the model #
            ######################
            self.model.eval()
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                # move to GPU
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                    
                ## update the average validation loss
                output = self.model(data)
                loss = criterion(output, target)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                
            # append training/validation output to output list 
            train_output.append('Epoch: {} train_loss: {:.6f} val_loss: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss
                ))
            
            ## save the model if validation loss has decreased
            if valid_loss < valid_loss_min:
                torch.save(self.model.state_dict(), save_path)
                print(('SAVE MODEL: val_loss decrease ({:.6f})'.format(valid_loss)))
                valid_loss_min = valid_loss
        
        self.history = train_output
        # return trained model
        return self.model


    def get_model(self):
        return self.model
