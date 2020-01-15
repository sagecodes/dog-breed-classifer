import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms


import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image


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

        # self.class_names = [item[4:].replace("_", " ") for item in loaders['train'].dataset.classes]
        
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

    def load_model(self):
        pass

    def predict_breed_transfer(self,img_path,verbose=False):
        # load the image and return the predicted breed
        image = Image.open(img_path)
        
        # Transform set to 244px recommended from pytorch doc 
        # for this pre trained network & change to tensor
        transform = transforms.Compose([transforms.Resize(size=(244, 244)),
                                        transforms.ToTensor()])
                                        
        img = transform(image)[:3,:,:].unsqueeze(0)
        
        if self.use_cuda:
            img = img.cuda()
        
        preds = self.model(img)
        
        prediction = torch.max(preds,1)[1].item()
        
        print(prediction)
        # if verbose:
        #     print("Predicted class is: {}(index: {})".format(
        #                                             self.class_names[prediction],
        #                                             prediction))        
        # # return only highest prediction index
        return prediction
