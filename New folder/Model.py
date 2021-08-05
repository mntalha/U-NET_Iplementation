# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 12:59:27 2021

@author: talha
"""

import torch.nn as nn
import torch.nn.functional as F
import torch 
class Model(nn.Module):
    

    
    def __init__(self):

        super(Model, self).__init__()
        
        keep_rate=0.50
        
        #contracting
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding="same",bias=True) 
        self.dropout1 = nn.Dropout2d(1-keep_rate)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding="same",bias=True)
        self.maxpooling1= nn.MaxPool2d(2)
        
        
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding="same",bias=True)
        self.dropout2 = nn.Dropout2d(1-keep_rate)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding="same",bias=True)
        self.maxpooling2= nn.MaxPool2d(2)
        
        
        self.conv5 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding="same",bias=True)
        self.dropout3 = nn.Dropout2d(1-keep_rate)
        self.conv6 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding="same",bias=True)
        self.maxpooling2= nn.MaxPool2d(2)
        
        self.conv7 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding="same",bias=True)
        self.dropout4 = nn.Dropout2d(1-keep_rate)
        self.conv8 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding="same",bias=True)
        self.maxpooling3= nn.MaxPool2d(2)
        

        self.conv9 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding="same",bias=True)
        self.dropout5 = nn.Dropout2d(1-keep_rate)
        self.conv10 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding="same",bias=True)


        #expansive path        
        self.conv11= nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=1,padding="same",bias=True) #
        self.conv12= nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding="same",bias=True)
        self.dropout6 = nn.Dropout2d(1-keep_rate)
        self.conv13= nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding="same",bias=True)
        
        self.conv14= nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=1,padding="same",bias=True) #
        self.concatenate2  = torch.cat((self.conv14,self.conv6))  #after relulardan sonra
        self.conv15= nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding="same",bias=True) #
        self.dropout7 = nn.Dropout2d(1-keep_rate)
        self.conv16= nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding="same",bias=True) #
        
        self.conv17= nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=1,padding="same",bias=True) #
        self.concatenate3  = torch.cat((self.conv17,self.conv4))  #after relulardan sonra
        self.conv18= nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding="same",bias=True) #
        self.dropout8 = nn.Dropout2d(1-keep_rate)
        self.conv19= nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding="same",bias=True) #
        
        self.conv20= nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=2,stride=1,padding="same",bias=True) #
        self.concatenate4  = torch.cat((self.conv20,self.conv2))  #after relulardan sonra
        self.conv21= nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding="same",bias=True) #
        self.dropout9 = nn.Dropout2d(1-keep_rate)
        self.conv22= nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding="same",bias=True) #
        
        
        self.outputs = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=1,stride=1,padding="same",bias=True) #


    def forward(self, x):
        
        
        #contracting
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        
        x1 = self.dropout1(x1)
        
        x1 = self.conv2(x1)
        x1 = F.relu(x1)
        
        x1 = self.maxpooling1(x1)
        
        ##
        x2 = self.conv3(x1)
        x2 = F.relu(x2)
        
        x2= self.dropout2(x2)
        
        x2= self.conv4(x2)
        x2 = F.relu(x2)
        
        x2 = self.maxpooling2(x2)
        
        ##
        
        x3 = self.conv5(x2)
        x3 = F.relu(x3)
        
        x3= self.dropout3(x3)
        
        x3= self.conv6(x3)
        x3 = F.relu(x3)
        
        x3 = self.maxpooling3(x3)
        ##
        
        x4 = self.conv7(x3)
        x4 = F.relu(x4)
        
        x4= self.dropout4(x4)
        
        x4= self.conv8(x4)
        x4 = F.relu(x4)
        
        x4 = self.maxpooling4(x4)
        ##
        
        x5 = self.conv9(x4)
        x5 = F.relu(x5)
        
        x5= self.dropout5(x5)
        
        x5= self.conv10(x5)
        x5 = F.relu(x5)
        ##
        
        #expansive path    
        
        x6 = self.conv11(x5)
        x6 = F.relu(x6)
        
        x6 = torch.cat((x6,x4))  # concatanete

        x6 = self.conv12(x6)
        x6 = F.relu(x6)
        
        x6= self.dropout6(x6)
                
        x6= self.conv13(x6)
        x6 = F.relu(x6)
        ##
        
        x7 = self.conv11(x6)
        x7 = F.relu(x7)
        
        x7 = torch.cat((x7,x3))  # concatanete

        x7 = self.conv12(x7)
        x7 = F.relu(x7)
        
        x7= self.dropout7(x7)
                
        x7= self.conv13(x7)
        x7 = F.relu(x7)
        ##
        
        
        x8 = self.conv14(x7)
        x8 = F.relu(x8)
        
        x8 = torch.cat((x8,x1))  # concatanete

        x8 = self.conv15(x8)
        x8 = F.relu(x8)
        
        x8= self.dropout8(x8)
                
        x8= self.conv16(x8)
        x8 = F.relu(x8)
        ##
        
        x9 = self.conv17(x8)
        x9 = F.relu(x9)
        
        x9 = torch.cat((x9,x1))  # concatanete

        x9 = self.conv18(x9)
        x9 = F.relu(x9)
        
        x9= self.dropout9(x9)
                
        x9= self.conv19(x9)
        x9 = F.relu(x9)
        ##
        outputs =  self.outputs(x9)
        
        out = F.sigmoid(outputs)

        return out

            
if __name__ == "__main__":
    model = Model()
    
