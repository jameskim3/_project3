import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class FocalLoss2(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.eps = eps
    def fl_onehot(self,index,classes,tar):
        y_onehot = torch.FloatTensor(index, classes).to("cuda")
        y_onehot.zero_()
        y_onehot.scatter_(1, tar, 1)
        return (y_onehot)        
    def forward(self, inputs, targets):
        y = self.fl_onehot(inputs.size()[0],inputs.size()[1],targets.view(-1,1))
        logit = F.softmax(inputs, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.mean()  

class FocalLoss3(nn.Module):# for cutmix
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss3, self).__init__()
        self.gamma = gamma
        self.eps = eps
    def fl_onehot(self,index,classes,tar):
        y_onehot = torch.FloatTensor(index, classes).to("cuda")
        y_onehot.zero_()
        y_onehot.scatter_(1, tar, 1)
        return (y_onehot)        
    def forward(self, inputs, targets):
        logit = F.softmax(inputs, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * targets * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.mean()

import torch.nn.functional as F
class DenseCrossEntropy(nn.Module):
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()
        
    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()
        logprobs = F.log_softmax(logits, dim=-1)
        loss = -labels * logprobs
        loss = loss.sum(-1)
        return loss.mean()        

from sklearn.metrics import accuracy_score
class Engine:
    def __init__(self,model,optimizer,device):
        self.model=model
        self.optimizer=optimizer
        self.device=device
    
    def loss_fn(self,targets,outputs):
        #return DenseCrossEntropy()(outputs,targets)
        #return nn.CrossEntropyLoss()(outputs,targets)
        #return FocalLoss2()(outputs,targets)
        return FocalLoss3()(outputs,targets)
    
    def train(self,data_loader):
        preds_for_acc = []
        labels_for_acc = []
        self.model.train()
        final_loss=0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs=data["img"].to(self.device)
            targets=data["tar"].to(self.device)
            outputs=self.model(inputs)
            loss=self.loss_fn(targets,outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
            ## Acc Check
            #labels_for_acc = np.concatenate((labels_for_acc, targets.cpu().numpy()), 0)
            #preds_for_acc = np.concatenate((preds_for_acc, np.argmax(outputs.cpu().detach().numpy(), 1)), 0)
            if len(labels_for_acc)==0:
                labels_for_acc = targets.cpu().numpy()
                preds_for_acc = outputs.cpu().detach().numpy()
            else:
                labels_for_acc=np.vstack((labels_for_acc,targets.cpu().numpy()))
                preds_for_acc=np.vstack((preds_for_acc,outputs.cpu().detach().numpy()))
        accuracy = np.uint(labels_for_acc.argmax(1)==preds_for_acc.argmax(1)).sum()/labels_for_acc.shape[0]
        return final_loss/len(data_loader),accuracy
    
    def validate(self,data_loader):
        preds_for_acc = []
        labels_for_acc = []
        self.model.eval()
        final_loss=0
        for data in data_loader:
            inputs=data["img"].to(self.device)
            targets=data["tar"].to(self.device)
            with torch.no_grad():
                outputs=self.model(inputs)
                loss=self.loss_fn(targets,outputs)
                final_loss += loss.item()
            ## Acc Check
            #labels_for_acc = np.concatenate((labels_for_acc, targets.cpu().numpy()), 0)
            #preds_for_acc = np.concatenate((preds_for_acc, np.argmax(outputs.cpu().detach().numpy(), 1)), 0)
            if len(labels_for_acc)==0:
                labels_for_acc = targets.cpu().numpy()
                preds_for_acc = outputs.cpu().detach().numpy()
            else:
                labels_for_acc=np.vstack((labels_for_acc,targets.cpu().numpy()))
                preds_for_acc=np.vstack((preds_for_acc,outputs.cpu().detach().numpy()))
        accuracy = np.uint(labels_for_acc.argmax(1)==preds_for_acc.argmax(1)).sum()/labels_for_acc.shape[0]
        return final_loss/len(data_loader),accuracy
    
    def predict(self,data_loader):
        self.model.eval()
        final_predictions = []
        for data in data_loader:
            inputs=data["img"].to(self.device)
            predictions = self.model(inputs)
            predictions = predictions.cpu()
            final_predictions.append(predictions.detach().numpy())
        return final_predictions