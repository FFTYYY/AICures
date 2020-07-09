import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class myCNN(nn.Module):
    ''' CNN model for property-prediction
    '''

    def __init__(self):
        super(myCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            # nn.BatchNorm1d(num_features=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            # nn.BatchNorm1d(num_features=1),
            nn.ReLU()
        )
        self.dense = nn.Sequential(
            nn.Linear(512, 2),
            nn.Sigmoid()
        )


    def forward(self, X):
        tmp = self.layer1(X)
        tmp = self.layer2(tmp)
        tmp = tmp.squeeze()
        output = self.dense(tmp)
        return output


def transform_label(labels):
    result = []
    for o in labels:
        tmp = [1.0 - o, o]
        result.append(tmp)
    return result


def compute_loss(logits, labels):
    ''' compute the loss
    '''
    loss = nn.BCELoss()
    return loss(logits, labels)


def train_one_step(model, optimizer, X, labels):
    model.train()
    optimizer.zero_grad()
    logits = model(X)
    loss = compute_loss(logits, labels)
    
    # compute gradient
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, X, labels):
    ''' 进行预测
    '''
    with torch.no_grad():
        logits = model(X)
        loss = compute_loss(logits, labels)
    return loss


def train(model, optimizer, train_set, train_label, valid_set, valid_label, max_epoch=100):
    ''' train the model
    '''
    train_loss = 0.0
    valid_loss = 0.0 
    for i in range(max_epoch):
        train_loss = train_one_step(model, optimizer, train_set, train_label)
        valid_loss = evaluate(model, valid_set, valid_label)
        # print('-- epoch %i -> train_loss is %f, valid_loss is %f' % (i, train_loss, valid_loss))
    

def predict(model, test_set, test_label):
    ''' predict the test set, and compute ROC_AUC and PRC_AUC
    '''
    with torch.no_grad():
        logits = model(test_set)
    logits = logits.numpy()
    pos_pr = logits[:,1]

    roc_auc = roc_auc_score(test_label, pos_pr)
    precision, recall, thresholds = precision_recall_curve(test_label, pos_pr)
    prc_auc = auc(recall, precision)

    return roc_auc, prc_auc


def model_main(train_set, train_label, valid_set, valid_label, test_set, test_label):
    ''' note: all arguments are lists
    '''
    model = myCNN()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    p_train_set = torch.tensor(train_set, dtype=torch.float32).unsqueeze(1)
    p_train_label = torch.tensor(transform_label(train_label), dtype=torch.float32)
    p_valid_set = torch.tensor(valid_set, dtype=torch.float32).unsqueeze(1)
    p_valid_label = torch.tensor(transform_label(valid_label), dtype=torch.float32)
    p_test_set = torch.tensor(test_set, dtype=torch.float32).unsqueeze(1)
    train(model, optimizer, p_train_set, p_train_label, p_valid_set, p_valid_label)
    roc_auc, prc_auc = predict(model, p_test_set, test_label)
    return roc_auc, prc_auc
