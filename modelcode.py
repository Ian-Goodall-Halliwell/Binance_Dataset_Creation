import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit,KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Define your LSTM model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size,dropout):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
class RNNmodel():
    def __init__(self,input_size, hidden_size,num_layers,output_size,cvs=5,learning_rate=0.01,num_epoch=100,dropout=0,weight_decay=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cvs = cvs
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epoch
        self.dropout = dropout
        self.weight_decay = weight_decay
    def train(self,train_dataloader,valid_dataloader):
        self.model = RNN(self.input_size, self.hidden_size, self.num_layers, self.output_size,self.dropout).to('cuda')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
        
        prevloss = 10000
        count = 1
        # Training loop
        first =False
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            self.model.train()
            for batch in train_dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                
                #inputs = inputs.view(inputs.size(0), inputs.size(2), -1)
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Reshape targets to match the shape of outputs
                targets = targets.view(-1, self.output_size)

                # Compute the loss
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Accumulate the loss
                total_loss += loss.item()
                
            self.model.eval()
            test_loss = 0
            
            with torch.no_grad():
                for validbatch in valid_dataloader:
                    validinputs, validtargets = validbatch
                    validinputs, validtargets = validinputs.to('cuda'), validtargets.to('cuda')
                    pred = self.model(validinputs)
                    test_loss += criterion(pred.flatten(end_dim=1), validtargets).item()
                    
                    
            # Compute the average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)
            test_avg_loss = test_loss / len(valid_dataloader)
            if test_avg_loss >= prevloss:
                if first == False:
                    state = self.model.state_dict()
                    optimstate = optimizer.state_dict()
                    first = True
                    
                count +=1
                if count >= 10:
                    self.model.load_state_dict(state)
                    break
            else:
                first = False
                count = 0
                prevloss = test_avg_loss
            # Print the average loss for the epoch
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_loss:.4f}, Valid Loss: {test_avg_loss:.4f}")
    def predict(self,test_dataloader):
            # Set the model to evaluation mode
        self.model.eval()

        # Prepare the input data for prediction
        #input_data = torch.tensor(input_data).unsqueeze(0)  # Add batch dimension
        
        # Forward pass to get the predictions
        with torch.no_grad():
            preds = []
            reals= []
            for batch in test_dataloader:
                predictions = self.model(batch[0].to('cuda'))
                reals.append(batch[1].to('cuda'))
                
                preds.append(predictions.flatten(end_dim=1))
        all_predictions = torch.cat(preds, dim=0)
        all_predictions = all_predictions.detach().to('cpu').numpy()
        all_reals = torch.cat(reals, dim=0)
        all_reals = all_reals.detach().to('cpu').numpy()
        return all_predictions, all_reals
    
    
    def cross_validate(self,X,y,sequence_length,batch_size,defaultdata):
        scores = []
        cv = KFold(5)
        for train,test in cv.split(X,y):
            # scaler = StandardScaler()
            # yt = scaler.fit_transform(y[train])
            # ytt = scaler.transform(y[test])
            # print(np.std(yt))
            # print(np.std(ytt))
            
            dataset = TimeSeriesDataset(X[train],y[train], sequence_length)
            dataset_true = TimeSeriesDataset(X[test],y[test], sequence_length)
            dataloader = DataLoader(dataset, batch_size=batch_size,pin_memory=True,pin_memory_device='cuda', shuffle=False,drop_last=False)
            dataloader_true = DataLoader(dataset_true, batch_size=batch_size,pin_memory=True,pin_memory_device='cuda', shuffle=False,drop_last=False)
            self.train(dataloader,dataloader_true)
            preds,all_reals = self.predict(dataloader_true)
            score = r2_score(all_reals,preds)
            scores.append(score)
        scores = np.mean(scores)
        return scores
    
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data,y, sequence_length):
        self.data = data
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - (self.sequence_length + 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.sequence_length]
        y = self.y[idx+self.sequence_length]
        return x, y