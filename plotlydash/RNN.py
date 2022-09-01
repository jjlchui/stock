#RNN for time series prediciton
 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
 
#Generate the sinewave
N = 1000
series = np.sin(np.linspace(0, 100, N))
 
plt.plot(series)
plt.show()
 
#Create the dataset
#We use the past T values to predict the next one:
T = 10
X = []
Y = []
 
for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)
    
device = torch.device('cuda:0')
X_np = np.array(X)
Y_np = np.array(Y)
X = torch.tensor(X).float().view(-1, T, 1).to(device)
Y = torch.tensor(Y).float().view(-1, 1).to(device)
    
print(X.shape, Y.shape)
 
#nice, so now we have our dataset. Let's split it into train and test sets:
    
X_train, Y_train = X[:N//2], Y[:N//2]
X_test, Y_test = X[N//2:], Y[N//2:]
 
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
 
#We have our train and test data, so now let's create the model
class Model(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_rnnlayers=1):
        super(Model, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers
    
        self.rnn_pipe = nn.RNN(
            input_size = self.D,
            hidden_size = self.M,
            num_layers = self.L,
            nonlinearity = 'relu',
            batch_first = True)
        self.dense = nn.Linear(self.M, self.K)
        
    def forward(self, x):
        #initialize hidden states - L x N x M
        h0 = torch.zeros(self.L, x.size(0), self.M).to(device)
        
        #Get RNN unit output
        #out is of size (N,T,M)
        out, _ = self.rnn_pipe(x, h0)
        
        #We only want h(T) at the final time step
        #N x M -> N x K
        out = self.dense(out[:,-1,:])
        return out
        
#instantiate the model
model = Model(1, 15, 1)
 
#put it on the GPU
print(model)
model.to(device)
 
#make sure it is on the GPU
print(next(model.parameters()).is_cuda)
 
#Train the model:
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
epochs = 1000
 
train_losses = []
test_losses = []
 
for i in range(epochs):
    out = model(X_train)
    train_loss = criterion(out, Y_train)
    train_losses.append(train_loss.item())
    
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    out_test = model(X_test)
    test_loss = criterion(out_test, Y_test)
    test_losses.append(test_loss.item())
    
    if i%50 ==0:
        print(f'epoch {i}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.04f}')
        
#plot the losses
plt.plot(train_losses, label='train_loss')
plt.plot(test_losses, label='test_losses')
plt.legend()
plt.show()
    
#make predictions the right way
X_current = X_test[0]
predictions = []
for t in range(Y_test.shape[0]):
    X_current = X_current.view(1, -1, 1)
    pred = model(X_current)
    predictions.append(pred.item())
    X_current = torch.cat((X_current.view(-1)[1:], pred.view(-1)))
    
print(predictions)
plt.plot(predictions, label='pred')
plt.plot(Y_np[N//2:], label='real')
plt.legend()
plt.show()
