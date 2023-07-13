import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.lin1 = nn.Linear(13, 256)
        self.lin2 = nn.Linear(256, 3)
        # self.lin3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        x = self.lin2(x)
        return x
    
    def save(self, file_name = 'model8.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optim = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx] 
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        self.optim.zero_grad()
        loss = self.criterion(target.detach(), pred)
        loss.backward()

        self.optim.step()