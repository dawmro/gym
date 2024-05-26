import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

print(T.cuda.is_available())


class DeepQNetwork(nn.Module):

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        # call constructor for base class
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims 
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # first layer of deep neural network, unpack list consisting of observation vector
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # second layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # output layer
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # loss
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.device = T.device('cpu')
        self.to(self.device)


    # forward propagation
    def forward(self, state):
        # pass state to first fully connected layer and then activate with relu function
        x = F.relu(self.fc1(state))
        # pass output to second fully connected layer and then activate with relu function
        x = F.relu(self.fc2(x))
        # pass output to final layer but dont activate it to get raw estimate
        actions = self.fc3(x)

        return actions


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=3e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        # integer representation of available actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        # counter to track position of first available memory for storing agent memory
        self.mem_cntr = 0

        # evaluation network
        self.Q_eval = DeepQNetwork(lr=self.lr, input_dims=input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions,)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # memory for done flags from env, to be used as a mask for setting value of next stares to zero in learning function
        self.terminal_terminated_memory = np.zeros(self.mem_size, dtype=bool)
        self.terminal_truncated_memory = np.zeros(self.mem_size, dtype=bool)

    # function to store transitions in agent memory
    def store_transition(self, state, action, reward, state_, terminated, truncated):
        # position of first unocupied memory, using % allows to wrap aroud and rewrite memory with new values
        index = self.mem_cntr % self.mem_size 
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_terminated_memory[index] = terminated
        self.terminal_truncated_memory[index] = truncated
        # this position in memory has been filled, increment memory counter
        self.mem_cntr += 1

    # function to choose action based of current state of environment
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # take observation, turn it into pytorch tensor and send it to device
            state = T.tensor([observation]).to(self.Q_eval.device)
            # pass state through deep q network
            actions = self.Q_eval.forward(state)
            # get integer coresponding to max action for that state, 
            # dereference it with item() because it will passed to env that doesn't take tensors but ints or numpy arrays as input
            action = T.argmax(actions).item()
        else:
            # exploration
            action = np.random.choice(self.action_space)

        return action
    
    # allow agent to learn
    def learn(self):
        # start learnig only when batch size of memory gets filled 
        if self.mem_cntr < self.batch_size:
            return
        
        # zero gradient on optimizer
        self.Q_eval.optimizer.zero_grad()
        # calculate position of maximal memory, select only upto last filled memory
        max_mem = min(self.mem_cntr, self.mem_size)
        # create batch, don't select same memory more than once
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        # needed to perform proper array slicing
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # convert numpy array subset of agent memory into pytorch tensor
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_terminated_batch = T.tensor(self.terminal_terminated_memory[batch]).to(self.Q_eval.device)
        terminal_truncated_batch = T.tensor(self.terminal_truncated_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        # feed forward through deep neural network to get relevant parameters for loss function
        # move agent estimate for value of current state torwards maximum value for next state -> selecting max action
        # dereference to get values of action we already took
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        # agent estimates for the next state
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_terminated_batch] = 0.0
        q_next[terminal_truncated_batch] = 0.0

        # calculate target values, max returns tuple: value and index, we need value [0]
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # decrease epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min








