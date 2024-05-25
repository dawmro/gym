import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import pickle
from timeit import default_timer as timer



# memory for experience replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    # memorize experience: state, action, new state, reward, terminated
    def append(self, transition):
        self.memory.append(transition)

    # take random sample of a given size from memory
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    # check length of memory
    def __len__(self):
        return len(self.memory)


# define model
class DQN(nn.Module):
    # pass in number of nodes in input state, hidden layer and output state
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # define first fully connnected layer
        self.fc1 = nn.Linear(in_states, h1_nodes)
        # define output layer
        self.out = nn.Linear(h1_nodes, out_actions)

    # do forward pass
    def forward(self, x):
        # apply ReLU activation
        x = F.relu(self.fc1(x))
        # calculate
        x = self.out(x)
        return x
    

# Frozen Lake Deep Q-Learning
class FrozenLakeDQL():
    # initialize hyperparamater alpha
    learning_rate_a = 0.001
    # initialize hyperparamater gamma
    discount_factor_g = 0.9
    # number of steps agent needs to take before syncing policy and target network
    network_sync_rate = 10
    # size of replay memory
    replay_memory_size = 1000
    # size of batch data set sampled from the replay memory
    mini_batch_size = 32

    # neural network
    # select nn loss function
    loss_fn = nn.MSELoss()
    # initialize empty optimizer, will be selected later
    optimizer = None

    # map action numbers into leters when printing
    ACTIONS = ['L', 'D', 'R', 'U']

    # train for how many episodes, if render map on screen, if make lake slippery
    def train(self, episodes, render=False, is_slippery=False):
        start_train = timer()

        # create env
        env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=is_slippery, render_mode="human" if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # epsilon = 1 -> 100% random actions
        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        # create policy and target network
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # copy weights/biases from one network to the other, making them the same
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print("Policy before training: ")
        self.print_dqn(policy_dqn)

        # initialize optimizer with values
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # track rewards collectd per episode
        rewards_per_episode = np.zeros(episodes)

        # track epsilon decay
        epsilon_history = []

        # track number of steps taken to know when sync policy and targret network
        step_count = 0 

        # train for a given amount of episodes
        for i in range(episodes):
            # reset env to initial map, position, set elf to state 0, 8x8 -> 64 states: [0-63]
            state = env.reset()[0]

            # end simulation condition: elf falls into a hole or reached goal
            terminated = False
            # end simulation condition: number of actions > 200
            truncated = False

            # until one of above conditions is met
            while(not terminated and not truncated):

                # if generated number less than epsilon, then explore
                if random.random() < epsilon:
                    # select random action from available set: 0=left, 1=down, 2=right, 3=up
                    action = env.action_space.sample()
                else:
                    # exploit by taking the best action
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # return new state, reward and two simulation ending conditions after executing a simulation step with selected action
                new_state, reward, terminated, truncated, _ = env.step(action)

                # save experience into memory
                memory.append((state, action, new_state, reward, terminated))
                
                # move to new state
                state = new_state

                # increase step counter
                step_count += 1
                
            # track if in current episode reward has been collected
            if reward == 1:
                rewards_per_episode[i] = 1

            # check memory if enough training data and at least one reward has been collected to do optimization on 
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                # get batch of training data
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # decrease epslilon to go from less exploration to more exploitation
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # copy policy network to target network after certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        env.close()

        training_time = timer() - start_train
        print("Training took %f seconds" % training_time)

        # save weights/biases to file
        torch.save(policy_dqn.state_dict(), "frozen_lake_dqn.pt")

        # create new graph
        plt.figure(1)

        # plot rewards vs episodes
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        # plot on 1 row x 2 columns grid, at cell 1
        plt.subplot(121)
        plt.plot(sum_rewards)

        # plot epsilon decay vs episodes
        plt.subplot(122)
        plt.plot(epsilon_history)

        # save plots
        plt.savefig("frozen_lake_dqn.png")

    # optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get number of input nodes from policy network
        num_states = policy_dqn.fc1.in_features

        # output layer of policy network
        current_q_list = []
        # output layer of target network
        target_q_list = []

        # loop through training data in mini batch to replay experience
        for state, action, new_state, reward, terminated in mini_batch:

            # if agent reached the goal (reward=1) or fell into hole (reward=0)
            if terminated:
                # set target q value to the reward
                target = torch.FloatTensor([reward])
            else:
                # calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max())

            # take states and pass them to policy network to get current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # take states and pass them to policy target to get target set of Q values
            target_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            # adjust specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # calculate loss for whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        '''
        Converts an state (int) to a tensor representation.
        For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

        Parameters: state=1, num_states=16
        Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        '''
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor
    

    def test(self, episodes, is_slippery=False):
        # create frozen lake instance
        env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=is_slippery, render_mode='human')
        # get number of states and actions
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_dqn.pt"))
        # switch model to evaluation (prediction) mode
        policy_dqn.eval()

        print("Policy (trained): ")
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            # reset env, set agent to state 0
            state = env.reset()[0]
            # end simulation condition: agent falls into a hole or reached goal
            terminated = False
            # end simulation condition: number of actions > 200
            truncated = False

            while(not terminated and not truncated):
                with torch.no_grad():
                    # select best action
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # execute action
                state, reward, terminated, trunceted, _ = env.step(action)

        env.close()


    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states





if __name__ == "__main__":
    frozen_lake = FrozenLakeDQL()
    is_slippery=True
    frozen_lake.train(20000, is_slippery=is_slippery)
    frozen_lake.test(10, is_slippery=is_slippery)




