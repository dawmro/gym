import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from timeit import default_timer as timer

def run(episodes, is_training=True, render=False):
    start = timer()

    # create FrozenLake env using 8x8 map, make it slippery and turn or rendering
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human" if render else None)

    if (is_training):
        # initialize q table, 64 x 4 array, defines the shape as a tuple
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open("frozen_lake8x8.pkl", "rb") as f:
            q = pickle.load(f)

    # initialize hyperparamater alpha
    learning_rate_a = 0.9
    # initialize hyperparamater gamma
    discount_factor_g = 0.9

    # epsilon greedy policy to balance exploration and exploitation
    # epsilon = 1 -> 100% random actions
    epsilon = 1
    # decay rate = 0.001 -> 1000 episodes to go to zero, train for more than this number
    epsilon_decay_rate = 0.0001
    # random number generator
    rng = np.random.default_rng()

    # initialize array to track episode rewards
    rewards_per_episode = np.zeros(episodes)

    # train for a given amount of episodes
    for i in range(episodes):
        # reset env to initial position, set elf to state 0, 8x8 -> 64 states: [0-63]
        state = env.reset()[0]

        # end simulation condition: elf falls into a hole or reached goal
        terminated = False
        # end simulation condition: number of actions > 200
        truncated = False

        # until one of above conditions is met
        while(not terminated and not truncated):

            # if generated number less than epsilon, then explore
            if is_training and rng.random() < epsilon:
                # select random action from available set: 0=left, 1=down, 2=right, 3=up
                action = env.action_space.sample()
            else:
                # exploit by taking the best action
                action = np.argmax(q[state,:])

            # return new state, reward and two simulation ending conditions after executing a simulation step with selected action
            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                # update table using q learning formula after taking a step
                q[state,action] = q[state,action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action])

            # assign new state
            state = new_state

        # decrease epslilon every step to go from less exploration to more exploitation
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # reduce learning rate to stabilize q values after exploration phase ended
        if(epsilon == 0):
            learning_rate_a = 0.0001

        # track if in current episode reward has been collected
        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    training_time = timer() - start
    print("This run took %f seconds" % training_time)

    # create graph for collected rewards
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if is_training:
        # save q table to file so it can be reused in the future
        with open("frozen_lake8x8.pkl","wb") as f:
            pickle.dump(q,f)

if __name__ == "__main__":
    #run(20000, is_training=True, render=False)
    run(1000, is_training=False, render=False)




