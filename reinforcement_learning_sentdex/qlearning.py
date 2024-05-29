import gymnasium as gym
import numpy as np

RENDER = False

env = gym.make("MountainCar-v0", render_mode="human" if RENDER else None)

LEARNING_RATE = 0.1
DISCOUNT = 0.95

EPISODES = 3000

SHOW_EVERY = 100


# make it work on every env
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# how big is each step through chunk of space
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


# size equals every combination of possible env observations times action space, initialized with negative values
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))


for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset()[0])
    terminated = False
    truncated = False

    while(not terminated and not truncated):
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = env.action_space.sample()
        new_state, reward, terminated, truncated, _ = env.step(action=action)

        new_discrete_state = get_discrete_state(new_state)
        if(not terminated and not truncated):
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"Solved on episode: {episode}")
            q_table[discrete_state + (action, )] = 1

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    if episode % SHOW_EVERY == 0:
        print(episode, new_state[0], env.goal_position)

env.close()

