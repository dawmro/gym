import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import os
from datetime import datetime
import cv2


QTABLES_DIR = 'qtables'
if not os.path.exists(QTABLES_DIR):
    os.makedirs(QTABLES_DIR)

QTABLES_IMAGES_DIR = 'qtables_images'
if not os.path.exists(QTABLES_IMAGES_DIR):
    os.makedirs(QTABLES_IMAGES_DIR)


def showTime():
    return str("["+datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+" UTC]")


def train(episodes=25000, show_every=100):

    RENDER = False

    env = gym.make("MountainCar-v0", render_mode="human" if RENDER else None)

    LEARNING_RATE = 0.05
    DISCOUNT = 0.95

    EPISODES = episodes

    SHOW_EVERY = show_every


    # make it work on every env
    DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
    # how big is each step through chunk of space
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

    epsilon = 0.5
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


    # size equals every combination of possible env observations times action space, initialized with negative values
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


    def get_discrete_state(state):
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(np.int32))


    for episode in range(EPISODES):
        episode_reward = 0
        discrete_state = get_discrete_state(env.reset()[0])
        terminated = False
        truncated = False

        while(not terminated and not truncated):
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = env.action_space.sample()
            new_state, reward, terminated, truncated, _ = env.step(action=action)
            episode_reward += reward

            new_discrete_state = get_discrete_state(new_state)
            if(not terminated and not truncated):
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action, )] = new_q
            elif new_state[0] >= env.goal_position:
                #print(f"Solved on episode: {episode}")
                q_table[discrete_state + (action, )] = 0

            discrete_state = new_discrete_state

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        ep_rewards.append(episode_reward)
        
        if episode % SHOW_EVERY == 0:
            average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
            aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

            np.save(f"{QTABLES_DIR}/{episode}-qtable.npy", q_table)
            
            print(f"Episode: {episode} | avg: {average_reward} | min: {min(ep_rewards[-SHOW_EVERY:])} | max: {max(ep_rewards[-SHOW_EVERY:])}")

    env.close()

    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
    plt.legend(loc=4)
    plt.show()



def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3


def show_qtable_actions(episodes=25000, show_every=100):

    style.use('ggplot')

    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    i = (((episodes // show_every) * show_every) - show_every)
    q_table = np.load(f"{QTABLES_DIR}/{i}-qtable.npy")


    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
            ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
            ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])

            ax1.set_ylabel("Action 0")
            ax2.set_ylabel("Action 1")
            ax3.set_ylabel("Action 2")

    plt.show()


# good luck running that in any resonable time
def qtables_to_images(episodes=25000, show_every=100, overwrite=True):

    style.use('ggplot')

    fig = plt.figure(figsize=(12, 9))

    for i in range(0, episodes, show_every):
        print(F"{showTime()} Loop: {i}")
        if overwrite == False and os.path.isfile(f"{QTABLES_IMAGES_DIR}/{i}.png"):
            pass
        else:
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            q_table = np.load(f"{QTABLES_DIR}/{i}-qtable.npy")

            for x, x_vals in enumerate(q_table):
                for y, y_vals in enumerate(x_vals):
                    ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
                    ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
                    ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])

                    ax1.set_ylabel("Action 0")
                    ax2.set_ylabel("Action 1")
                    ax3.set_ylabel("Action 2")

            #plt.show()
            plt.savefig(f"{QTABLES_IMAGES_DIR}/{i}.png")
            plt.clf()


def make_video(episodes=25000, show_every=100):
    # windows:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Linux:
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('qlearn.avi', fourcc, 60.0, (1200, 900))

    for i in range(0, episodes, show_every):
        img_path = f"{QTABLES_IMAGES_DIR}/{i}.png"
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


if __name__ == "__main__":
    episodes=30000
    show_every=10

    #train(episodes=episodes, show_every=show_every)
    #show_qtable_actions(episodes=episodes, show_every=show_every)
    #qtables_to_images(episodes=episodes, show_every=show_every, overwrite=False)
    #make_video(episodes=episodes, show_every=show_every)
