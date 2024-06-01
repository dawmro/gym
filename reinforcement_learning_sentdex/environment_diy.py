import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time


style.use("ggplot")

# 10 by 10 grid
SIZE = 10
# how many episodes
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PEANLTY = 300
FOOD_REWARD = 25

epsilon = 0.1
EPS_DECAY = 0.9998
SHOW_EVERY = 10

start_q_table = "qtable-1717284764.pickle"

LEARNING_RATE = 0.1
DISCOUNT = 0.95

# keys in dict
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

# BGR color
d = {1: (255, 175, 0), 
     2: (0, 255, 0),
     3: (0, 0, 255)}


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def __str__(self):
        return f"{self.x}, {self.y}"
    
    # overload subtraction operator
    def __sub__(self, other):
        return (self.x - other.x, self.y-other.y)
    
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # fix for being out of bounds
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


# test if it works
player = Blob()
food = Blob()
enemy = Blob()
print(player)
print(food)
print(player-food)
player.move()
print(player-food)
player.action(2)
print(player-food)


# create or load q-table
if start_q_table is None:
    q_table = {}
    # observation space -> (x1, y1),(x2,y2), for every combination
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    # add every combination to table
                    # key in table is tuple of tuples, values initialized as random
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)

        player.action(action)

        #### todo
        enemy.move()
        food.move()
        ##########

        # contact with enemy
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PEANLTY
        # contact with food
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        # make new observation after movement
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        # calculate new Q
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PEANLTY:
            new_q = -ENEMY_PEANLTY
        else:
            # use q function
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # update q table
        q_table[obs][action] = new_q

        # show env
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8) 
            env[food.x][food.y] = d[FOOD_N]
            env[player.x][player.y] = d[PLAYER_N]
            env[enemy.x][enemy.y] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300))
            cv2.imshow("", np.array(img))
            # display screen until end conditions are met
            if reward == FOOD_REWARD or reward == -ENEMY_PEANLTY:
                # frames will stop refreshing for 500ms until key is pressed
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                # refreshing fast, frames will stop refreshing for 1ms until key is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # collect reward
        episode_reward += reward
        # break if end conditions are met
        if reward == FOOD_REWARD or reward == -ENEMY_PEANLTY:
            break
    
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)

        






