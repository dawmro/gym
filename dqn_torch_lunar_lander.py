import gymnasium as gym
from dqn_torch import Agent
from utils import plotLearning
import numpy as np


if __name__ == "__main__":
    env = gym.make('LunarLander-v2', render_mode="human")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.002)
    # track gent scores and epsilon
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        terminated = False
        truncated = False
        # done = truncated or terminated 
        observation = env.reset()[0]
        while(not terminated and not truncated):
            # choose action based on current state of env
            action = agent.choose_action(observation)
            # make a step
            observation_, reward, terminated, truncated, info, = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, terminated, truncated)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        # average score for last 100 games
        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
        # plot learning curve

    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)




