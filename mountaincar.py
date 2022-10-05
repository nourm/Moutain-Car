import numpy as np
import matplotlib.pyplot as plt
import gym
# a table containing the several positions
# in this case I used start = -1.2, stop = 0.6 and a 20 sample to generate
pos_space = np.linspace(-1.2, 0.6, 20)

# a table containing the several velocities
vel_space = np.linspace(-0.07, 0.07, 20)


def get_state(observation):
    # obs is a list containing two elemnts: position of the car and its velocity
    pos, vel = observation

    # defining bins taking the given position from the observation and the position space, same for velocity
    # a bin defines the number of equal-width bins in a given range
    # Its purpose is to analyze the frequency of quantitative data that covers a range of possible values
    pos_bin = int(np.digitize(pos, pos_space))
    vel_bin = int(np.digitize(vel, vel_space))

    return (pos_bin, vel_bin)


# function to return states depending on observations on velocity and position


def max_action(Q, state, actions=[0,1,2]):
    #constructing an array for all the values in our action space
    values = np.array(Q[state,a] for a in actions)
    action = np.argmax(values)

    return action
# main code
if __name__ == '__main__':
    # gym.make at its core calls the constructor corresponding to the environment id
    env = gym.make('MountainCar-v0')
    env.max_episode_steps=1000
    n_games = 4000
    alpha = 0.1
    gamme= 0.99
    eps = 1.0

    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos,vel))

    Q= {}
    for state in states:
        for action in [0,1,2]:
            Q[state,action] = 0

    score = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        done = False
        obs = env.reset()
        state = get_state(obs[0])
        #print(obs[0], state)
        if i % 1000 == 0 and i> 0:
            print('episode', i, 'score', score, 'epsilone %.3f', eps)
        score = 0
        while not done:
            action = np.random.choice([0,1,2]) if np.random.random() < eps else max_action(Q, state)
            obs_, reward, done, autreino, info = env.step(action)
            state_ = get_state(obs_)
            score += reward
            action_ = max_action(Q, state_)
            Q[state, action] = Q[state, action] + alpha *(reward + gamme*Q[state_, action_] - Q[state,action])
            state = state_

        total_rewards[i] = score
        eps = eps - 2/n_games if eps >0.01 else 0.01

    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0,t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig('mountaincar.png')

