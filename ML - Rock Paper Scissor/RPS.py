import numpy as np
import random
from RPS_game import mrugesh, abbey, quincy, kris

actions = ['R', 'P', 'S']
states = [(a, b, c, x, y, z) for a in actions for b in actions for c in actions for x in actions for y in actions for z in actions]
state_idx = {state: i for i, state in enumerate(states)}

Q = np.zeros((len(states), len(actions)))

alpha = 0.2
gamma = 0.9
epsilon_base = 0.1
decay_rate = 0.995

player_previous_action = None
opponent_previous_action = 'None'
opponent_history = []
player_history = []
experience_buffer = []
max_buffer_size = 150
train = 'None'

def get_state():
    if len(player_history) < 4:
        return ('P', 'P', 'P', 'P', 'P', 'P')
    if len(player_history) >= 4:
        state = (player_history[-3], player_history[-2], player_history[-1], opponent_history[-3], opponent_history[-2], opponent_history[-1])
        return state

def get_reward(player_action, opponent_action):
    if (player_action == 'R' and opponent_action == 'S') or \
       (player_action == 'P' and opponent_action == 'R') or \
       (player_action == 'S' and opponent_action == 'P'):
        return 1
    elif player_action == opponent_action:
        return 0
    else:
        return -2

def player(prev_play):
    global Q, alpha, gamma, epsilon_base, decay_rate, player_history, opponent_history, player_previous_action, opponent_previous_action, experience_buffer, max_buffer_size, train
    
    if train != "Done":
        print("Done")
        train = game(agent, abbey, 10000)
        train = game(agent, kris, 10000)
        train = game(agent, mrugesh, 10000)
        train = game(agent, quincy, 1000)
    
    if prev_play:
        opponent_previous_action = prev_play
        opponent_history.append(prev_play)

    next_state = get_state()  # estado del ultimo juego

    if len(player_history) >= 4 and len(opponent_history) >= 4:
        current_state = (player_history[-4], player_history[-3], player_history[-2], opponent_history[-4], opponent_history[-3], opponent_history[-2])
        action = player_history[-1]
        reward = get_reward(action, opponent_history[-1])
        experience = {'current_state': current_state, 'action': action, 'reward': reward, 'next_state': next_state}
        experience_buffer.append(experience)

        if len(experience_buffer) > max_buffer_size:
            experience_buffer.pop(0)

        Q_current = Q[state_idx[current_state], actions.index(action)]
        Q[state_idx[current_state], actions.index(action)] += alpha * (reward + gamma * np.max(Q[state_idx[next_state], :]) - Q_current)

        if len(experience_buffer) > 1:
            random_experience = random.choice(experience_buffer[:-1])
            random_current_state = random_experience['current_state']
            random_action = random_experience['action']
            random_reward = random_experience['reward']
            random_next_state = random_experience['next_state']
            random_Q_current = Q[state_idx[random_current_state], actions.index(random_action)]
            Q[state_idx[random_current_state], actions.index(random_action)] += alpha * (random_reward + gamma * np.max(Q[state_idx[random_next_state], :]) - random_Q_current)

    epsilon = epsilon_base * (decay_rate ** len(player_history))
    action = np.random.choice(actions) if np.random.uniform(0, 1) < epsilon else actions[np.argmax(Q[state_idx[next_state], :])]  # accion del juego futuro
    player_previous_action = action
    player_history.append(action)

    return action

def agent(prev_play):
    global Q, alpha, gamma, epsilon_base, decay_rate, player_history, opponent_history, player_previous_action, opponent_previous_action, experience_buffer

    if prev_play:
        opponent_previous_action = prev_play
        opponent_history.append(prev_play)

    next_state = get_state()  # estado del ultimo juego

    if len(player_history) >= 4 and len(opponent_history) >= 4:
        current_state = (player_history[-4], player_history[-3], player_history[-2], opponent_history[-4], opponent_history[-3], opponent_history[-2])
        action = player_history[-1]
        reward = get_reward(action, opponent_history[-1])
        experience = {'current_state': current_state, 'action': action, 'reward': reward, 'next_state': next_state}
        experience_buffer.append(experience)

        if len(experience_buffer) > max_buffer_size:
            experience_buffer.pop(0)

        Q_current = Q[state_idx[current_state], actions.index(action)]
        Q[state_idx[current_state], actions.index(action)] += alpha * (reward + gamma * np.max(Q[state_idx[next_state], :]) - Q_current)

        if len(experience_buffer) > 1:
            random_experience = random.choice(experience_buffer[:-1])
            random_current_state = random_experience['current_state']
            random_action = random_experience['action']
            random_reward = random_experience['reward']
            random_next_state = random_experience['next_state']
            random_Q_current = Q[state_idx[random_current_state], actions.index(random_action)]
            Q[state_idx[random_current_state], actions.index(random_action)] += alpha * (random_reward + gamma * np.max(Q[state_idx[random_next_state], :]) - random_Q_current)

    epsilon = epsilon_base * (decay_rate ** len(player_history))
    action = np.random.choice(actions) if np.random.uniform(0, 1) < epsilon else actions[np.argmax(Q[state_idx[next_state], :])]  # accion del juego futuro
    player_previous_action = action
    player_history.append(action)

    return action

def game(player1, player2, num_games, verbose=False):
    p1_prev_play = ""
    p2_prev_play = ""
    results = {"p1": 0, "p2": 0, "tie": 0}

    for _ in range(num_games):
        p1_play = player1(p2_prev_play)
        p2_play = player2(p1_prev_play)

        if p1_play == p2_play:
            results["tie"] += 1
        elif (p1_play == "P" and p2_play == "R") or (
                p1_play == "R" and p2_play == "S") or (p1_play == "S"
                                                       and p2_play == "P"):
            results["p1"] += 1
        elif p2_play == "P" and p1_play == "R" or p2_play == "R" and p1_play == "S" or p2_play == "S" and p1_play == "P":
            results["p2"] += 1

        p1_prev_play = p1_play
        p2_prev_play = p2_play

    train = "Done"

    return train