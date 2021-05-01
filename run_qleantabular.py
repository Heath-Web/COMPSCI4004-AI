from collections import defaultdict
import os
os.chdir('..')
from matplotlib import pyplot as plt
import numpy as np
from QLearningAgent import QLearningAgent
import virl

def q_to_u(agent):
    U = defaultdict(lambda: -1000.)
    for state_action, value in agent.Q.items():
        state, action = state_action
        if U[state] < value:
            U[state] = value
    return U

def plot_state_reward(states, rewards):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
    states = np.array(states)
    for i in range(4):
        axes[0].plot(states[:,i], label=labels[i]);
    axes[0].set_xlabel('weeks since start of epidemic')
    axes[0].set_ylabel('State s(t)')
    axes[0].legend()
    axes[1].plot(rewards);
    axes[1].set_title('Reward')
    axes[1].set_xlabel('weeks since start of epidemic')
    axes[1].set_ylabel('reward r(t)')
    axes[0].set_title("Problem id = 0  Noisy = False")
    plt.show()

def get_state_period(state,operate):
    return (int(state[0]/operate),int(state[1]/operate),int(state[2]/operate),int(state[3]/operate))

def run_single_trial(agent,env):
    """
    Execute trial for given agent_program
    and environment when training
    :param agent: Q-learning agent
    :param env: task environment (VIRL)
    :return total_reward: total reward of this single trail
    """
    total_reward = 0.0 # total reward for each iteration
    current_state = env.reset() # reset environment
    current_reward = 0 # current reward
    done = False
    i = None
    while True:
        current_state = get_state_period(current_state,15000000) # pre-process current state (scale down the number of the state)
        percept = (current_state, current_reward,done,i)
        next_action = agent(percept) # get next action
        if next_action is None:
            break
        current_state, current_reward, done, i = env.step(action=next_action) # take an action
        total_reward += current_reward # compute total reward
    return total_reward

def train(q_agent,env,num_of_iters):
    """
    This function is used to train a q learning agent with a given environment
    and number of iterations.
    :param q_agent: Q-learning agent
    :param env: task environment (VIRL)
    :param num_of_iters: number of iteration
    :return None
    """
    train_rewards = [] # list of rewards of all total rewards
    for i in range(num_of_iters):
        total_reward = run_single_trial(q_agent, env)
        train_rewards.append(total_reward)
        print("Episode/trial: "+ i + "  Total reward:" + total_reward)

def test(q_agent,env):
    current_state = env.reset()
    states = []
    rewards = []
    actions = []
    states.append(current_state)
    done = False
    while not done:
        action = q_agent.show_bestaction(get_state_period(current_state,15000000),done)
        actions.append(action)
        current_state, r, done, i = env.step(action)
        states.append(current_state)
        rewards.append(r)
    print(actions)
    return states,rewards

if __name__ == '__main__':
    # Training
    train_env = virl.Epidemic(problem_id = 0,stochastic=False, noisy=False) # training environment
    q_agent = QLearningAgent(train_env, epsilon=0.2, alpha=lambda n: 60./(59+n)) # initialize q learning agent, set up parameters
    num_of_iters = 2000 # number of iteration
    train(q_agent,train_env,num_of_iters) # training

    # First Test
    test_env = virl.Epidemic(problem_id = 0, stochastic=False, noisy=False)
    states, rewards = test(q_agent, test_env)
    plot_state_reward(states, rewards)

    # Second Test
    test_env = virl.Epidemic(problem_id = 2, stochastic=False, noisy=False)
    states, rewards = test(q_agent, test_env)
    plot_state_reward(states, rewards)

    # Third Test
    test_env = virl.Epidemic(problem_id = 0, stochastic=False, noisy=True)
    states, rewards = test(q_agent, test_env)
    plot_state_reward(states, rewards)


