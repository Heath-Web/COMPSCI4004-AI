from collections import defaultdict
import numpy as np

class QLearningAgent:
    """
    An exploratory Q-learning agent. It avoids having to learn the transition
    model because the Q-value of a state can be related directly to those of
    its neighbors
    """

    def __init__(self, env, epsilon, alpha=None):
        self.gamma = 0.9
        self.all_act = (0,1,2,3)

        self.epsilon  = epsilon
        self.Q = defaultdict(float)
        self.Nsa = defaultdict(float)
        self.s = None
        self.a = None
        self.r = None

        if alpha:
            self.alpha = alpha # learning Rate
        else:
            self.alpha = lambda n: 1. / (1 + n)  # udacity video


    def actions_in_state(self, done):
        """
        Return actions possible in given state.
        Useful for max and argmax.
        """
        if done:
            return [None]
        else:
            return self.all_act

    def __call__(self, percept):
        """
        used for training, the process of  Q Learning

        :param percept: contain 4 params, current state , current reward, done , i
        :return: the next action
        """
        alpha = self.alpha # learning rate
        gamma = self.gamma # discount rate
        Q, Nsa = self.Q, self.Nsa
        s, a, r = self.s, self.a, self.r

        actions_in_state = self.actions_in_state

        s1, r1, done, i = percept  # current state and reward;  s' and r'

        # update Q value
        if a is not None:
            Nsa[s, a] += 1
            Q[s, a] = Q[s,a] + alpha(Nsa[s, a]) * (r + gamma * max(Q[s1, a1] for a1 in actions_in_state(done)) - Q[s, a])

        # Update for next iteration
        if done:
            self.s = self.a = self.r = None
        else:
            self.s, self.r = s1, r1
            epsilon = self.epsilon
            if np.random.uniform(0, 1, 1)[0] < epsilon:
                # Explore
                self.a = np.random.choice(range(4), 1)[0]
            else:
                # Exploit
                self.a = max(actions_in_state(done), key=lambda a1: Q[s1,a1])
        return self.a


    def show_bestaction(self,state,done):
        """
        Used for test, give the best action according to the q table

        :param state: current state
        :param done: Each episode ends after 52 weeks
        :return: the best action
        """
        actions_in_state = self.actions_in_state

        a = max(actions_in_state(done), key = lambda a1: self.Q[state,a1])
        print(state,self.Q[state,a])
        return a




