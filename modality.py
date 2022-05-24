#In modality.py the Modality class is implemented, which serves for simulating advice
#for the modalities speech and gestures. Thus, it simulates categorical output
#distributions, which normally would be the output of speech and gesture classifiers.
#The simulated adive can be either correct or incorrect, certain or uncertain.



import numpy as np
import matplotlib.pyplot as plt



class Modality:


    def __init__(self, n_actions, false_classes):
        '''
        constructor of Modality
        
        :param n_actions: number of actions
        :param false_classes: list that indicated which class is confused with which class
        '''
        self.n_actions = n_actions

        # select false action mapping
        self.false_classes = false_classes

        # load right q function for lookup
        self.lookup_q = np.load('policies/policy_grid_world_holes.npy')



    def get_simulated_input(self, s, prob_correct=1, certain=False, lower_prob=0.5, upper_prob=0.99):
        '''
        return simulated human input, extracted from learned q function

        :param s: current state
        :param prob_correct: probability that the modality returns correct action, should be between 0 and 1
        :param certain: if True a certain probability distribution is returned, otherwise a random one with specific properties
        :param lower_prob: lower bound for random distribution, should be between 0 and 1
        :param upper_prob: upper bound for random distribution, should be between 0 and 1
        :return: modality distribution, n.array
        '''
        true_action = np.argmax(self.lookup_q[s])
        rand = np.random.uniform(0, 1)

        # select how simulated distribution looks like
        if rand < prob_correct:
            if certain:
                dis = self.get_simulated_right_certain_distribution(true_action)
            else:
                dis = self.get_simulated_right_distribution(true_action, lower_prob=lower_prob, upper_prob=upper_prob)
        else:
            if certain:
                dis = self.get_simulated_false_certain_distribution(true_action)
            else:
                dis = self.get_simulated_false_distribution(true_action, lower_prob=lower_prob, upper_prob=upper_prob)

        return dis



    def __make_distribution(self, action, accuracy, deviation):
        '''
        generates a random distribution with a specific probability for action

        :param action: action that should have a specific probability
        :param accuracy: accuracy of the simulated input for true action
        :param deviation: deviation of the simulated input for true action
        '''
        distribution = np.zeros(self.n_actions)

        rand = np.random.uniform(accuracy - deviation, accuracy + deviation)
        rest = 1 - rand

        distribution[action] = rand
        other = []

        for i in range(self.n_actions):
            if i != action:
                rand = np.random.uniform(0, rest)
                distribution[i] = rand
                rest -= rand
                other.append(action)

        index = np.random.choice(other)
        distribution[index] += rest

        return distribution



    def plot_distribution(self, distribution):
        '''
        plots the input distribution in a bar plot

        :param distribution: distribution to be plotted
        '''
        plt.figure()
        plt.bar(range(self.n_actions), distribution)
        plt.xlabel('Actions')
        plt.ylabel('probability')
        plt.show()



    def get_simulated_right_distribution(self, true_action, lower_prob=0.7, upper_prob=0.99):
        '''
        generates a distribution with the highest probability at the index of the true action

        :param int true_action: index of true action
        :param lower_prob: lower probability bound
        :param upper_prob: upper probability bound
        :return: generated distribution
        '''

        dis = np.zeros(self.n_actions)
        p = np.random.uniform(lower_prob, upper_prob)
        dis[true_action] = p

        for i in range(dis.shape[0]):
            if (true_action != dis.shape[0] - 1 and i == dis.shape[0] - 1) or (
                    true_action == dis.shape[0] - 1 and i == dis.shape[0] - 2):
                dis[i] = 1 - p
            elif i != true_action:
                rand = np.random.uniform(0, 1 - p)
                p += rand
                dis[i] = rand

        return dis



    def get_simulated_right_certain_distribution(self, true_action):
        '''
        generates a distribution with the highest probability at the index of the true action with fixed probabilities

        :param true_action: index of true action
        :return: generated distribution
        '''
        if self.n_actions == 4:
            dis = np.ones(self.n_actions) * 0.05
            dis[true_action] = 0.85
        elif self.n_actions == 5:
            dis = np.ones(self.n_actions) * 0.05
            dis[true_action] = 0.8
        elif self.n_actions == 6:
            dis = np.ones(self.n_actions) * 0.04
            dis[true_action] = 0.8
        elif self.n_actions == 7:
            dis = np.ones(self.n_actions) * 0.04
            dis[true_action] = 0.76
        elif self.n_actions == 8:
            dis = np.ones(self.n_actions) * 0.03
            dis[true_action] = 0.79

        return dis



    def get_simulated_false_distribution(self, true_action, lower_prob=0.7, upper_prob=0.99):
        '''
        generates a distribution with the highest probability at the index that is not the true action

        :param true_action: index of true action
        :param lower_prob: lower probability bound
        :param upper_prob: upper probability bound
        :return: generated distribution
        '''
        false_action = self.false_classes[true_action]

        dis = np.zeros(self.n_actions)
        p = np.random.uniform(lower_prob, upper_prob)
        dis[false_action] = p

        for i in range(dis.shape[0]):
            if (false_action != dis.shape[0] - 1 and i == dis.shape[0] - 1) or (
                    false_action == dis.shape[0] - 1 and i == dis.shape[0] - 2):
                dis[i] = 1 - p
            elif i != false_action:
                rand = np.random.uniform(0, 1 - p)
                p += rand
                dis[i] = rand

        return dis



    def get_simulated_false_certain_distribution(self, true_action):
        '''
        generates a distribution with the highest probability at the index of the true action with fixed probabilities

        :param true_action: index of true action
        :return: generated distribution
        '''
        false_action = self.false_classes[true_action]

        if self.n_actions == 4:
            dis = np.ones(self.n_actions) * 0.05
            dis[false_action] = 0.85
        elif self.n_actions == 5:
            dis = np.ones(self.n_actions) * 0.05
            dis[false_action] = 0.8
        elif self.n_actions == 6:
            dis = np.ones(self.n_actions) * 0.04
            dis[false_action] = 0.8
        elif self.n_actions == 7:
            dis = np.ones(self.n_actions) * 0.04
            dis[false_action] = 0.76
        elif self.n_actions == 8:
            dis = np.ones(self.n_actions) * 0.03
            dis[false_action] = 0.79

        return dis



    def get_uniform_distribution(self):
        dis = np.ones(self.n_actions) * 1/self.n_actions
        return dis
