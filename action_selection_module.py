#In action_selection_module.py the ActionSelectionModule class is implemented, which
#contains different methods for selecting an action from a categorical distribution
#over actions. In particular, the action can be chosen by sampling, as we propose
#for our approach, or by taking the action with the highest probability if it is
#above a specified threshold, as proposed by the compared approach by Cruz et al. (2018).



import numpy as np



class ActionSelectionModule:


    def __init__(self, mdp, action_selection_method):
        '''
        constructor of ActionSelectionModule

        :param mdp: instance of MDP
        :param action_selection_method: name of the method to use for the action selection, 'all_probabilities', 'highest_probability', 'highest_probability_threshold'
        '''

        self.mdp = mdp
        self.action_selection_method = action_selection_method



    def select_action(self, dis, a_pi=None):              
        '''
        selects action with highest probability directly from human input given as a distribution dis

        :param dis: human action distribution
        :param a_pi: action suggested by policy
        :return: selected action, int
        '''

        if self.action_selection_method == 'argmax_threshold':
            action = self.__select_action_with_highest_probability_threshold(dis, a_pi)
        elif self.action_selection_method == 'sampling':
            action = self.__select_action_from_distribution(dis)
        else:
            action = self.__select_action_with_highest_probability(dis)

        return action


    def __select_action_with_highest_probability(self, dis):
        """
        selects the action with the highest probability
        in contrast to standard argmax, if multiple classes have the same (highest) probability,
        the choice among these is also random

        :param np.array dis: distribution from which action should be selected
        :return: selected action, int
        """
        max = np.max(dis)
        maxes = []

        for i in range(dis.shape[0]):
            if dis[i] == max:
                maxes.append(i)

        idx = np.random.choice(maxes)
        return idx



    def __select_action_with_highest_probability_threshold(self, dis, a_pi):
        '''
        selects the action with the highest probability, if probability smaller than threshold then return policy action

        :param dis: distribution from which action should be selected
        :param a_pi: action suggested by policy
        :return: selected action, int
        '''
        a = self.__select_action_with_highest_probability(dis)
        if dis[a] < 0.25:
            # print 'Probability below Threshold: take policy action'
            a = a_pi

        return a



    def __select_action_from_distribution(self, dis):
        '''
        selects action form distribution, samples over distribution

        :param dis: distribution from which action should be selected
        :return: selected action, int
        '''
        p = np.random.uniform(0, 1)

        sum_up = 0
        for i in range(dis.shape[0]):
            if dis[i] != -np.Inf:
                if sum_up <= p <= sum_up + dis[i]:
                    return i
                else:
                    sum_up += dis[i]
