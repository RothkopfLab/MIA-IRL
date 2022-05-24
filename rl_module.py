#In rl_module.py the RlModule class is implemented, which contains all functionality
#for Q-Learning in a given grid world. The Q-learning algorithm chooses the action
#proposed by the learned Q-values if no advice is given. If advice is given, the
#action proposed by fusion_module and action_selection_module is selected. 



import time
import numpy as np
import matplotlib.pyplot as plt

from modality import Modality



class RlModule:

    def __init__(self, mdp, advice, action_selection_module=None, fusion_method=None, fusion_module=None,
                 simulated_advice=False, false_actions=[0,1,2,3], sim_probs_correct=1,
                 sim_certain=False, sim_lower_bound=0.5, sim_upper_bound=0.99):
        '''
        constructor of RlModule

        :param mdp: Mdp instance
        :param advice: if true than use advice, if false use no advice
        :param action_selection_module: ActionSelectionModule instance
        :param fusion_method: name of the fusion method that is used
        :param fusion_module: FusionModule instance, only required if simulated_advice is true
        :param simulated_advice: true if simulated advice should be used, false otherwise
        :param false_actions: permutation of confused actions if classifier simulations are wrong
        :param sim_probs_correct: contains probabilities that modality classifier mi is correct
        :param sim_certain: if True simulated distribution is a certain one
        :param sim_lower_bound: lower bound of random generated simulated distributions
        :param sim_upper_bound: upper bound of random generated simulated distributions
        '''

        self.mdp = mdp
        self.advice = advice
        self.action_selection_module = action_selection_module
        self.fusion_method = fusion_method
        self.fusion_module = fusion_module
        self.simulated_advice = simulated_advice
        self.false_actions = false_actions

        # if simulated advice is given and multiple modalities are fused
        if self.advice and self.simulated_advice and self.fusion_module is not None:
            #collect all modalities in self.m
            self.nr_modalities = sim_probs_correct.shape[0]
            self.m = []
            for i in range(self.nr_modalities):
                self.m.append(Modality(self.mdp.n_actions, self.false_actions))
            

            self.sim_probs_correct = sim_probs_correct
            self.sim_certain = sim_certain
            self.sim_lower_bound = sim_lower_bound
            self.sim_upper_bound = sim_upper_bound

        self.q = np.array([])  # table for q function
        self.v = np.array([])  # table for visits

        self.reset()



    def __my_argmax(self, array):
        '''
        returns list with all indexes of the max element in the array

        :param array: array whose maximum should be found
        :return: list index of max element
        '''
        maximum = np.amax(array)
        argmax = []
        for index in range(array.shape[0]):
            if array[index] == maximum:
                argmax.append(index)
        return argmax



    def reset(self):
        '''
        reset the Module, initialize q function and visits and reset advice module

        '''
        self.mdp.reset()

        # initialize Q table
        self.q = np.zeros((self.mdp.n_states, self.mdp.n_actions))
        self.q = self.mdp.compute_available_actions_Q(self.q)

        # visited counter
        self.v = np.zeros(self.mdp.n_states)



    def q_learning(self, discount=0.98, max_episode=False, max_episodes=0, n_advice=6, save=False, load=False,
                   render=False, execute_on_robot=False):
        '''
            Q-learning algorithm, agent learns the policy to reach a goal in a grid world

            :param discount: discount factor
            :param max_episode: if this is true then q_learning is executed for the number of episodes
             given by episodes, otherwise until convergence
            :param max_episodes: number of episodes to be executed if max_episode is True
            :param n_advice: number of episodes to give advice
            :param save: if True than the advice is saved in location
            :param load: if True than the advice is loaded from location
            :param render: True if environment should be rendered
            :param execute_on_robot: True if actions should be executed on robot
            :return: episode need for convergence, average success, average rewards
            :rtype: (int, np.array, np.array, np.array, int, np.array)
        '''

        avg_success = np.array([])
        avg_deviation = np.array([])
        avg_rewards = np.array([])

        # for convergence
        convergent = False
        con_counter = 0
        conv_episode = 0

        episode = 0
        i = 0

        while not convergent if not max_episode else i < max_episodes:
            # start state
            start = self.mdp.reset()
            if render:
                self.mdp.render()
            s = self.mdp.get_state_as_int(start)
            steps = 0

            while not self.mdp.finished and steps < self.mdp._max_episode_steps:

                # increment state visited
                self.v[s] = self.v[s] + 1
                epsilon = 0.1
                alpha = 1 / self.v[s]

                # fusion of human action advice input
                dis_fused = None
                p = np.random.uniform()
                if self.advice and i < n_advice:
                    # if we want to use simulated advice
                    if self.simulated_advice and self.fusion_module is not None:
                        dists = np.zeros((self.nr_modalities, 4))
                        modality_indices = range(self.nr_modalities)
                        modality_indices[0] = 1 #this is done to not change the resulting convergence plots due to random effects after reorganizing the code
                        modality_indices[1] = 0
                        for j in modality_indices:
                            dists[j] = self.m[j].get_simulated_input(s, self.sim_probs_correct[j],
                                                                     self.sim_certain, self.sim_lower_bound,
                                                                     self.sim_upper_bound)

                        #fuse
                        dis_fused = self.fusion_module.get_fused_pruned_output(dists, self.q[s])


                # discrete action from q function
                rand = np.random.uniform(0, 1)

                # exploitation
                if rand > epsilon:
                    a_pi = np.random.choice(self.__my_argmax(self.q[s]))
                # exploration
                else:
                    a_pi = np.random.choice(self.mdp.n_actions)

                if dis_fused is not None:
                    # take input action
                    action = self.action_selection_module.select_action(dis_fused, a_pi)
                else:
                    action = a_pi


                s_, r, finished, info = self.mdp.step(action)
            
                s_ = self.mdp.get_state_as_int(s_)
                self.q[s, action] = self.q[s, action] + alpha * (
                        r + discount * np.amax(self.q[s_]) - self.q[s, action])

                s = s_
                steps += 1
                if render:
                    self.mdp.render()
                    time.sleep(0.1)

            accuracy, deviation, reward = self.evaluate(100)
            avg_success = np.append(avg_success, accuracy)
            avg_deviation = np.append(avg_deviation, deviation)
            avg_rewards = np.append(avg_rewards, reward)
            episode += 1
            i += 1


            # calculate if learning curve converges
            # if not max_episode:
            if accuracy >= 0.9999:
                con_counter += 1
            if con_counter == 10:
                convergent = True
                conv_episode = i


        if conv_episode == 0:
            conv_episode = max_episodes

        return episode, avg_success, avg_deviation, avg_rewards, conv_episode, self.q

    def evaluate(self, test_episodes=100):
        '''
        evaluates the learned policy over 100 episodes

        :param test_episodes: number of episodes to run
        :return: average accuracy and average reward
        '''
        success = np.array([])
        reward = 0
        for n in range(test_episodes):
            # random start state
            self.mdp.init_object_states = None
            start = self.mdp.reset()
            s = self.mdp.get_state_as_int(start)
            steps = 0
            while not self.mdp.finished and steps < self.mdp._max_episode_steps:
                a = np.random.choice(self.__my_argmax(self.q[s]))
                s, r, finished, info = self.mdp.step(a)
                #self.mdp.finished = finished
                s = self.mdp.get_state_as_int(s)
                reward += r
                steps += 1
            success = np.append(success, self.mdp.success)

        accuracy = np.mean(success)
        deviation = np.std(success)
        reward = reward / test_episodes
        return accuracy, deviation, reward
