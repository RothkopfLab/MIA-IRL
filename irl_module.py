#irl_module.py is the starting point for reproducing the simulated results given in
#the paper. It contains the functionality for evaluating all compared (I)RL approaches
#and plotting the resulting learning curves.



import os
import sys
import numpy as np

from rl_module import RlModule
from envs.taxi import GridWorldHoles
from action_selection_module import ActionSelectionModule
from fusion_module import FusionModule

#imports and settings for plots
fs = 16 #fontsize for plots
import matplotlib
matplotlib.rc('xtick', labelsize=fs) 
matplotlib.rc('ytick', labelsize=fs)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'



#Interactive Reinforcement Learning (IRL) Module
class IRLModule:

    
    def __init__(self, probs_correct, false_actions=None):
    	'''
    	constructor of IRLModule, sets parameters for simulated advice

    	:param probs_correct: contains probabilities that modality classifier mi is correct
    	:param false_actions: permutation of confused actions if classifier simulations are
         wrong, default None
    	'''

        #set mdp to gridworld holes, the maximum number of episodes to find the goal is set to 15
        self.mdp = GridWorldHoles(max_episode_steps=15)

        #set parameters for simulated advice
        self.probs_correct = probs_correct
        self.false_actions = false_actions



    def evaluate_different_strategies(self, types=['q_learning'], runs=20, episodes=100, nr_advice=10,
                                      render=False):
    	'''
    	evaluates all (I)RL approaches given in types and plots their learning curves

    	:param types: list of approaches to be compared, default only standard Q-learning
    	:param runs: number of repeated runs that are simulated for evaluating, default 20
    	:param epsisodes: number of maximum episodes that are executed in a run, default 100
    	:param nr_advice: number of episodes in which advice is given in the beginning of each run, default 10
    	:param render: True if grid world and agent moving around in it should be shown
    	'''
        rewards = np.zeros((len(types), runs, episodes))
        Q = np.zeros((len(types), runs, self.mdp.n_states, self.mdp.n_actions))


        for i, type in enumerate(types):
            print('evaluating %s...' %type)
            rewards[i, :, :], Q[i, :, :, :] = self.evaluate(type, runs, episodes, nr_advice, render)


        self.plot_evaluation(types, rewards)
        


    def evaluate(self, type='q_learning', runs=20, episodes=100, nr_advice=10, render=False):
    	'''
    	evaluates multiple runs of the (I)RL approach given in type.

    	:param type: the type of approach, e.g. q_learning (default) for standard Q-learning, iop for fusing
         advice with IOP
    	:param runs: nr of repeated runs that are simulated for evaluating, default 20
    	:param episodes: number of maximum episodes that are executed in a run, default 100
    	:param nr_advice: number of episodes in which advice is given in the beginning of each run, default 10
    	:param render: True if grid world and agent moving around in it should be shown
    	:return: rewards array and array Q containing the Q values and 
    	'''

        #containers for all accuracies, reards, and the number of episodes needed for convergence for each run
        all_accuracies = np.zeros((runs, episodes))
        all_rewards = np.zeros_like(all_accuracies)
        conv_episodes = np.zeros(runs)

        #array that stores all q values
        Q = np.zeros((runs, self.mdp.n_states, self.mdp.n_actions))
        
        #standard Q-learning without advice
        if type == 'q_learning':
            #set RL module
            rl_module = RlModule(self.mdp, advice=False)
        
        else:
            #Q-learning with advice fused according to Cruz
            if type == 'cruz':
                fusion_method = 'Cruz'
                fusion_module = FusionModule(fusion_method)
                #action are selected with argmax method and additionally only probabilities over a threshold
                #are considered
                action_selection_module = ActionSelectionModule(self.mdp, 'argmax_threshold')
                probs_correct = self.probs_correct[:2] #only consider first two probs_correct for two classifiers

            
            #Q-learning with advice fused with IOP
            elif type == 'iop':
                fusion_method = 'IOP'
                fusion_module = FusionModule(fusion_method)
                #actions are selected by sampling from the complete fused distribution
                action_selection_module = ActionSelectionModule(self.mdp, 'sampling')
                probs_correct = self.probs_correct[:2] #only consider first two probs_correct for two classifiers


            #Q-learning with advice fused with IOP using 3 advice modalities
            elif type == '3m':
                fusion_method = 'IOP'
                fusion_module = FusionModule(fusion_method)
                #actions are selected by sampling from the complete fused distribution
                action_selection_module = ActionSelectionModule(self.mdp, 'sampling')
                probs_correct = self.probs_correct #use all probs_correct


            #set RL module
            rl_module = RlModule(self.mdp, True, action_selection_module,
                                fusion_method, fusion_module=fusion_module,
                                simulated_advice=True, false_actions=self.false_actions,
                                sim_probs_correct=probs_correct)

        #repeat Q-learning for multiple runs
        for i in range(runs):
            rl_module.reset()
            _, all_accuracies[i], _, all_rewards[i], conv_episodes[i], Q[i] = rl_module.q_learning(discount=0.98,
                                                                                                    max_episode=True,
                                                                                                    max_episodes=episodes,
                                                                                                    n_advice=nr_advice,
                                                                                                    load=True,
                                                                                                    render=render)
        
        #print the number of episodes needed for convergence for all runs
        print('%s: episodes needed to converge:' %type)
       	print(conv_episodes)

        #return the rewards and learned q values
        return all_rewards, Q



    def plot_evaluation(self, types, rewards):
    	'''
    	plots the learning curves of all types of IRL approaches in types using the given rewards

    	:param types: list of approach types that should be compared in the plot
    	:param rewards: the rewards to be plotted as learning curves, dim: nr_types x nr_runs x nr_episodes
    	'''

        #set up plot with size
        fig, ax = plt.subplots(figsize=(5,4))

        #set axes labels, ticks, and grid
        ax.set_ylabel('reward', fontsize=fs)
        ax.set_xlabel('episodes', fontsize=fs)
        ax.set_xlim([0,50])
        ax.set_xticks([0, 20, 40])
        ax.set_xticklabels(['0', '20', '40'])
        ax.set_ylim([-40,125])
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(['0', '50', '100'])

        ax.grid()

        #set colors and linestyles for different curves
        colors = ['b', 'g', 'r', 'black']
        linestyles = ['dotted', 'dashed', 'solid', 'dashdot']
        z_orders = [10, 20, 15, 5]

        #iterate through all approach types and plot their learning curves
        for i, type in enumerate(types):
            self.plot_learning_curve(rewards[i, :, :], type, ax, colors[i], linestyles[i], z_orders[i])

        plt.legend(fontsize=fs, handlelength=1.4, loc='best')
        plt.tight_layout()
        plt.show()



    def plot_learning_curve(self, rewards, type, ax, color, linestyle, zorder):
    	'''
    	plots the learning curve from given rewards (mean and standard deviation of rewards over episodes
        in multiple runs)
    	
    	:param rewards: the rewards to be plotted, dim: nr_runs x nr_episodes
    	:param type: the type of the plotted (I)RL approach, e.g. 'iop' if IOP fusion is used
    	:param ax: the plot's axis
    	:param color: the color of the respective learning curve
    	:param linestyle: the linestyle of the respective learning curve
    	:param zorder: defines the order of the learning curve in z direction
    	'''

        #compute mean and standard devation of given rewards over multiple runs
        mu = np.mean(rewards, axis=0)
        std = np.std(rewards, axis=0)

        #set labels for the plot's legend according to type
        if type == 'q_learning':
            label = 'Q-Learning'
        elif type == 'cruz':
            label = 'C-IRL'
        elif type == 'iop':
            label = 'MIA-IRL'
        elif type == '3m':
            label = 'MIA-IRL-3'

        #plot mean and standard deviation of learning curves
        ax.plot(mu, color=color, linestyle=linestyle, linewidth=2.5, label=label, zorder=zorder)
        ax.fill_between(range(len(mu)), mu-std, mu+std, color=color, alpha=0.3, zorder=zorder)





if __name__ == '__main__':

    #set random seed
    np.random.seed(1)

    #types of approaches that should be compared: standard non-interactive Q-learning,
    #IRL with IOP fusion, and IRL with Cruz's fusion method
    types = ['q_learning', 'cruz', 'iop']
    # types = ['q_learning', 'iop']


    #if True, renders the grid world and the agent moving around in it
    render = False

    #read command line arguments to decide which figure should be created, 3(b), (c), (d), or (e)
    run_args = sys.argv
    if len(run_args) > 1:
        run_mode = run_args[1]
    #if no argument is given, Figure 3(b) is created
    else:
        run_mode = 'b'

    #Figure 3(b): both classifiers are correct
    if run_mode == 'b':
        print('Replicating Figure 3(b):')
        prob_correct_m1 = 1.0
        prob_correct_m2 = 1.0
        probs_correct = np.array([prob_correct_m1, prob_correct_m2])
        irl_module = IRLModule(probs_correct)

    #Figure 3(c): modality classifier m1 is correct, modality classifier m2 confuses actions left and right
    elif run_mode == 'c':
        print('Replicating Figure 3(c):')
        prob_correct_m1 = 1.0
        prob_correct_m2 = 0.0
        probs_correct = np.array([prob_correct_m1, prob_correct_m2])
        false_actions = [0,1,3,2]  #0:up, 1:right, 2:left, 3:right
        irl_module = IRLModule(probs_correct, false_actions)

    #Figure 3(d): modality classifier m1 is correct, modality classifier m2 confuses actions left/right and up/down
    elif run_mode == 'd':
        print('Replicating Figure 3(d):')
        prob_correct_m1 = 1.0
        prob_correct_m2 = 0.0
        probs_correct = np.array([prob_correct_m1, prob_correct_m2])
        false_actions = [1,0,3,2]  #0:up, 1:right, 2:left, 3:right
        irl_module = IRLModule(probs_correct, false_actions)

    #Figure 3(e): in 20% of cases both classifiers are false, MIA-IRL-3 with 3 modalities additionally evaluated
    elif run_mode == 'e':
        print('Replicating Figure 3(e):')
        types = ['q_learning', 'cruz', 'iop', '3m'] #additional type 3m for MIA-IRL with 3 modalities
        prob_correct_m1 = 0.8
        prob_correct_m2 = 0.0
        prob_correct_m3 = 0.6
        probs_correct = np.array([prob_correct_m1, prob_correct_m2, prob_correct_m3])
        false_actions = [1,0,3,2]  #0:up, 1:right, 2:left, 3:right
        irl_module = IRLModule(probs_correct, false_actions)

    else:
        print('The provided argument %s is not supported.' %run_mode)
        sys.exit()


    #compare the learning curves of all aproaches in types, 50 random runs, max 200 episodes,
    #10 epsisodes advice
    irl_module.evaluate_different_strategies(types, runs=50, episodes=200, nr_advice=10, render=render)
