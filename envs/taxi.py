import gym
import os
from gym import spaces
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from gym.envs.classic_control import rendering

# Entries in the grid:
FREE, START, WALL, AGENT, PASSENGER, HOLE, GOAL = range(7)
COLORS = {WALL: [0, 0, 0], AGENT: [1, 1, 0], PASSENGER: [0.2, 0.9, 0.2],
          HOLE: [0.8, 0.1, 0], GOAL: [0.2, 0.8, 0.2]}
IMGS = {AGENT: "taxi", PASSENGER: "passenger", GOAL: "flag", HOLE: "flame"}  #
ACTION_VIS = {0: (180, 0, 0.31), 1: (0, 0, -0.31), 2: (270, -0.31, 0), 3: (90, 0.31, 0)}


# Actions: up, down, left, right


class AbstractTaxi(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, plan):
        self._seed = 0

        self._max_episode_steps = np.inf
        self._performed_steps = 0
        self.success_prob = 1
        self.rewards = [[100, 200, 300, 400], -100, 0]  # reaching goal with (0, 1, 2, 3) passengers, hole, wall

        ''' initialize system state '''
        self.this_file_path = os.path.dirname(os.path.realpath(__file__))
        grid_map_path = os.path.join(self.this_file_path, "grid_plans", plan)
        self.grid_map = self._read_grid_map(grid_map_path)
        self.passenger_list = np.argwhere(self.grid_map == PASSENGER).tolist()
        self.n_passengers = len(self.passenger_list)
        size_passengers = tuple(np.ones(self.n_passengers).astype(int))
        self.action_pos_dict = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.zeros(2 + self.n_passengers),
                                            np.array(self.grid_map.shape + size_passengers))

        self.n_actions = int(self.action_space.n)
        state_ranges = (self.observation_space.high - self.observation_space.low)
        self.n_states = int(np.product(state_ranges)) * (self.n_passengers + 1)
        self.viewer = None
        self.step_wait = 750

        self.agent_state = self._get_start_state()
        self.action_to_vis = None

        self.finished = False
        self.success = False

    def get_state_as_int(self, state=None):
        if state is None:
            state = self.agent_state

        s_int = 0
        m = 1

        for i in range(len(state)):
            s_int += state[i] * m
            m = m * self.observation_space.high[i] - self.observation_space.low[i]

        return int(s_int)

    def compute_available_actions_Q(self, Q):
        Q = np.zeros((self.n_states, self.n_actions))
        '''
        for s in range(self.n_states):
            av_actions = self.get_available_actions_int(s, False)
            for a in range(self.n_actions):
                if a in av_actions:
                    pass
                else:
                    Q[s, a] = -np.Inf
        '''
        return Q

    def step(self, action):

        """ return next observation, reward, finished, success """
        done = self._performed_steps > self._max_episode_steps
        self.finished = done
        self._performed_steps += 1

        info = {"success": False, "disaster": False}

        action = int(action)
        if action == -1:  # aborted action by user
            return self.agent_state, 0, done, info

        # if np.random.rand() < self.success_prob:
        # action = int(action)
        # else:
        # action = np.random.choice(4)

        next_pos = self.agent_state[:2] + self.action_pos_dict[action]
        if not self._valid(next_pos):  # Agent walks against the border:
            self.finished = done
            return self.agent_state, 0, done, info

        field_to_go = self.grid_map[next_pos[0], next_pos[1]]

        if field_to_go == PASSENGER:
            self.agent_state[self._passenger_id(next_pos)] = 1
        if not field_to_go == WALL:
            self.agent_state[:2] = next_pos
        if field_to_go == WALL:
            self.finished = False
            return self.agent_state, self.rewards[2], False, info
        if field_to_go == GOAL:
            self.success = True
            self.finished = True

            if self.agent_state[2:].sum() == self.n_passengers:
                reward = self.rewards[0][self.agent_state[2:].sum()]
                info = {"success": reward == self.rewards[0][self.n_passengers], "disaster": False}
                return self.agent_state, reward, True, info
        elif field_to_go == HOLE:
            info = {"success": False, "disaster": True}
            self.finished = True
            # self.success = False
            return self.agent_state, self.rewards[1], True, info
        return self.agent_state, 0, done, info

    def reset(self):
        self._performed_steps = 0
        self.agent_state = self._get_start_state()
        self.success = False
        self.finished = False
        return self.agent_state

    def render(self, mode='human', close=False):

        def flip(i):
            return self.grid_map.shape[0] - i - 1

        fig, ax = plt.subplots(figsize=self.grid_map.T.shape, dpi=64)
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        background = np.ones((self.grid_map.shape[0], self.grid_map.shape[1], 3))
        imgs = []
        for i in range(self.grid_map.shape[0]):
            for j in range(self.grid_map.shape[1]):
                current = self.grid_map[i, j]
                if self.agent_state[0] == i and self.agent_state[1] == j:
                    current = AGENT
                    if self.action_to_vis is not None:

                        if self.action_to_vis == -1:
                            img_name = os.path.join(self.this_file_path, "img/aborted.png")
                            img = Image.open(img_name)
                            if img.mode != "RGBA":
                                img = img.convert('RGBA')
                            extent = [j - 0.35, j + 0.35, flip(i) - 0.35, flip(i) + 0.35]
                            imgs += [(extent, img)]

                            # self.action_to_vis =0

                        # if self.action_to_vis.action == -1:
                        # print('vis aborting action')
                        # else:
                        else:
                            img_name = os.path.join(self.this_file_path, "img/down.png")
                            rot, bias_x, bias_y = ACTION_VIS[self.action_to_vis]
                            img = Image.open(img_name).rotate(rot, expand=True)
                            if img.mode != "RGBA":
                                img = img.convert('RGBA')
                            w, h = img.size[0] / 64, img.size[1] / 64
                            x = j + bias_x
                            y = flip(i) + bias_y
                            extent = [x - w, x + w, y - h, y + h]
                            imgs += [(extent, img)]
                elif (current == PASSENGER and
                      self.agent_state[self._passenger_id([i, j])]):
                    current = FREE
                if current in COLORS.keys():
                    background[flip(i), j] = COLORS[current]
                if current in IMGS.keys():
                    img_name = os.path.join(self.this_file_path, "img",
                                            IMGS[current] + ".png")
                    img = Image.open(img_name)
                    if img.mode != "RGBA":
                        img = img.convert('RGBA')
                    extent = [j - 0.35, j + 0.35, flip(i) - 0.35, flip(i) + 0.35]
                    imgs += [(extent, img)]
        plt.imshow(background, interpolation="nearest")
        ax.set_xticks(np.arange(self.grid_map.shape[1]) + 0.5, minor=True)
        ax.set_yticks(np.arange(self.grid_map.shape[0]) + 0.5, minor=True)
        plt.grid(True, which="minor")
        for extent, img in imgs:
            plt.imshow(img, extent=extent, cmap='gray')
        plt.autoscale()
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        if mode == "human":
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(data)
            return self.viewer.isopen
        elif mode == "rgb_array":
            return data

    def vis_a(self, action):
        self.action_to_vis = action

    @staticmethod
    def _read_grid_map(grid_map_path):
        with open(grid_map_path, 'r') as f:
            grid_map = f.readlines()
        grid_map_array = np.array(
            list(map(
                lambda x: list(map(
                    lambda y: int(y),
                    x.split(' ')
                )),
                grid_map
            ))
        )
        return grid_map_array

    def _get_start_state(self):
        states = np.where(self.grid_map == START)
        choice = np.random.choice(states[0].shape[0])

        passenger_states = np.zeros(self.n_passengers)
        xy_states = np.array([states[0][choice], states[1][choice]])
        start_state = np.concatenate((xy_states, passenger_states), axis=0).astype(int)

        return start_state

    def _valid(self, state):
        return 0 <= state[0] < self.grid_map.shape[0] and \
               0 <= state[1] < self.grid_map.shape[1]

    def _passenger_id(self, pos):
        for i, pas_pos in enumerate(self.passenger_list):
            if np.all(pas_pos == pos):
                return i + 2
        raise ValueError("Given position is not a passenger position")



class GridWorldHoles(AbstractTaxi):
    def __init__(self, max_episode_steps=100):
        super(GridWorldHoles, self).__init__("gridworld_holes.txt")
        self._max_episode_steps = max_episode_steps
