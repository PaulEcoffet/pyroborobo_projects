from abc import ABC
from typing import Optional
from ray.rllib.env import MultiAgentEnv
import numpy as np
from pyroborobo import Pyroborobo, PyWorldObserver
from gym.spaces import Box
import sys


class WorldObserver(PyWorldObserver):
    def __init__(self, world):
        PyWorldObserver.__init__(self, world)
        self.rob: Optional[Pyroborobo] = None

    def set_roborobo(self, rob):
        self.rob = rob

    def reset(self):
        for rob in self.rob.robots:
            rob.find_random_location()

    def stepPre(self):
        pass

    def stepPost(self):
        pass


class WandererRoborobo(MultiAgentEnv):
    action_space = Box(np.array([-1, -1], dtype=np.float32), np.array([1, 1], dtype=np.float32))
    observation_space = Box(np.concatenate((np.repeat(0, 8), [-2, -15])), np.concatenate((np.repeat(1, 8), [2, 15])))

    def __init__(self, nbrobs: int, max_moves: int):
        super().__init__()
        self.nbrobs = nbrobs
        self.max_moves = max_moves
        self.moves = 0
        self.rob = None
        self.wms = None

    def __del__(self):
        if self.rob:
            self.rob.close()

    def reset(self):
        if self.rob is None:
            self.rob = Pyroborobo('config/wanderer.properties', WorldObserver, "dummy", "dummy", "dummy",
                                  {'gInitialNumberOfRobots': str(self.nbrobs)})
            self.rob.start()
            wo = self.rob.world_observer
            wo.set_roborobo(self.rob)
            self.wms = self.rob.world_models
        else:
            print('reset')
            wo = self.rob.world_observer
            wo.reset()
        self.moves = 0
        obs_dict = {'player'+str(i): self.get_obs(i) for i in range(self.nbrobs)}
        return obs_dict

    def get_obs(self, i):
        return np.concatenate((self.wms[i].get_camera_sensors_dist(), [self.wms[i].translation, self.wms[i].rotation]))

    def step(self, action_dict):
        self.moves += 1

        for i in range(self.nbrobs):
            self.wms[i].translation = action_dict['player'+str(i)][0] * 2
            self.wms[i].rotation = action_dict['player'+str(i)][1] * 15
        stop = self.rob.update(1)
        if stop:
            sys.exit(0)
        obs_dict = {'player'+str(i): self.get_obs(i) for i in range(self.nbrobs)}
        punish = {i: (1 - obs_dict['player'+str(i)][2]) + (1 - obs_dict['player'+str(i)][3] + (1 - obs_dict['player'+str(i)][4])
                                                           + (np.abs(obs_dict['player'+str(i)][9])/15))
                  for i in range(self.nbrobs)}
        done = {'player'+str(i): self.moves > self.max_moves for i in range(self.nbrobs)}
        done['__all__'] = all(done.values())
        rewards = {'player'+str(i): self.wms[i].translation - 10*punish[i] for i in range(self.nbrobs)}
        return obs_dict, rewards, done, {}

    def render(self):
        pass

