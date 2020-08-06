from abc import ABC
from typing import Optional
from ray.rllib.env import MultiAgentEnv
import numpy as np
from pyroborobo import Pyroborobo, PyWorldObserver
from gym.spaces import Box
import sys


class NegotiateRoborobo(MultiAgentEnv):
    action_space = Box(np.array([-1] * 4, dtype=np.float32), np.array([1] * 4, dtype=np.float32))
    observation_space = Box(np.repeat(0, 8 * 4 + 2), np.repeat(1, 8 * 4 + 2))

    def __init__(self, nbrobs: int):
        super().__init__()
        self.nbrobs = nbrobs
        self.rob = None
        self.wms = None

    def __del__(self):
        if self.rob:
            self.rob.close()

    def reset(self):
        if self.rob is None:
            self.rob = Pyroborobo('config/pynegociate.properties', None, None, None, None,
                                  {'gInitialNumberOfRobots': str(self.nbrobs)})
            self.rob.start()
            wo = self.rob.world_observer
            self.wms = self.rob.world_models
        else:
            print('reset')
            wo = self.rob.world_observer
            wo.reset()
        obs_dict = {'player'+str(i): self.wms[i].get_observations() for i in range(self.nbrobs)}
        return obs_dict

    def step(self, action_dict):
        for i in range(self.nbrobs):
            self.wms[i].set_actions(action_dict['player'+str(i)])
        stop = self.rob.update(1)
        if stop:
            sys.exit(0)
        obs_dict = {}
        rewards_dict = {}
        done_dict = {}
        for i in range(self.nbrobs):
            obs = self.wms[i].get_observations()
            if obs['seeking']:
                obs_dict['player'+str(i)] = obs['obs']
                rewards_dict['player'+str(i)] = self.wms[i].get_reward()
                done_dict['player'+str(i)] = self.wms[i].get_done()
        if done_dict:
            done_dict['__all__'] = all(done_dict.values())
        else:
            done_dict['__all__'] = False
        return obs_dict, rewards_dict, done_dict, {}

    def render(self):
        pass

