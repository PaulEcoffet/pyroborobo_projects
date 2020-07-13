from ray.rllib.env import MultiAgentEnv
import numpy as np
from pyroborobo import Pyroborobo, PyController, PyWorldModel, PyWorldObserver
from gym.spaces import Box

class WandererRoborobo(MultiAgentEnv):
    action_space = Box(np.array([-2, -30], dtype=np.float32), np.array([2, 30], dtype=np.float32))
    observation_space = Box(np.repeat(0, 8), np.repeat(1, 8))

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
            self.rob = Pyroborobo('config/wanderer.properties', "dummy", "dummy", "dummy", "dummy", {'gInitialNumberOfRobots': str(self.nbrobs)})
            self.rob.start()
            self.wms = self.rob.getWorldModels()
        else:
            print('reset')
            wo = self.rob.getWorldObserver()
            wo.reset()
            self.wms = self.rob.getWorldModels()
        self.moves = 0
        obs_dict = {'player'+str(i): self.wms[i].getCameraSensorsDist() for i in range(self.nbrobs)}
        return obs_dict

    def step(self, action_dict):
        self.moves += 1
        for i in range(self.nbrobs):
            self.wms[i].speed = action_dict['player'+str(i)][0]
            self.wms[i].rotspeed = action_dict['player'+str(i)][1]
        self.rob.update(1)
        obs_dict = {'player'+str(i): self.wms[i].getCameraSensorsDist() for i in range(self.nbrobs)}
        punish = {i: (1 - obs_dict['player'+str(i)][2]) + (1 - obs_dict['player'+str(i)][3] + (1 - obs_dict['player'+str(i)][4]))
                  for i in range(self.nbrobs)}
        done = {'player'+str(i): self.moves > self.max_moves for i in range(self.nbrobs)}
        done['__all__'] = all(done.values())
        rewards = {'player'+str(i): self.wms[i].speed - punish[i] for i in range(self.nbrobs)}
        return obs_dict, rewards, done, {}

    def render(self):
        pass

