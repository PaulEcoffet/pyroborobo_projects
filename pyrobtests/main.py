from pyroborobo import Pyroborobo, PyController, PyWorldModel
from objects import GateObject, SwitchObject
import numpy as np

class WanderController(PyController):
    world_model: PyWorldModel

    def __init__(self, wm: PyWorldModel):
        PyController.__init__(self, wm)

    def reset(self):
        pass

    def step(self):
        self.world_model.translation = 1
        self.world_model.rotation = 0
        dists = self.world_model.get_camera_sensors_dist()
        ids = self.world_model.get_camera_object_ids()
        angles = self.world_model.get_camera_angles()
        lf_sensors = [i for i, a in enumerate(angles) if 0 <= a < np.pi]
        rf_sensors = [i for i, a in enumerate(angles) if -np.pi < a < 0]

        # avoid walls
        avoid_left_score = 0


rob = Pyroborobo.create("config/wanderer.properties", None, WanderController, None, None,
                 {"gate": GateObject, "switch": SwitchObject}, {})


if __name__ == "__main__":
    rob.start()
    rob.update(1000)
    Pyroborobo.close()
