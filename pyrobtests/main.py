from pyroborobo import Pyroborobo, PyController, PyWorldModel
from objects import GateObject, SwitchObject
import numpy as np

WALL_ID = 0

class WanderController(PyController):
    world_model: PyWorldModel

    def __init__(self, wm: PyWorldModel):
        PyController.__init__(self, wm)

    def reset(self):
        pass

    def step(self):
        self.world_model.translation = 2
        self.world_model.rotation = 0
        dists = self.world_model.camera_pixel_distance
        maxdisthalf = self.world_model.maxdistcamera // 2
        ids = self.world_model.camera_objects_ids

        # %go to resource
        is_resource = (2 < ids) & (ids < 100)  # ids is in not gate nor switch and not a robot (id < 100)
        must_avoid = (ids == WALL_ID) | (ids > 1000)  # id is a wall or a robot (id > 1000)
        if np.any(is_resource[self.world_model.fl_sensors]):
            self.world_model.rotation = 10
        if np.any(is_resource[self.world_model.fr_sensors]):
            self.world_model.rotation = -10

        # even if there are resources, avoid walls if too close (half dist)
        if np.any((dists[self.world_model.fl_sensors + self.world_model.f_sensors] < maxdisthalf)
                  & must_avoid[self.world_model.fl_sensors + self.world_model.f_sensors]):
            self.world_model.rotation = -10
        if np.any((dists[self.world_model.fr_sensors] < maxdisthalf)
                  & must_avoid[self.world_model.fr_sensors]):
            self.world_model.rotation = 10




rob = Pyroborobo.create("config/wanderer.properties",
                        None, WanderController, PyWorldModel, None,
                 {"gate": GateObject, "switch": SwitchObject}, {})


if __name__ == "__main__":
    rob.start()
    nbgen = 1
    for i in range(nbgen):
        rob.update(1000)
    print("over")
    Pyroborobo.close()
