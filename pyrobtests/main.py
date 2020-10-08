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

        is_resource_l = 2 < ids[1] < 100  # ids is in not gate nor switch and not a robot (id < 100)
        is_resource_r = 2 < ids[3] < 100
        must_avoid_l = ids[1] == WALL_ID or ids[1] > 1000  # id is a wall or a robot (id > 1000)
        must_avoid_r = ids[3] == WALL_ID or ids[3] > 1000  # id is a wall or a robot (id > 1000)
        must_avoid_f = ids[2] == WALL_ID or ids[2] > 1000
        if is_resource_l:
            self.world_model.rotation = -10 + np.random.normal(0, 4)
        if is_resource_r:
            self.world_model.rotation = 10 + np.random.normal(0, 4)

        # even if there are resources, avoid walls if too close (half dist)
        if (dists[2] < maxdisthalf and must_avoid_f)\
                or (dists[1] < maxdisthalf and must_avoid_l):
            self.world_model.rotation = 10 + np.random.normal(0, 4)
        if dists[3] < maxdisthalf and must_avoid_r:
            self.world_model.rotation = -10 + np.random.normal(0, 4)


rob = Pyroborobo.create("config/wanderer.properties",
                        None, WanderController, PyWorldModel, None,
                        {"gate": GateObject, "switch": SwitchObject}, {})


if __name__ == "__main__":
    rob.start()
    nbgen = 1
    for i in range(nbgen):
        rob.update(10000)
    print("over")
    Pyroborobo.close()
