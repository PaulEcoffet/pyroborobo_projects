from pyroborobo import Pyroborobo, PyController, SquareObject, CircleObject, PyWorldModel


class TestController(PyController):
    world_model: PyWorldModel

    def __init__(self, wm: PyWorldModel):
        PyController.__init__(self, wm)

    def step(self):
        self.world_model.translation = 1
        self.world_model.rotation = 0


class TestSquareObject(SquareObject):

    def __init__(self, id_, data):
        SquareObject.__init__(self, id_)

    def step(self):
        pass

    def inspect(self, prefix=""):
        return "Hey ben dis-donc tiens"


class TestCircleObject(CircleObject):
    def __init__(self, id_: int, data: dict):
        CircleObject.__init__(self, id_)
        print(data)
        data.get("messageTo", 0)


    def step(self):
        pass

    def inspect(self, prefix=""):
        print("hey ben dis donc tiens")

    def is_walked(self, id):
        print("Aïe!")


rob = Pyroborobo("config/wanderer.properties", None, None, None, None,
                 {"pysquare": TestSquareObject, "pycircle": TestCircleObject}, {})


if __name__ == "__main__":
    rob.start()
    rob.update(10000)
