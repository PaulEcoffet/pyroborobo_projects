from pyroborobo import SquareObject, CircleObject, Pyroborobo


class GateObject(SquareObject):
    def __init__(self, id_, data):
        SquareObject.__init__(self, id_)
        self.rob = Pyroborobo.get()
        self.triggered = False
        self.regrow_time_max = data['regrowTimeMax']
        self.regrow_time = 0

    def step(self):
        if self.triggered:
            self.regrow_time -= 1
            if self.regrow_time <= 0:
                self.show()
                self.register()
                self.triggered = False

    def inspect(self, prefix=""):
        return "Hey ben dis-donc tiens"

    def trigger(self, id_):
        if not self.triggered:
            self.triggered = True
            self.regrow_time = self.regrow_time_max
            print(f"I'm triggered by {id_}")
            self.unregister()
            self.hide()


class SwitchObject(CircleObject):
    def __init__(self, id_: int, data: dict):
        CircleObject.__init__(self, id_)
        self.rob = Pyroborobo.get()
        print(data)
        self.message = data.get("sendMessageTo", 0)
        self.triggered = False
        self.regrow_time = 0
        self.regrow_time_max = data['regrowTimeMax']

    def step(self):
        if self.triggered:
            self.regrow_time -= 1
            if self.regrow_time <= 0:
                self.relocate()
                self.register()
                self.show()
                self.triggered = False

    def inspect(self, prefix=""):
        print("hey ben dis donc tiens")

    def is_walked(self, id_):
        if not self.triggered:
            self.rob.objects[self.message].trigger(id_)
            self.unregister()
            self.hide()
            self.triggered = True
            self.regrow_time = self.regrow_time_max

