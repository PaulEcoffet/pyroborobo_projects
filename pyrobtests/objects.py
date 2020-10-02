from pyroborobo import SquareObject, CircleObject, Pyroborobo


class GateObject(SquareObject):
    def __init__(self, id_, data):
        SquareObject.__init__(self, id_)
        self.rob = Pyroborobo.get()
        self.triggered = False
        self.regrowtimemax = data['regrowTimeMax']
        self.regrowtime = 0

    def step(self):
        if self.triggered:
            self.regrowtime -= 1
            if self.regrowtime <= 0:
                self.show()
                self.register()
                self.triggered = False

    def inspect(self, prefix=""):
        return "Hey ben dis-donc tiens"

    def trigger(self, id_):
        if not self.triggered:
            self.triggered = True
            self.regrowtime = self.regrowtimemax
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
        self.regrowtime = 0
        self.regrowtimemax = data['regrowTimeMax']

    def step(self):
        if self.triggered:
            self.regrowtime -= 1
            if self.regrowtime <= 0:
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
            self.regrowtime = self.regrowtimemax

