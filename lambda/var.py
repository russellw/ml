# this class is intended for logic variables, not necessarily program variables
class Var:
    def __init__(self, t=None):
        self.t = t

    def __repr__(self):
        if not hasattr(self, "name"):
            return "Var"
        return self.name
