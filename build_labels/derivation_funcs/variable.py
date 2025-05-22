#### "variable.py"
#### Grace Hadiyanto
#### CS439 FA14

class Variable:
    """A class that represents a variable inside a context free grammar.
    The variable has a name and a list of rules that contain possible 
    productions."""
    __slots__ = {'_name', '_rules'}
    def __init__(self, name):
        self._name = name
        self._rules = []

    def __str__(self):
        return 'name: {}\nrules: {}'.format(self._name, self._rules)

    @property
    def name(self):
        """I'm the 'name' property"""
        return self._name
    @name.setter
    def name(self, name):
        self._name = name

    @property
    def rules(self):
        """I'm the 'rules' property"""
        return self._rules
    @rules.setter
    def rules(self, rules):
        self._rules = rules
