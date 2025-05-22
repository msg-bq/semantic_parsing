#### "grammar.py"
#### Grace Hadiyanto
#### CS439 FA14

class Grammar:
    """A class that represents the grammar of a context free grammar. It holds
    dictionary that maps each variable name to a corresponding variable object. 
    The _start_variable instance variable holds a pointer to the 
    grammar's start variable object."""

    __slots__ = {'_variable_dict', '_start_variable'}

    def __init__(self, variable_list):
        self._variable_dict = {variable.name: variable for variable in variable_list}
        self._start_variable = self._find_start()

    def _find_start(self):
        """Finds the designated start variable and saves a pointer to it."""
        return self.variable_dict['$START']

    @property
    def variable_dict(self):
        """I'm the 'variable_dict' property"""
        return self._variable_dict
    @variable_dict.setter
    def variable_dict(self, variable_dict):
        self._variable_dict = variable_dict

    @property
    def start_variable(self):
        """I'm the 'start_variable' property"""
        return self._start_variable
    @start_variable.setter
    def start_variable(self, start_variable):
        self._start_variable = start_variable
