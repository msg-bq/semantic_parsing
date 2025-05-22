#### "variable_reference.py"
#### Grace Hadiyanto
#### CS439 FA14

class VariableReference:
    """This class represents the occurence of a variable in the current
    string. It holds a pointer to the current string, a start index, end index,
    and name of the variable."""
    
    __slots__ = {'_current_string', '_start_index', '_end_index', '_name'}

    def __init__(self, current_string, start_index, end_index):
        self._current_string = current_string
        self._start_index = start_index
        self._end_index = end_index
        self._name = self._current_string[start_index:end_index + 1]

    def __str__(self):
        return 'curr_str: {}\nstart: {}\nend: {}\nname: {}'.format(self._current_string,
                                                                   self._start_index,
                                                                   self._end_index,
                                                                   self._name)

    @property
    def current_string(self):
        """I'm the 'current_string' property"""
        return self._current_string
    @current_string.setter
    def current_string(self, a_string):
        self._current_string = a_string

    @property
    def start_index(self):
        """I'm the 'start_index' property"""
        return self._start_index
    @start_index.setter
    def start_index(self, index):
        self._start_index = index

    @property
    def end_index(self):
        """I'm the 'end_index' property"""
        return self._end_index
    @end_index.setter
    def end_index(self, index):
        self._end_index = index

    @property
    def name(self):
        """I'm the 'name' property"""
        return self._name
