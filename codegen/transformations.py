from typing import Union, List


class ArgRef:
    """Reference to a function argument"""

    def __init__(self, num):
        """
        :param num: number of argument (0-based)
        """
        self.num = num


class MultipleArgs:
    """"Reference to several function arguments"""

    def __init__(self, start=0, stop=None):
        """
        :param start: index of first argument
        :param stop: index following last argument, None for all up to end
        """
        self.start = start
        self.stop = stop


class TransformToAttribute:
    def __init__(self, attribute: Union[str, List[str]]):
        self.attribute = attribute


class SpliceArgs:
    def __init__(self, args):
        self.args = args
