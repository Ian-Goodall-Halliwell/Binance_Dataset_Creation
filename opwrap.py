from typing import List,Union,Type

from expressions import Expression, ExpressionOps
from subutils import get_callable_kwargs
class OpsWrapper:
    """Ops Wrapper"""

    def __init__(self):
        self._ops = {}

    def reset(self):
        self._ops = {}

    def register(self, ops_list: List[Union[Type[ExpressionOps], dict]]):
        """register operator

        Parameters
        ----------
        ops_list : List[Union[Type[ExpressionOps], dict]]
            - if type(ops_list) is List[Type[ExpressionOps]], each element of ops_list represents the operator class, which should be the subclass of `ExpressionOps`.
            - if type(ops_list) is List[dict], each element of ops_list represents the config of operator, which has the following format:
                {
                    "class": class_name,
                    "module_path": path,
                }
                Note: `class` should be the class name of operator, `module_path` should be a python module or path of file.
        """
        for _operator in ops_list:
            if isinstance(_operator, dict):
                _ops_class, _ = get_callable_kwargs(_operator)
            else:
                _ops_class = _operator

            if not issubclass(_ops_class, Expression):
                raise TypeError("operator must be subclass of ExpressionOps, not {}".format(_ops_class))
                
            self._ops[_ops_class.__name__] = _ops_class

    def __getattr__(self, key):
        if key not in self._ops:
            raise AttributeError("The operator [{0}] is not registered".format(key))
        return self._ops[key]


Operators = OpsWrapper()

def register_all_ops():
    from ops import OpsList
    """register all operator"""
    Operators.reset()
    Operators.register(OpsList)