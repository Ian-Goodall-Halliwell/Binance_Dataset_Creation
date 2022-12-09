

class Expression():
    """
    Expression base class

    Expression is designed to handle the calculation of data with the format below
    data with two dimension for each instrument,

    - feature
    - time:  it  could be observation time or period time.

        - period time is designed for Point-in-time database.  For example, the period time maybe 2014Q4, its value can observed for multiple times(different value may be observed at different time due to amendment).
    """

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        from ops import Gt  # pylint: disable=C0415

        return Gt(self, other)

    def __ge__(self, other):
        from ops import Ge  # pylint: disable=C0415

        return Ge(self, other)

    def __lt__(self, other):
        from ops import Lt  # pylint: disable=C0415

        return Lt(self, other)

    def __le__(self, other):
        from ops import Le  # pylint: disable=C0415

        return Le(self, other)

    def __eq__(self, other):
        from ops import Eq  # pylint: disable=C0415

        return Eq(self, other)

    def __ne__(self, other):
        from ops import Ne  # pylint: disable=C0415

        return Ne(self, other)

    def __add__(self, other):
        from ops import Add  # pylint: disable=C0415

        return Add(self, other)

    def __radd__(self, other):
        from ops import Add  # pylint: disable=C0415

        return Add(other, self)

    def __sub__(self, other):
        from ops import Sub  # pylint: disable=C0415

        return Sub(self, other)

    def __rsub__(self, other):
        from ops import Sub  # pylint: disable=C0415

        return Sub(other, self)

    def __mul__(self, other):
        from ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __rmul__(self, other):
        from ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __div__(self, other):
        from ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rdiv__(self, other):
        from qlib.data.ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __truediv__(self, other):
        from ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rtruediv__(self, other):
        from ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __pow__(self, other):
        from ops import Power  # pylint: disable=C0415

        return Power(self, other)

    def __rpow__(self, other):
        from ops import Power  # pylint: disable=C0415

        return Power(other, self)

    def __and__(self, other):
        from ops import And  # pylint: disable=C0415

        return And(self, other)

    def __rand__(self, other):
        from ops import And  # pylint: disable=C0415

        return And(other, self)

    def __or__(self, other):
        from ops import Or  # pylint: disable=C0415

        return Or(self, other)

    def __ror__(self, other):
        from ops import Or  # pylint: disable=C0415

        return Or(other, self)

    def load(self, df, start_index, end_index, *args):
        

        if start_index is not None and end_index is not None and start_index > end_index:
            raise ValueError("Invalid index range: {} {}".format(start_index, end_index))
        try:
            series = self._load_internal(df, start_index, end_index, *args)
        except Exception as e:
            print(e)
            raise
        series.name = str(self)
        return series
    
class ExpressionOps(Expression):
    """Operator Expression

    This kind of feature will use operator for feature
    construction on the fly.
    """