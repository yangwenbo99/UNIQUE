class Functor():
    def __call__(self, *args):
        return None

    def __add__(self, other):
        return SumFunctor(self, other)

    def __sub__(self, other):
        return DifferenceFunction(self, other)

    def then(self, other):
        return ThenFunctor(self, other)

class ThenFunctor(Functor):
    def __init__(self, a: Functor, b: Functor):
        self._a = a
        self._b = b

    def __call__(self, *args):
        self._a(*args)
        return self._b(*args)

class FunctorWrap(Functor):
    def __init__(self, func):
        self._func = func

    def __call__(self, *args):
        return self._func(*args)

class SumFunctor(Functor):
    def __init__(self, a: Functor, b: Functor):
        self._a = a
        self._b = b

    def __call__(self, *args):
        return self._a(*args) + self._b(*args)

class DifferenceFunction(Functor):
    def __init__(self, a: Functor, b: Functor):
        self._a = a
        self._b = b

    def __call__(self, *args):
        return self._a(*args) - self._b(*args)
