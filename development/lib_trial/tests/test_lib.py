
from lib_trial.module_a import ClassA
from lib_trial import ClassB


def test_a():
    aa = ClassA()
    bb = ClassB()
    assert isinstance(aa, ClassA)
    assert isinstance(bb, ClassB)
