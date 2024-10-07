
import numpy as np


class ComplexArray:
    def __init__(self, real, imag):
        self._data = np.array(real) + 1j * np.array(imag)

    @property
    def real(self):
        return self._data.real

    @property
    def imag(self):
        return self._data.imag

    def __repr__(self):
        return f"ComplexArray({self._data})"

    def __add__(self, other):
        return ComplexArray(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return ComplexArray(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        return ComplexArray(self.real * other.real - self.imag * other.imag,
                            self.imag * other.real + self.real * other.imag)

    def __truediv__(self, other):
        denom = other.real**2 + other.imag**2
        return ComplexArray((self.real * other.real + self.imag * other.imag) / denom,
                            (self.imag * other.real - self.real * other.imag) / denom)

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        if isinstance(value, ComplexArray):
            self._data[index] = value._data
        else:
            raise ValueError("Value must be an instance of ComplexArray.")

    def __array__(self):
        return self._data  # NumPyに渡すときに複素数型として返す

    def shape(self):
        return self._data.shape

# 使用例
real_part = np.random.rand(3, 4)  # 3行4列の実部
imag_part = np.random.rand(3, 4)  # 3行4列の虚部
complex_array = ComplexArray(real_part, imag_part)

# NumPyの関数に直接渡す
sum_array = np.sum(complex_array)  # 合計を計算
print("Sum of Complex Array:", sum_array, sum_array.dtype)

# 例: NumPyの関数にそのまま渡して複素数型として扱う
mean_array = np.mean(complex_array, axis=0)
print("Mean of Complex Array (by column):", mean_array, mean_array.dtype)
