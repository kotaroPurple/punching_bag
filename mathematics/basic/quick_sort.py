
import random


def quicksort(values: list[float]|list[int]) -> list[float]|list[int]:
    if len(values) <= 1:
        return values

    pivot = random.choice(values)
    left = [v for v in values if v < pivot]
    middle = [v for v in values if v == pivot]
    right = [v for v in values if v > pivot]
    return quicksort(left) + middle + quicksort(right)


def is_sorted(values: list[float]|list[int]) -> bool:
    return all(values[i] <= values[i+1] for i in range(len(values)-1))


if __name__ == '__main__':
    values = [random.randint(1, 100) for _ in range(20)]
    sorted_values = quicksort(values)
    print(f'{values=}')
    print(f'{sorted_values=}')
    print('sorted:', is_sorted(sorted_values))
