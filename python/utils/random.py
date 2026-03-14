import random as _random

_rng = _random.Random()


def set_seed(seed):
    _rng.seed(seed)


def random():
    return _rng.random()


def gauss(mu=0.0, sigma=1.0):
    return _rng.gauss(mu, sigma)


def randint(a, b):
    return _rng.randint(a, b)


def shuffle(lst):
    """Fisher-Yates in-place shuffle."""
    n = len(lst)
    for i in range(n - 1, 0, -1):
        j = _rng.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]


def choice(lst):
    return _rng.choice(lst)


def sample(lst, k):
    return _rng.sample(lst, k)


def uniform(lo, hi):
    return _rng.uniform(lo, hi)


def get_rng():
    return _rng
