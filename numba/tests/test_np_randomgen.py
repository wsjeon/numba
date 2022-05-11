import numpy as np
import numba
from numba.tests.support import TestCase

from numpy.random import MT19937, Generator


class TestRandomGenerators(TestCase):
    def check_numpy_parity(self, distribution_func,
                           bitgen_instance=None, seed=1):
        distribution_func = numba.njit(distribution_func)
        if bitgen_instance is None:
            numba_rng_instance = np.random.default_rng(seed=seed)
            numpy_rng_instance = np.random.default_rng(seed=seed)
        else:
            numba_rng_instance = Generator(bitgen_instance(seed))
            numpy_rng_instance = Generator(bitgen_instance(seed))

        # Check parity for different size cases
        for size in [None, (), (100,), (10,20,30)]:
            numba_res = distribution_func(numba_rng_instance, size)
            numpy_res = distribution_func.py_func(numpy_rng_instance,
                                                  size)

            assert np.allclose(numba_res, numpy_res)

        # Check if the end state of both BitGenerators is same
        # after drawing the distributions
        numba_gen_state = numba_rng_instance.__getstate__()['state']
        numpy_gen_state = numpy_rng_instance.__getstate__()['state']

        for _state_key in numpy_gen_state:
            assert np.all(numba_gen_state[_state_key]
                          == numpy_gen_state[_state_key])

    def test_random(self):
        dist_func = lambda x, size:x.random(size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

        dist_func = lambda x, size:x.random(size=size, dtype=np.float32)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_standard_normal(self):
        dist_func = lambda x, size:\
            x.standard_normal(size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

        dist_func = lambda x, size:\
            x.standard_normal(size=size, dtype=np.float32)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_standard_exponential(self):
        dist_func = lambda x, size:\
            x.standard_exponential(size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

        dist_func = lambda x, size:\
            x.standard_exponential(size=size, dtype=np.float32)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

        dist_func = lambda x, size:\
            x.standard_exponential(size=size, method='inv')
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

        dist_func = lambda x, size:\
            x.standard_exponential(size=size, dtype=np.float32, method='inv')
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_standard_gamma(self):
        dist_func = lambda x, size:\
            x.standard_gamma(shape=3.0, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

        dist_func = lambda x, size:\
            x.standard_gamma(shape=3.0, size=size, dtype=np.float32)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_normal(self):
        dist_func = lambda x, size:\
            x.normal(loc=1.5, scale=3, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_exponential(self):
        dist_func = lambda x, size:\
            x.exponential(scale=1.5,size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_gamma(self):
        dist_func = lambda x, size:\
            x.gamma(shape=5.0, scale=1.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)
