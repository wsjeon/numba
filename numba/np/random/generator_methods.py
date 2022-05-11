import numpy as np
from numba.core import types
from numba.core.extending import overload_method
from numba.core.imputils import Registry
from numba.np.numpy_support import as_dtype, from_dtype
from numba.np.random.generator_core import next_float, next_double
from numba.np.random.distributions import \
    (random_standard_exponential_inv_f, random_standard_exponential_inv,
     random_standard_exponential, random_standard_normal_f,
     random_standard_gamma, random_standard_normal,
     random_standard_exponential_f, random_standard_gamma_f, random_normal,
     random_exponential, random_gamma)


registry = Registry('generator_methods')


def get_proper_func(func_32, func_64, dtype):
    if isinstance(dtype, types.Omitted):
        dtype = dtype.value

    if not isinstance(dtype, types.Type):
        dt = np.dtype(dtype)
        nb_dt = from_dtype(dt)
        np_dt = dtype
    else:
        nb_dt = dtype
        np_dt = as_dtype(nb_dt)

    np_dt = np.dtype(np_dt)

    if np_dt == np.float32:
        next_func = func_32
    else:
        next_func = func_64

    return next_func, nb_dt


# Overload the Generator().random()
@overload_method(types.NumPyRandomGeneratorType, 'random')
def NumPyRandomGeneratorType_random(inst, size=None, dtype=np.float64):
    dist_func, nb_dt = get_proper_func(next_float, next_double, dtype)
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator))
        return impl
    else:
        def impl(inst, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator)
            return out
        return impl


# Overload the Generator().standard_exponential() method
@overload_method(types.NumPyRandomGeneratorType, 'standard_exponential')
def NumPyRandomGeneratorType_standard_exponential(inst, size=None,
                                                  dtype=np.float64,
                                                  method=None):
    if isinstance(method, types.Omitted):
        method = method.value

    # TODO: This way of selecting methods works practically but is
    # extremely hackish. we should try doing the default
    # method==inv comparision over here if possible
    if method:
        dist_func, nb_dt = get_proper_func(
            random_standard_exponential_inv_f,
            random_standard_exponential_inv,
            dtype
        )
    else:
        dist_func, nb_dt = get_proper_func(random_standard_exponential_f,
                                           random_standard_exponential,
                                           dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, size=None, dtype=np.float64, method=None):
            return nb_dt(dist_func(inst.bit_generator))
        return impl
    else:
        def impl(inst, size=None, dtype=np.float64, method=None):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator)
            return out
        return impl


# Overload the Generator().standard_normal() method
@overload_method(types.NumPyRandomGeneratorType, 'standard_normal')
def NumPyRandomGeneratorType_standard_normal(inst, size=None, dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_standard_normal_f,
                                       random_standard_normal,
                                       dtype)
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator))
        return impl
    else:
        def impl(inst, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator)
            return out
        return impl


# Overload the Generator().standard_gamma() method
@overload_method(types.NumPyRandomGeneratorType, 'standard_gamma')
def NumPyRandomGeneratorType_standard_gamma(inst, shape, size=None,
                                            dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_standard_gamma_f,
                                       random_standard_gamma,
                                       dtype)
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, shape, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, shape))
        return impl
    else:
        def impl(inst, shape, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, shape)
            return out
        return impl


# Overload the Generator().normal() method
@overload_method(types.NumPyRandomGeneratorType, 'normal')
def NumPyRandomGeneratorType_normal(inst, loc=0.0, scale=1.0,
                                    size=None):
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, loc=0.0, scale=1.0, size=None):
            return random_normal(inst.bit_generator, loc, scale)
        return impl
    else:
        def impl(inst, loc=0.0, scale=1.0, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_normal(inst.bit_generator, loc, scale)
            return out
        return impl


# Overload the Generator().exponential() method
@overload_method(types.NumPyRandomGeneratorType, 'exponential')
def NumPyRandomGeneratorType_exponential(inst, scale=1.0, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, scale=1.0, size=None):
            return random_exponential(inst.bit_generator, scale)
        return impl
    else:
        def impl(inst, scale=1.0, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_exponential(inst.bit_generator, scale)
            return out
        return impl


# Overload the Generator().gamma() method
@overload_method(types.NumPyRandomGeneratorType, 'gamma')
def NumPyRandomGeneratorType_gamma(inst, shape, scale=1.0, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, shape, scale=1.0, size=None):
            return random_gamma(inst.bit_generator, shape, scale)
        return impl
    else:
        def impl(inst, shape, scale=1.0, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_gamma(inst.bit_generator, shape, scale)
            return out
        return impl
