
import numpy as np
from numba.core import types
from numba.core.extending import overload_method
from numba.core.imputils import Registry
from numba.np.numpy_support import as_dtype, from_dtype
from numba.np.random.generator_core import next_float, next_double
from numba.np.random.distributions import \
    (random_standard_exponential_inv_f, random_standard_exponential_inv,
     random_standard_exponential, random_standard_normal_f,
     random_standard_gamma, random_standard_normal, random_power,
     random_standard_exponential_f, random_standard_gamma_f, random_normal,
     random_exponential, random_gamma, random_beta,
     random_f,random_chisquare,random_standard_cauchy,random_pareto,
     random_weibull, random_laplace, random_gumbel, random_logistic,
     random_lognormal, random_rayleigh, random_standard_t, random_wald,
     random_vonmises, random_geometric, random_zipf, random_triangular,
     random_poisson, random_negative_binomial)
from numba.np.random.random_methods import \
    (random_bounded_uint64_fill, random_bounded_uint32_fill,
     random_bounded_uint16_fill, random_bounded_uint8_fill,
     random_bounded_bool_fill, _randint_arg_check)


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


# Overload the Generator().integers()
@overload_method(types.NumPyRandomGeneratorType, 'integers')
def NumPyRandomGeneratorType_integers(inst, low, high=None, size=None,
                                      dtype=np.int64, endpoint=False):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(dtype, types.Omitted):
        dtype = dtype.value

    if not isinstance(dtype, types.Type):
        dt = np.dtype(dtype)
        nb_dt = from_dtype(dt)
        _dtype = dtype
    else:
        nb_dt = dtype
        _dtype = as_dtype(nb_dt)

    if _dtype == np.int32:
        int_func = random_bounded_uint32_fill
        lower_bound = -0x80000000
        upper_bound = 0x7FFFFFFF
    elif _dtype == np.int64:
        int_func = random_bounded_uint64_fill
        lower_bound = -0x8000000000000000
        upper_bound = 0x7FFFFFFFFFFFFFFF
    elif _dtype == np.int16:
        int_func = random_bounded_uint16_fill
        lower_bound = -0x8000
        upper_bound = 0xFFFF
    elif _dtype == np.int8:
        int_func = random_bounded_uint8_fill
        lower_bound = -0x80
        upper_bound = 0xFF
    elif _dtype == np.uint32:
        int_func = random_bounded_uint32_fill
        lower_bound = -0x80000000
        upper_bound = 0x7FFFFFFF
    elif _dtype == np.uint64:
        int_func = random_bounded_uint64_fill
        lower_bound = -0x8000000000000000
        upper_bound = 0x7FFFFFFFFFFFFFFF
    elif _dtype == np.uint16:
        int_func = random_bounded_uint16_fill
        lower_bound = -0x8000
        upper_bound = 0xFFFF
    elif _dtype == np.uint8:
        int_func = random_bounded_uint8_fill
        lower_bound = -0x80
        upper_bound = 0xFF
    elif _dtype == np.bool_:
        int_func = random_bounded_bool_fill
        lower_bound = -1
        upper_bound = 2
    else:
        raise TypeError('Unsupported dtype %r for integers' % _dtype)

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, low, high=None, size=None,
                 dtype=np.int64, endpoint=False):
            low, rng = _randint_arg_check(low, high, endpoint,
                                          lower_bound, upper_bound)
            mask = None
            return int_func(inst.bit_generator, low, rng, mask, 1, dtype)[0]
        return impl
    else:
        def impl(inst, low, high=None, size=None,
                 dtype=np.int64, endpoint=False):
            low, rng = _randint_arg_check(low, high, endpoint,
                                          lower_bound, upper_bound)
            mask = None
            return int_func(inst.bit_generator, low, rng, mask, size, dtype)
        return impl


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


# Overload the Generator().beta() method
@overload_method(types.NumPyRandomGeneratorType, 'beta')
def NumPyRandomGeneratorType_beta(inst, a, b, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a, b, size=None):
            return random_beta(inst.bit_generator, a, b)
        return impl
    else:
        def impl(inst, a, b, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_beta(inst.bit_generator, a, b)
            return out
        return impl


# Overload the Generator().chisquare() method
@overload_method(types.NumPyRandomGeneratorType, 'f')
def NumPyRandomGeneratorType_f(inst, dfnum, dfden, size=None):
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, dfnum, dfden, size=None):
            return random_f(inst.bit_generator, dfnum, dfden)
        return impl
    else:
        def impl(inst, dfnum, dfden, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_f(inst.bit_generator, dfnum, dfden)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'chisquare')
def NumPyRandomGeneratorType_chisquare(inst, df, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, df, size=None):
            return random_chisquare(inst.bit_generator, df)
        return impl
    else:
        def impl(inst, df, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_chisquare(inst.bit_generator, df)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'standard_cauchy')
def NumPyRandomGeneratorType_standard_cauchy(inst, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, size=None):
            return random_standard_cauchy(inst.bit_generator)
        return impl
    else:
        def impl(inst, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_standard_cauchy(inst.bit_generator)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'pareto')
def NumPyRandomGeneratorType_pareto(inst, a, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a, size=None):
            return random_pareto(inst.bit_generator, a)
        return impl
    else:
        def impl(inst, a, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_pareto(inst.bit_generator, a)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'weibull')
def NumPyRandomGeneratorType_weibull(inst, a, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a, size=None):
            return random_weibull(inst.bit_generator, a)
        return impl
    else:
        def impl(inst, a, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_weibull(inst.bit_generator, a)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'power')
def NumPyRandomGeneratorType_power(inst, a, size=None):
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a, size=None):
            return random_power(inst.bit_generator, a)
        return impl
    else:
        def impl(inst, a, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_power(inst.bit_generator, a)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'laplace')
def NumPyRandomGeneratorType_laplace(inst, loc=0.0, scale=1.0, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, loc=0.0, scale=1.0, size=None):
            return random_laplace(inst.bit_generator, loc, scale)
        return impl
    else:
        def impl(inst, loc=0.0, scale=1.0, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_laplace(inst.bit_generator, loc, scale)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'gumbel')
def NumPyRandomGeneratorType_gumbel(inst, loc=0.0, scale=1.0, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, loc=0.0, scale=1.0, size=None):
            return random_gumbel(inst.bit_generator, loc, scale)
        return impl
    else:
        def impl(inst, loc=0.0, scale=1.0, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_gumbel(inst.bit_generator, loc, scale)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'logistic')
def NumPyRandomGeneratorType_logistic(inst, loc=0.0, scale=1.0, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, loc=0.0, scale=1.0, size=None):
            return random_logistic(inst.bit_generator, loc, scale)
        return impl
    else:
        def impl(inst, loc=0.0, scale=1.0, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_logistic(inst.bit_generator, loc, scale)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'lognormal')
def NumPyRandomGeneratorType_lognormal(inst, mean=0.0, sigma=1.0, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, mean=0.0, sigma=1.0, size=None):
            return random_lognormal(inst.bit_generator, mean, sigma)
        return impl
    else:
        def impl(inst, mean=0.0, sigma=1.0, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_lognormal(inst.bit_generator, mean, sigma)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'rayleigh')
def NumPyRandomGeneratorType_rayleigh(inst, scale=1.0, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, scale=1.0, size=None):
            return random_rayleigh(inst.bit_generator, scale)
        return impl
    else:
        def impl(inst, scale=1.0, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_rayleigh(inst.bit_generator, scale)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'standard_t')
def NumPyRandomGeneratorType_standard_t(inst, df, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, df, size=None):
            return random_standard_t(inst.bit_generator, df)
        return impl
    else:
        def impl(inst, df, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_standard_t(inst.bit_generator, df)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'wald')
def NumPyRandomGeneratorType_wald(inst, mean, scale, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, mean, scale, size=None):
            return random_wald(inst.bit_generator, mean, scale)
        return impl
    else:
        def impl(inst, mean, scale, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_wald(inst.bit_generator, mean, scale)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'vonmises')
def NumPyRandomGeneratorType_vonmises(inst, mu, kappa, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, mu, kappa, size=None):
            return random_vonmises(inst.bit_generator, mu, kappa)
        return impl
    else:
        def impl(inst, mu, kappa, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_vonmises(inst.bit_generator, mu, kappa)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'geometric')
def NumPyRandomGeneratorType_geometric(inst, p, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, p, size=None):
            return random_geometric(inst.bit_generator, p)
        return impl
    else:
        def impl(inst, p, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_geometric(inst.bit_generator, p)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'zipf')
def NumPyRandomGeneratorType_zipf(inst, a, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a, size=None):
            return random_zipf(inst.bit_generator, a)
        return impl
    else:
        def impl(inst, a, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_zipf(inst.bit_generator, a)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'triangular')
def NumPyRandomGeneratorType_triangular(inst, left, mode, right, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, left, mode, right, size=None):
            return random_triangular(inst.bit_generator, left, mode, right)
        return impl
    else:
        def impl(inst, left, mode, right, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_triangular(inst.bit_generator,
                                           left, mode, right)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'poisson')
def NumPyRandomGeneratorType_poisson(inst, lam , size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, lam , size=None):
            return random_poisson(inst.bit_generator, lam)
        return impl
    else:
        def impl(inst, lam , size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_poisson(inst.bit_generator, lam)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'negative_binomial')
def NumPyRandomGeneratorType_negative_binomial(inst, n, p, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst,  n, p , size=None):
            return random_negative_binomial(inst.bit_generator, n, p)
        return impl
    else:
        def impl(inst, n, p , size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_negative_binomial(inst.bit_generator, n, p)
            return out
        return impl
