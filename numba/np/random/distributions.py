import numpy as np

from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
                                        ziggurat_nor_r, fi_double,
                                        wi_float, ki_float,
                                        ziggurat_nor_inv_r_f,
                                        ziggurat_nor_r_f, fi_float,
                                        we_double, ke_double,
                                        ziggurat_exp_r, fe_double,
                                        we_float, ke_float,
                                        ziggurat_exp_r_f, fe_float,
                                        M_PI, INT64_MAX,
                                        ziggurat_nor_inv_r)
from numba.np.random.generator_core import (next_double, next_float,
                                            next_uint32, next_uint64)

# All following implementations are direct translations from:
# https://github.com/numpy/numpy/blob/7cfef93c77599bd387ecc6a15d186c5a46024dac/numpy/random/src/distributions/distributions.c


@register_jitable
def random_standard_normal(bitgen):
    while 1:
        r = next_uint64(bitgen)
        idx = r & 0xff
        r >>= 8
        sign = r & 0x1
        rabs = (r >> 1) & 0x000fffffffffffff
        x = rabs * wi_double[idx]
        if (sign & 0x1):
            x = -x
        if rabs < ki_double[idx]:
            return x
        if idx == 0:
            while 1:
                xx = -ziggurat_nor_inv_r * np.log1p(-next_double(bitgen))
                yy = -np.log1p(-next_double(bitgen))
                if (yy + yy > xx * xx):
                    if ((rabs >> 8) & 0x1):
                        return -(ziggurat_nor_r + xx)
                    else:
                        return ziggurat_nor_r + xx
        else:
            if (((fi_double[idx - 1] - fi_double[idx]) *
                    next_double(bitgen) + fi_double[idx]) <
                    np.exp(-0.5 * x * x)):
                return x


@register_jitable
def random_standard_normal_f(bitgen):
    while 1:
        r = next_uint32(bitgen)
        idx = r & 0xff
        sign = (r >> 8) & 0x1
        rabs = (r >> 9) & 0x0007fffff
        x = rabs * wi_float[idx]
        if (sign & 0x1):
            x = -x
        if (rabs < ki_float[idx]):
            return x
        if (idx == 0):
            while 1:
                xx = -ziggurat_nor_inv_r_f * np.log1p(-next_float(bitgen))
                yy = -np.log1p(-next_float(bitgen))
                if (yy + yy > xx * xx):
                    if ((rabs >> 8) & 0x1):
                        return -(ziggurat_nor_r_f + xx)
                    else:
                        return ziggurat_nor_r_f + xx
        else:
            if (((fi_float[idx - 1] - fi_float[idx]) * next_float(bitgen) +
                 fi_float[idx]) < np.exp(-0.5 * x * x)):
                return x


@register_jitable
def random_standard_exponential(bitgen):
    while 1:
        ri = next_uint64(bitgen)
        ri >>= 3
        idx = ri & 0xFF
        ri >>= 8
        x = ri * we_double[idx]
        if (ri < ke_double[idx]):
            return x
        else:
            if idx == 0:
                return ziggurat_exp_r - np.log1p(-next_double(bitgen))
            elif ((fe_double[idx - 1] - fe_double[idx]) * next_double(bitgen) +
                  fe_double[idx] < np.exp(-x)):
                return x


@register_jitable
def random_standard_exponential_f(bitgen):
    while 1:
        ri = next_uint32(bitgen)
        ri >>= 1
        idx = ri & 0xFF
        ri >>= 8
        x = ri * we_float[idx]
        if (ri < ke_float[idx]):
            return x
        else:
            if (idx == 0):
                return ziggurat_exp_r_f - np.log1p(-next_float(bitgen))
            elif ((fe_float[idx - 1] - fe_float[idx]) * next_float(bitgen) +
                  fe_float[idx] < np.exp(-x)):
                return x


@register_jitable
def random_standard_exponential_inv(bitgen):
    return -np.log1p(-next_double(bitgen))


@register_jitable
def random_standard_exponential_inv_f(bitgen):
    return -np.log1p(-next_float(bitgen))


@register_jitable
def random_standard_gamma(bitgen, shape):
    if (shape == 1.0):
        return random_standard_exponential(bitgen)
    elif (shape == 0.0):
        return 0.0
    elif (shape < 1.0):
        while 1:
            U = next_double(bitgen)
            V = random_standard_exponential(bitgen)
            if (U <= 1.0 - shape):
                X = pow(U, 1. / shape)
                if (X <= V):
                    return X
            else:
                Y = -np.log((1 - U) / shape)
                X = pow(1.0 - shape + shape * Y, 1. / shape)
                if (X <= (V + Y)):
                    return X
    else:
        b = shape - 1. / 3.
        c = 1. / np.sqrt(9 * b)
        while 1:
            while 1:
                X = random_standard_normal(bitgen)
                V = 1.0 + c * X
                if (V > 0.0):
                    break

            V = V * V * V
            U = next_double(bitgen)
            if (U < 1.0 - 0.0331 * (X * X) * (X * X)):
                return (b * V)

            if (np.log(U) < 0.5 * X * X + b * (1. - V + np.log(V))):
                return (b * V)


@register_jitable
def random_standard_gamma_f(bitgen, shape):
    if (shape == 1.0):
        return random_standard_exponential_f(bitgen)
    elif (shape == 0.0):
        return 0.0
    elif (shape < 1.0):
        while 1:
            U = next_float(bitgen)
            V = random_standard_exponential_f(bitgen)
            if (U <= 1.0 - shape):
                X = pow(U, 1.0 / shape)
                if (X <= V):
                    return X
            else:
                Y = -np.log((1.0 - U) / shape)
                X = pow(1.0 - shape + shape * Y, 1.0 / shape)
                if (X <= (V + Y)):
                    return X
    else:
        b = shape - 1.0 / 3.0
        c = 1.0 / np.sqrt(9.0 * b)
        while 1:
            while 1:
                X = random_standard_normal_f(bitgen)
                V = 1.0 + c * X
                if (V > 0.0):
                    break

            V = V * V * V
            U = next_float(bitgen)
            if (U < 1.0 - 0.0331 * (X * X) * (X * X)):
                return (b * V)

            if (np.log(U) < 0.5 * X * X + b * (1.0 - V + np.log(V))):
                return (b * V)


@register_jitable
def random_normal(bitgen, loc, scale):
    return loc + scale * random_standard_normal(bitgen)


@register_jitable
def random_normal_f(bitgen, loc, scale):
    return loc + scale * random_standard_normal_f(bitgen)


@register_jitable
def random_exponential(bitgen, scale):
    return scale * random_standard_exponential(bitgen)


@register_jitable
def random_exponential_f(bitgen, scale):
    return scale * random_standard_exponential_f(bitgen)


@register_jitable
def random_uniform(bitgen, lower, range):
    return lower + range * next_double(bitgen)


@register_jitable
def random_uniform_f(bitgen, lower, range):
    return lower + range * next_float(bitgen)


@register_jitable
def random_gamma(bitgen, shape, scale):
    return scale * random_standard_gamma(bitgen, shape)


@register_jitable
def random_gamma_f(bitgen, shape, scale):
    return scale * random_standard_gamma_f(bitgen, shape)


@register_jitable
def random_beta(bitgen, a, b):
    if a <= 1.0 and b <= 1.0:
        while 1:
            U = next_double(bitgen)
            V = next_double(bitgen)
            X = pow(U, 1.0 / a)
            Y = pow(V, 1.0 / b)
            XpY = X + Y
            if XpY <= 1.0 and XpY > 0.0:
                if (X + Y > 0):
                    return X / XpY
                else:
                    logX = np.log(U) / a
                    logY = np.log(V) / b
                    logM = min(logX, logY)
                    logX -= logM
                    logY -= logM

                    return np.exp(logX - np.log(np.exp(logX) + np.exp(logY)))
    else:
        Ga = random_standard_gamma(bitgen, a)
        Gb = random_standard_gamma(bitgen, b)
        return Ga / (Ga + Gb)


@register_jitable
def random_chisquare(bitgen, df):
    return 2.0 * random_standard_gamma(bitgen, df / 2.0)


@register_jitable
def random_f(bitgen, dfnum, dfden):
    return ((random_chisquare(bitgen, dfnum) * dfden) /
            (random_chisquare(bitgen, dfden) * dfnum))


@register_jitable
def random_standard_cauchy(bitgen):
    return random_standard_normal(bitgen) / random_standard_normal(bitgen)


@register_jitable
def random_pareto(bitgen, a):
    return np.expm1(random_standard_exponential(bitgen) / a)


@register_jitable
def random_weibull(bitgen, a):
    if (a == 0.0):
        return 0.0
    return pow(random_standard_exponential(bitgen), 1. / a)


@register_jitable
def random_power(bitgen, a):
    return pow(-np.expm1(-random_standard_exponential(bitgen)), 1. / a)


@register_jitable
def random_laplace(bitgen, loc, scale):
    U = next_double(bitgen)
    while U <= 0:
        U = next_double(bitgen)
    if (U >= 0.5):
        U = loc - scale * np.log(2.0 - U - U)
    elif (U > 0.0):
        U = loc + scale * np.log(U + U)
    return U


@register_jitable
def random_gumbel(bitgen, loc, scale):
    U = 1.0 - next_double(bitgen)
    while U >= 1.0:
        U = 1.0 - next_double(bitgen)
    return loc - scale * np.log(-np.log(U))


@register_jitable
def random_logistic(bitgen, loc, scale):
    U = next_double(bitgen)
    while U <= 0.0:
        U = next_double(bitgen)
    return loc + scale * np.log(U / (1.0 - U))


@register_jitable
def random_lognormal(bitgen, mean, sigma):
    return np.exp(random_normal(bitgen, mean, sigma))


@register_jitable
def random_rayleigh(bitgen, mode):
    return mode * np.sqrt(2.0 * random_standard_exponential_inv(bitgen))


@register_jitable
def random_standard_t(bitgen, df):
    num = random_standard_normal(bitgen)
    denom = random_standard_gamma(bitgen, df / 2)
    return np.sqrt(df / 2) * num / np.sqrt(denom)


@register_jitable
def random_wald(bitgen, mean, scale):
    mu_2l = mean / (2 * scale)
    Y = random_standard_normal(bitgen)
    Y = mean * Y * Y
    X = mean + mu_2l * (Y - np.sqrt(4 * scale * Y + Y * Y))
    U = next_double(bitgen)
    if (U <= mean / (mean + X)):
        return X
    else:
        return mean * mean / X


@register_jitable
def random_vonmises(bitgen, mu, kappa):
    if (kappa < 1e-8):
        return M_PI * (2 * next_double(bitgen) - 1)
    else:
        if (kappa < 1e-5):
            s = (1. / kappa + kappa)
        else:
            if (kappa <= 1e6):
                r = 1 + np.sqrt(1 + 4 * kappa * kappa)
                rho = (r - np.sqrt(2 * r)) / (2 * kappa)
                s = (1 + rho * rho) / (2 * rho)
            else:
                result = mu + np.sqrt(1. / kappa) * \
                    random_standard_normal(bitgen)
                if (result < -M_PI):
                    result += 2 * M_PI
                if (result > M_PI):
                    result -= 2 * M_PI
                return result

        while 1:
            U = next_double(bitgen)
            Z = np.cos(M_PI * U)
            W = (1 + s * Z) / (s + Z)
            Y = kappa * (s - W)
            V = next_double(bitgen)
            if ((Y * (2 - Y) - V >= 0) or (np.log(Y / V) + 1 - Y >= 0)):
                break

        U = next_double(bitgen)

        result = np.arccos(W)
        if (U < 0.5):
            result = -result
        result += mu
        neg = (result < 0)
        mod = np.fabs(result)
        mod = (np.fmod(mod + M_PI, 2 * M_PI) - M_PI)
        if (neg):
            mod *= -1

        return mod


@register_jitable
def random_geometric_search(bitgen, p):
    X = 1
    sum = prod = p
    q = 1.0 - p
    U = next_double(bitgen)
    while (U > sum):
        prod *= q
        sum += prod
        X = X + 1
    return X


@register_jitable
def random_geometric_inversion(bitgen, p):
    return np.ceil(-random_standard_exponential(bitgen) / np.log1p(-p))


@register_jitable
def random_geometric(bitgen, p):
    if (p >= 0.333333333333333333333333):
        return random_geometric_search(bitgen, p)
    else:
        return random_geometric_inversion(bitgen, p)


@register_jitable
def random_zipf(bitgen, a):
    am1 = a - 1.0
    b = pow(2.0, am1)
    while 1:
        U = 1.0 - next_double(bitgen)
        V = next_double(bitgen)
        X = np.floor(pow(U, -1.0 / am1))
        if (X > INT64_MAX or X < 1.0):
            continue

        T = pow(1.0 + 1.0 / X, am1)
        if (V * X * (T - 1.0) / (b - 1.0) <= T / b):
            return X


@register_jitable
def random_triangular(bitgen, left, mode,
                      right):
    base = right - left
    leftbase = mode - left
    ratio = leftbase / base
    leftprod = leftbase * base
    rightprod = (right - mode) * base

    U = next_double(bitgen)
    if (U <= ratio):
        return left + np.sqrt(U * leftprod)
    else:
        return right - np.sqrt((1.0 - U) * rightprod)


@register_jitable
def random_loggam(x):
    a = [8.333333333333333e-02, -2.777777777777778e-03,
         7.936507936507937e-04, -5.952380952380952e-04,
         8.417508417508418e-04, -1.917526917526918e-03,
         6.410256410256410e-03, -2.955065359477124e-02,
         1.796443723688307e-01, -1.39243221690590e+00]

    if ((x == 1.0) or (x == 2.0)):
        return 0.0
    elif (x < 7.0):
        n = int(7 - x)
    else:
        n = 0

    x0 = x + n
    x2 = (1.0 / x0) * (1.0 / x0)
    # /* log(2 * M_PI) */
    lg2pi = 1.8378770664093453e+00
    gl0 = a[9]

    for k in range(0, 9):
        gl0 *= x2
        gl0 += a[8 - k]

    gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * np.log(x0) - x0
    if (x < 7.0):
        for k in range(1, n + 1):
            gl = gl - np.log(x0 - 1.0)
            x0 = x0 - 1.0

    return gl


@register_jitable
def random_poisson_mult(bitgen, lam):
    enlam = np.exp(-lam)
    X = 0
    prod = 1.0
    while (1):
        U = next_double(bitgen)
        prod *= U
        if (prod > enlam):
            X += 1
        else:
            return X


@register_jitable
def random_poisson_ptrs(bitgen, lam):

    slam = np.sqrt(lam)
    loglam = np.log(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2)

    while (1):
        U = next_double(bitgen) - 0.5
        V = next_double(bitgen)
        us = 0.5 - np.fabs(U)
        k = int((2 * a / us + b) * U + lam + 0.43)
        if ((us >= 0.07) and (V <= vr)):
            return k

        if ((k < 0) or ((us < 0.013) and (V > us))):
            continue

        # /* log(V) == log(0.0) ok here */
        # /* if U==0.0 so that us==0.0, log is ok since always returns */
        if ((np.log(V) + np.log(invalpha) - np.log(a / (us * us) + b)) <=
           (-lam + k * loglam - random_loggam(k + 1))):
            return k


@register_jitable
def random_poisson(bitgen, lam):
    if (lam >= 10):
        return random_poisson_ptrs(bitgen, lam)
    elif (lam == 0):
        return 0
    else:
        return random_poisson_mult(bitgen, lam)


@register_jitable
def random_negative_binomial(bitgen, n, p):
    Y = random_gamma(bitgen, n, (1 - p) / p)
    return random_poisson(bitgen, Y)
