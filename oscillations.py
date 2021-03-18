"""Compute neutrino oscillations.

Inspired by github.com/discully/oscillations.

Example session:

    >>> U = get_PMNS(theta12, theta13, theta23, delta)  # in radians
    >>> dm2 = get_dm2(7.53e-5, m2_32=2.453e-3)  # in eV^2
    >>> p_e_to_mu = get_prob(nu.e, nu.mu, 500, 2, U, dm2)  # 500m/2MeV
    0.01958854098883325

"""

from dataclasses import dataclass

import numpy as np

@dataclass
class _Nu:
    e: int = 0
    mu: int = 1
    tau: int = 2
    ebar: int = 3
    mubar: int = 4
    taubar: int = 5

    @staticmethod
    def bar(flavor):
        if flavor < 3:
            return flavor + 3
        else:
            return flavor - 3

    @staticmethod
    def is_bar(flavor):
        return flavor >= 3

nu = _Nu()

def get_PMNS(theta12, theta13, theta23, delta):
    """Compute the PMNS matrix for the given parameters."""
    c = np.cos
    s = np.sin
    e = np.exp
    atmo_term = np.array([
        [1, 0, 0],
        [0, c(theta23), s(theta23)],
        [0, -s(theta23), c(theta23)],
    ])
    reactor_term = np.array([
        [c(theta13), 0, s(theta13) * e(-1j * delta)],
        [0, 1, 0],
        [-s(theta13) * e(1j * delta), 0, c(theta13)],
    ])
    solar_term = np.array([
        [c(theta12), s(theta12), 0],
        [-s(theta12), c(theta12), 0],
        [0, 0, 1],
    ])
    # @ symbol is Python-speak for matrix multiplication!
    return atmo_term @ reactor_term @ solar_term

def get_PMNS_sin2(sin2_12, sin2_13, sin2_23, delta):
    """Compute the PMNS matrix when given sin^2(theta) values, plus delta."""
    theta12 = np.arcsin(np.sqrt(sin2_12))
    theta13 = np.arcsin(np.sqrt(sin2_13))
    theta23 = np.arcsin(np.sqrt(sin2_23))
    return get_PMNS(theta12, theta13, theta23, delta)

def get_dm2(m2_21, m2_31=None, m2_32=None):
    """Compute all 3 mass-squared splittings and return a 2D array.

    The row index represents the "reference" mass,
    and the column index represents the mass to subtract from the reference.
    They are zero-indexed.

    Examples:
    - result[2, 1] == m2_32
    - result[1, 2] == -m2_32
    - result[1, 0] = m2_21
    - result[1, 1] == 0

    Either m2_31 or m2_32 can be provided.
    If neither is provided, a ValueError is raised.
    If both are provided, they are inserted into the array unmodified."""
    if m2_31 is None:
        m2_31 = m2_21 + m2_32
    elif m2_32 is None:
        m2_32 = m2_31 - m2_21
    else:
        pass
    return np.array([
        [0, -m2_21, -m2_31],
        [m2_21, 0, -m2_32],
        [m2_31, m2_32, 0],
    ])

def get_prob(from_flavor, to_flavor, L_m, E_MeV, U_PMNS, dm2):
    """Compute the oscillation probability.

    The length parameter can be a numpy array, in which case
    the return value is the same shape.
    """
    if not nu.is_bar(from_flavor) and not nu.is_bar(to_flavor):
        U = U_PMNS
        Ustar = U_PMNS.conjugate()
    elif nu.is_bar(from_flavor) and nu.is_bar(to_flavor):
        U = U_PMNS.conjugate()
        Ustar = U_PMNS
        from_flavor = nu.bar(from_flavor)
        to_flavor = nu.bar(to_flavor)
    else:
        return 0  # no chance of nu-nubar oscillations
    first_dotproduct = Ustar[to_flavor, :] * U[from_flavor, :]
    second_dotproduct = Ustar[from_flavor, :] * U[to_flavor, :]
    # outer product gets me all combinations, so then I can "sum over i,j"
    all_U_products = np.outer(first_dotproduct, second_dotproduct)
    phase_term = np.exp(1j * (2 * 1.267) * dm2[:, :, np.newaxis] * L_m / E_MeV)
    all_terms = all_U_products[:, :, np.newaxis] * phase_term
    result = np.sum(all_terms, axis=(0, 1))
    if any(result.imag > 1e-10):
        raise ValueError(f"Computation failed to return a real result: {result}")
    if hasattr(L_m, '__len__'):
        pass
    else:
        result = result[0]
    return result.real

def prob_to_all(from_flavor, L_m, E_MeV, U_PMNS, dm2):
    """Compute the probability to e, mu and tau."""
    if nu.is_bar(from_flavor):
        to_flavors = np.array([nu.ebar, nu.mubar, nu.taubar])
    else:
        to_flavors = np.array([nu.e, nu.mu, nu.tau])
    if hasattr(L_m, '__len__'):
        num_lengths = len(L_m)
    else:
        num_lengths = 1
    result = np.empty((len(to_flavors), num_lengths), dtype=float)
    for i, to_flavor in enumerate(to_flavors):
        result[i] = get_prob(from_flavor, to_flavor, L_m, E_MeV, U_PMNS, dm2)
    return result
