def elastisity_alpha(rho):
    """Solid isotropic material penalization (SIMP)."""
    alpha_min = 1e-6
    return alpha_min + rho**3 * (1 - alpha_min)


def elastisity_alpha_derivative(rho):
    """SIMP derivative"""
    alpha_min = 1e-6
    return 3 * rho**2 * (1 - alpha_min)


def fluid_alpha(rho):
    """Does this have a name?"""
    q = 0.1
    alpha_min = 2.5 / 100**2
    alpha_max = 2.5 / 0.01**2

    return alpha_max + (alpha_min - alpha_max) * rho * (1 + q) / (rho + q)


def fluid_alpha_derivative(rho):
    """Unnamed derivaive"""
    q = 0.1
    alpha_min = 2.5 / 100**2
    alpha_max = 2.5 / 0.01**2   

    return (alpha_min - alpha_max) * q * (1 + q) / (rho + q) ** 2
