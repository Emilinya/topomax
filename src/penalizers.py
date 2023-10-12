class ElasticPenalizer:
    """Solid isotropic material penalization (SIMP)."""

    penalization = 3
    minimum = 1e-6

    @classmethod
    def eval(cls, rho):
        p, m = cls.penalization, cls.minimum

        return m + rho**p * (1 - m)

    @classmethod
    def derivative(cls, rho):
        p, m = cls.penalization, cls.minimum

        return p * rho ** (p - 1) * (1 - m)


class FluidPenalizer:
    """What is this called?"""

    penalization = 0.1
    minimum = 2.5 / 100**2
    maximum = 2.5 / 0.01**2

    @classmethod
    def eval(cls, rho):
        q, mini, maxi = cls.penalization, cls.minimum, cls.maximum

        return maxi + (mini - maxi) * rho * (1 + q) / (rho + q)

    @classmethod
    def derivative(cls, rho):
        q, mini, maxi = cls.penalization, cls.minimum, cls.maximum

        return (mini - maxi) * q * (1 + q) / (rho + q) ** 2
