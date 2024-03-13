from __future__ import annotations

import dolfin as df

from designs.definitions import Side


class SidesDomain(df.SubDomain):
    def __init__(
        self,
        domain_size: tuple[float, float],
        sides: list[Side],
    ):
        super().__init__()
        self.w, self.h = domain_size
        self.sides = sides

    def inside(self, pos, on_boundary):
        if not on_boundary:
            return False

        for side in self.sides:
            if side == Side.LEFT:
                if df.near(pos[0], 0.0):
                    return True
            elif side == Side.RIGHT:
                if df.near(pos[0], self.w):
                    return True
            elif side == Side.TOP:
                if df.near(pos[1], self.h):
                    return True
            elif side == Side.BOTTOM:
                if df.near(pos[1], 0.0):
                    return True
            else:
                raise ValueError(f"Malformed side: {side}")

        return False
