import dolfin as df

from designs.design_parser import Side, Region


class SidesDomain(df.SubDomain):
    def __init__(self, domain_size: tuple[float, float], sides: list[str]):
        super().__init__()
        self.domain_size = domain_size
        self.sides = sides

    def inside(self, pos, on_boundary):
        if not on_boundary:
            return False

        for side in self.sides:
            if side == Side.LEFT:
                return df.near(pos[0], 0.0)
            elif side == Side.RIGHT:
                return df.near(pos[0], self.domain_size[0])
            elif side == Side.TOP:
                return df.near(pos[1], self.domain_size[1])
            elif side == Side.BOTTOM:
                return df.near(pos[1], 0.0)
            else:
                raise ValueError(f"Malformed side: {side}")

        return False


class RegionDomain(df.SubDomain):
    def __init__(self, region: Region):
        super().__init__()
        cx, cy = region.center
        w, h = region.size
        self.x_region = (cx - w / 2, cx + w / 2)
        self.y_region = (cy - h / 2, cy + h / 2)

    def inside(self, pos, _):
        return df.between(pos[0], self.x_region) and df.between(pos[1], self.y_region)


class PointDomain(df.SubDomain):
    def __init__(self, point: tuple[float, float]):
        super().__init__()
        self.point = point

    def inside(self, pos, _):
        return df.near(pos[0], self.point[0]) and df.near(pos[1], self.point[1])
