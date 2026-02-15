import sys
import superquadric as sq
from superquadric_param import SuperQuadricParams
import visualization as vis
import numpy as np

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]



    # TODO: pipeline    
    test = SuperQuadricParams(1,5,1,0.3,0.5,[2,2,1],[5,5,5])
    mesh = sq.superquadric_mesh(test)
    vis.show_mesh_and_points(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
