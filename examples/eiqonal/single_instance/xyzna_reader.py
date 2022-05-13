import numpy as np

def read_xyzna(filename):
    points = []
    normals = []
    area = []
    with open(filename, 'r') as f:
        num_points = int(f.readline().strip())
        for line_idx in range(num_points):
            line = f.readline()
            vals = line.strip().split()
            points.append([float(vals[0]), float(vals[1]), float(vals[2])])
        for line_idx in range(num_points):
            line = f.readline()
            vals = line.strip().split()
            normals.append([float(vals[0]), float(vals[1]), float(vals[2])])
        for line_idx in range(num_points):
            line = f.readline()
            vals = line.strip().split()
            area.append([float(vals[0])])
    assert len(points)==len(normals)
    assert len(points)==len(area)
    return np.array(points), np.array(normals), np.array(area)