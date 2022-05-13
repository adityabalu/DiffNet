def write_xyz(filename, points):
    with open(filename,'w') as f:
        for i in range(0,points.shape[0]):
            f.write('%0.6f %0.6f %0.6f\n'%(points[i,0], points[i,1], points[i,2]))

def write_xyzna(filename, points, normals, areas):
    with open(filename,'w') as f:
        f.write('%d\n'%(points.shape[0]))
        for i in range(0,points.shape[0]):
            f.write('%0.18f %0.18f %0.18f\n'%(points[i,0], points[i,1], points[i,2]))
        for i in range(0,points.shape[0]):
            f.write('%0.18f %0.18f %0.18f\n'%(normals[i,0], normals[i,1], normals[i,2]))
        for i in range(0,points.shape[0]):
            f.write('%0.18f\n'%(areas[i]))