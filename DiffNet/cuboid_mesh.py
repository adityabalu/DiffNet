import numpy as np

class CuboidMesh():
    """docstring for CuboidMesh"""
    def __init__(self, arg):
        super(CuboidMesh, self).__init__()        
    @staticmethod
    def meshgrid_3d(x_1d, y_1d, z_1d):
        '''
        Suppose x_1d.shape = M, y_1d.shape = N, z_1d.shape = P
        Then, this function will return 3D arrays x_3d,y_3d,z_3d each of size (P,N,M)
        The numpy meshgrid gives unusual ordering.
        '''
        M = x_1d.shape[0]
        N = y_1d.shape[0]
        P = z_1d.shape[0]
        x_2d, y_2d = np.meshgrid(x_1d, y_1d)
        x_3d = np.tile(x_2d, (P,1,1))
        y_3d = np.tile(y_2d, (P,1,1))
        z_3d = np.reshape(np.repeat(z_1d, (N*M)), (P,N,M))
        # print("x_3d = \n", x_3d)
        # print("y_3d = \n", y_3d)
        # print("z_3d = \n", z_3d)
        # exit()
        return x_3d, y_3d, z_3d