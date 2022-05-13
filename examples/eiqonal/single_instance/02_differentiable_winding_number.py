import time
import math
import numpy as np
import matplotlib
import trimesh
import torch
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
import matplotlib.pyplot as plt
from xyzna_reader import read_xyzna
from vis_linspecer import linspecer



def compute_winding_torch(points, normals, area, q):
    points = points.permute(0, 2, 1)
    normals = normals.permute(0, 2, 1)
    area = area.permute(0, 2, 1)
    points = points.unsqueeze(-2).unsqueeze(0)
    normals = normals.unsqueeze(-2).unsqueeze(0)
    area = area.unsqueeze(-2).unsqueeze(0)
    q = q.unsqueeze(-1)

    winding_number = torch.sum((torch.stack([torch.stack([torch.sum(area*(points - q[:, :, q_idx, q_idy, :])*normals,2)
        for q_idx in range(q.size(-3))], 0) for q_idy in range(q.size(-2))],
        0)) / (torch.stack([torch.stack([torch.sum(torch.sqrt((points - q[:, :, q_idx, q_idy, :])**2),2)
        for q_idx in range(q.size(-3))], 0) for q_idy in range(q.size(-2))], 0))**3, -1)

    winding_number = winding_number.permute(2,3,0,1,4)
    return winding_number


def VisMC(VD, Threshold=0.5, path='MCs'):
    # Padding required to remove artifacts in the isosurfaces..
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector
    VD = np.pad(VD,2,pad_with)
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes((VD>0.5).astype('float'), pitch=1.0)
    return mesh

def main():

    num_probe_points = 25
    q_0 = torch.linspace(-0.5, 0.5, num_probe_points)
    qxx, qyy, qzz = torch.meshgrid(q_0,q_0,q_0)
    q = torch.stack((qxx, qyy, qzz), 0)

    winding_dict = {}
    plt.figure()

    points, normals, area = read_xyzna('model.xyzna')
    
    q = q.cuda().unsqueeze(0)
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    normals = torch.from_numpy(normals).unsqueeze(0).cuda()
    area = torch.from_numpy(area).unsqueeze(0).cuda()

    start = time.time()
    winding = compute_winding_torch(points, normals, area, q)

    mesh = VisMC(winding.detach().cpu().numpy().squeeze(), Threshold=0.5)
    with open('mc.obj','w') as f:
        trimesh.exchange.export.export_mesh(mesh, f, file_type='obj')
    end = time.time()
    print('time taken:', (end-start)*100)
    print(winding.shape)

    # for c_idx, noise_lvl in enumerate(sorted(noise_list)):
    #     points, normals, area = read_xyzna('noisy/%d/model_50.xyzna'%((noise_lvl)))
    #     start = time.time()
    #     winding = compute_winding_cuda(points, normals, area, q)
    #     winding = np.reshape(winding, (num_probe_points,num_probe_points,num_probe_points))
    #     mesh = VisMC(winding.squeeze(), Threshold=0.5)
    #     with open('mc_%d.obj'%noise_lvl,'w') as f:
    #         trimesh.exchange.export.export_mesh(mesh, f, file_type='obj')
    #     end = time.time()
    #     print('time taken:', (end-start)*100)
    #     print(winding.shape)



    # num_probe_points = 10
    # q_0 = np.linspace(-0.6,0.6,num_probe_points)
    # qxx, qyy, qzz = np.meshgrid(q_0,q_0,q_0)
    # q = np.array([qxx, qyy, qzz]).T

    # winding_dict = {}
    # plt.figure()
    # noise_list = [f for f in range(0,101,10)]
    # col = linspecer(len(noise_list))
    # for c_idx, noise_lvl in enumerate(sorted(noise_list)):
    #     points, normals, area = read_xyzna('noisy/%d/model_50.xyzna'%((noise_lvl)))
    #     start = time.time()
    #     winding = compute_winding_3d(points, normals, area, q)
    #     mesh = VisMC(winding.squeeze(), Threshold=0.5)
    #     with open('mc_%d.obj'%noise_lvl,'w') as f:
    #         trimesh.exchange.export.export_mesh(mesh, f, file_type='obj')
    #     end = time.time()
    #     print('time taken:', (end-start)*100)
    #     print(winding.shape)

if __name__ == '__main__':
    main()