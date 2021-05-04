import torch
from torch import nn
import numpy as np

def grid(lx, ly, nx, ny):
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    xv, yv = np.meshgrid(x, y)

    # xv_4d = make_4d_float_tensor(xv, args)
    # yv_4d = make_4d_float_tensor(yv, args)
    return xv, yv

def grid_3d(lx, ly, lz, nx, ny, nz):
    xline = np.linspace(0, lx, nx)
    yline = np.linspace(0, ly, ny)
    zline = np.linspace(0, lz, nz)
    
    x = np.kron(np.ones((1,nz)),(np.kron(np.ones((1,ny)),xline)))
    y = np.kron(np.ones((1,nz)),(np.kron(yline,np.ones((1,nx)))))
    z = np.kron(zline,np.ones((nx*ny)))

    x = np.reshape(x, (nz, ny, nx))
    y = np.reshape(y, (nz, ny, nx))
    z = np.reshape(z, (nz, ny, nx))

    # xv_4d = make_4d_float_tensor(xv, args)
    # yv_4d = make_4d_float_tensor(yv, args)
    return x,y,z

def calc_BC_poisson(args):
    if args.nsd == 2:
        homogeneous_1d = np.zeros(args.output_size)
        homogeneous = torch.FloatTensor(np.repeat(homogeneous_1d[np.newaxis, np.newaxis, :], args.batch_size, axis=0))
        
        x_1d = np.linspace(0,1,args.output_size)
        y_1d = np.linspace(0,1,args.output_size)
        x = torch.FloatTensor(np.repeat(x_1d[np.newaxis, np.newaxis, :], args.batch_size, axis=0))
        y = torch.FloatTensor(np.repeat(y_1d[np.newaxis, np.newaxis, :], args.batch_size, axis=0))
    elif args.nsd == 3:
        homogeneous_2d = np.zeros((args.output_size, args.output_size))
        homogeneous = torch.FloatTensor(np.repeat(homogeneous_2d[np.newaxis, np.newaxis, :, :], args.batch_size, axis=0))
        
        x_1d = np.linspace(0,1,args.output_size)
        y_1d = np.linspace(0,1,args.output_size)
        x_2d = np.repeat(x_1d[np.newaxis, :], args.output_size, axis=0)
        y_2d = np.repeat(y_1d[:, np.newaxis], args.output_size, axis=1)
        x = torch.FloatTensor(np.repeat(x_2d[np.newaxis, np.newaxis, :], args.batch_size, axis=0))
        y = torch.FloatTensor(np.repeat(y_2d[np.newaxis, np.newaxis, :], args.batch_size, axis=0))


    if (args.manufactured_sol):
        # print("in BC, manufactured_sol")
        bc_x_0 = homogeneous
        bc_x_1 = homogeneous
        bc_y_0 = homogeneous
        bc_y_1 = homogeneous
    elif (args.log_K):
        # print("in BC, real case")
        bc_x_0 = y ** 0
        bc_x_1 = homogeneous
        bc_y_0 = homogeneous
        bc_y_1 = homogeneous
        bc_z_0 = homogeneous
        bc_z_1 = homogeneous
    else:
        # print("in BC, real case")
        bc_x_0 = 4 * y * (1.0 - y)
        bc_x_1 = homogeneous
        bc_y_0 = homogeneous
        bc_y_1 = homogeneous

    # print("homogeneous = ", homogeneous.type(), ", size = ", homogeneous.shape)
    # print("bc_x_0 = ", bc_x_0.type(), ", size = ", bc_x_0.shape)
    
    list_of_bc = [bc_x_0, bc_x_1, bc_y_0, bc_y_1, bc_z_0, bc_z_1]
    return list_of_bc

def create_coefficient_samples(args):
    n_samples_1d = args.n_samples_1d
    if args.sampling_method == 'structured':
        n_samples = (n_samples_1d ** args.n_sum_nu)
    else:
        n_samples = args.n_samples
    # print("n_samples = ", n_samples)
    a_min = -3
    a_max = 3
    coeff_1d = np.linspace(a_min, a_max, n_samples_1d)

    coeff_full_batch = np.zeros((n_samples,args.n_sum_nu))

    for j in range(args.n_sum_nu):
        scalar_repeat = n_samples_1d**j
        array_repeat = n_samples_1d**(args.n_sum_nu - (j+1))
        coeff_full_batch[:,j] = np.kron(np.ones(array_repeat), coeff_1d.repeat(scalar_repeat))

    return coeff_full_batch

def get_fields_of_coefficients(args, scalar_coeff_batch, mesh_size):
    max_terms_in_nu_sum = 6
    # if args.nsd == 2:
    #     all_coeff_tensors = (torch.zeros((max_terms_in_nu_sum,args.batch_size,1,mesh_size,mesh_size)))
    # elif args.nsd == 3:
    #     all_coeff_tensors = (np.zeros((args.n_sum_nu,args.batch_size,1,mesh_size,mesh_size, mesh_size)))

    nsd = args.nsd
    n_sum_nu = args.dataset.nu_calc_info.n_sum_nu
    # batch_size = args.optimization.batch_size
    # the batch_size here isn't necessarily the optimization batch size
    # it is just the number of values queried, i.e., the size of the coeff_batch
    batch_size = scalar_coeff_batch.shape[0]

    # print("size of scalar_coeff_batch = ", scalar_coeff_batch.shape, flush=True)
    # print("scalar_coeff_batch: size = ", scalar_coeff_batch.shape, ", type = ", scalar_coeff_batch.type(), flush=True)
    all_coeff_tensors = []
    for i in range(n_sum_nu):        
        coeff_rand_list = scalar_coeff_batch[:,i]
        # print("size of coeff_rand_list = ", coeff_rand_list.shape, flush=True)

        if nsd == 2:
            # coeff_repeated = np.repeat(coeff_rand_list.cpu().numpy().reshape(batch_size, 1), (mesh_size**2))
            # coeff_4d_np = coeff_repeated.reshape(batch_size, 1, mesh_size, mesh_size)
            coeff_4d_np = (coeff_rand_list.repeat(mesh_size**2)).reshape(batch_size, 1, mesh_size, mesh_size)            
        elif nsd == 3:
            # coeff_repeated = np.repeat(coeff_rand_list.cpu().numpy().reshape(batch_size, 1), (mesh_size**3))
            # coeff_4d_np = coeff_repeated.reshape(batch_size, 1, mesh_size, mesh_size, mesh_size)    
            coeff_4d_np = (coeff_rand_list.repeat(mesh_size**3)).reshape(batch_size, 1, mesh_size, mesh_size, mesh_size)
            # print("coeff_repeated: size = ", coeff_repeated.shape, flush=True)
            # print(coeff_4d_np)
    
        # all_coeff_tensors.append(torch.FloatTensor(coeff_4d_np).type_as(scalar_coeff_batch))
        all_coeff_tensors.append(coeff_4d_np.type_as(scalar_coeff_batch))
        # print("all_coeff_tensors[0]: size = ", all_coeff_tensors[0].shape, ", type = ", all_coeff_tensors[0].type(), flush=True)


    if n_sum_nu < max_terms_in_nu_sum:
        for i in range(n_sum_nu,max_terms_in_nu_sum):
            if nsd == 2:
                all_coeff_tensors.append((torch.zeros((batch_size,1,mesh_size,mesh_size))).type_as(scalar_coeff_batch))
            elif nsd == 3:
                all_coeff_tensors.append((torch.zeros((batch_size,1,mesh_size,mesh_size,mesh_size))).type_as(scalar_coeff_batch))
    return all_coeff_tensors   

def calc_coefficients_nu_logK(args):
    a_min = -3
    a_max = 3
    array_of_coeff_lists = np.zeros((args.batch_size, args.n_sum_nu), dtype=np.float32)
    list_of_coeff_tensors = []
    for i in range(args.n_sum_nu):        
        coeff_rand_list = a_min + (a_max - a_min) * np.random.rand(args.batch_size)
        
        if args.nsd == 2:
            coeff_repeated = np.repeat(coeff_rand_list.reshape(args.batch_size, 1), (args.output_size**2))
            coeff_4d_np = coeff_repeated.reshape(args.batch_size, 1, args.output_size, args.output_size)
        elif args.nsd == 3:
            coeff_repeated = np.repeat(coeff_rand_list.reshape(args.batch_size, 1), (args.output_size**3))
            coeff_4d_np = coeff_repeated.reshape(args.batch_size, 1, args.output_size, args.output_size, args.output_size)
        
        array_of_coeff_lists[:, i] = coeff_rand_list
        list_of_coeff_tensors.append(torch.FloatTensor(coeff_4d_np))
    return array_of_coeff_lists, list_of_coeff_tensors

def calculate_omega_based_on_eta(eta):
    tol = 1e-6
    if np.abs(eta - 0.1) < tol:
        omega = np.array([
                     2.627675432985797,
                     5.307324799118128,
                     8.067135580679963,
                    10.908707509765620,
                    13.819191590843053,
                    16.782691099052428,
                    19.785505130248573,
                    22.817253043828419,
                    25.870442560222948,
                    28.939736049582585
            ])
    elif np.abs(eta - 0.2) < tol:
        omega = np.array([
                     2.284453709564703,
                     4.761288969346805,
                     7.463676172029721,
                    10.326611007844360,
                    13.286241503970587,
                    16.303128640923813,
                    19.355160454004977,
                    22.429811599309446,
                    25.519693779498752,
                    28.620245932841211
            ])
    elif np.abs(eta - 0.5) < tol:
        omega = np.array([
                     1.720667178038759,
                     4.057515676220868,
                     6.851236918963457,
                     9.826360878869767,
                    12.874596358343892,
                    15.957331424826481,
                    19.058668810723926,
                    22.171076812994045,
                    25.290574447713286,
                    28.414873450382377
            ])
    elif np.abs(eta - 0.7) < tol:
        omega = np.array([
                     1.513246031735345,
                     3.851891808005561,
                     6.703141757332143,
                     9.716730053822916,
                    12.788857060379099,
                    15.887318867290485,
                    18.999652186088099,
                    22.120134252280451,
                    25.245793691314280,
                    28.374941402170549
            ])
    elif np.abs(eta - 1.0) < tol:
        omega = np.array([
                     1.306542374188806,
                     3.673194406304252,
                     6.584620042564173,
                     9.631684635691871,
                    12.723240784131329,
                    15.834105369332415,
                    18.954971410841591,
                    22.081659635942589,
                    25.212026888550827,
                    28.344864149599882
            ])
    return omega

def construct_KL_sum(args, x, y, z, rand_tensor_list):
    local_batch_size = rand_tensor_list[0].shape[0]
    # print("local_batch_size.shape = ", local_batch_size)
    # for i in range(6):
    #     print("a"+str(i)+" = ", rand_tensor_list[i])


    eta_x = args.dataset.nu_calc_info.eta_x
    eta_y = args.dataset.nu_calc_info.eta_y
    eta_z = args.dataset.nu_calc_info.eta_z
    
    omega_x = calculate_omega_based_on_eta(eta_x)
    omega_y = calculate_omega_based_on_eta(eta_y)
    omega_z = calculate_omega_based_on_eta(eta_z)

    sin = torch.sin
    cos = torch.cos

    sigma_x = 1
    sigma_y = 1
    sigma_z = 1
    
    # omegaX omegaY are vectors
    lambda_x = 2.0 * eta_x * sigma_x / (1.0 + (eta_x * omega_x) ** 2);
    lambda_y = 2.0 * eta_y * sigma_y / (1.0 + (eta_y * omega_y) ** 2);
    lambda_z = 2.0 * eta_z * sigma_z / (1.0 + (eta_z * omega_z) ** 2);
    
    kl_sum = 0*x # 4d

    if args.nsd == 2:
        for i in range(6):
            # kl_sum += rand_tensor_list[0] * sqrt(lambda_x[0]) * sqrt(lambda_y[0]) * (eta_x * omega_x[0] * cos(omega_x[0] * x) + sin(omega_x[0] * x)) * (eta_y * omega_y[0] * cos(omega_y[0] * y) + sin(omega_y[0] * y))
            kl_sum += rand_tensor_list[i] * np.sqrt(lambda_x[i]) * np.sqrt(lambda_y[i]) * (eta_x * omega_x[i] * cos(omega_x[i] * x) + sin(omega_x[i] * x)) * (eta_y * omega_y[i] * cos(omega_y[i] * y) + sin(omega_y[i] * y))   

    if args.nsd == 3:
        for i in range(6):
            # kl_sum += rand_tensor_list[0] * sqrt(lambda_x[0]) * sqrt(lambda_y[0]) * (eta_x * omega_x[0] * cos(omega_x[0] * x) + sin(omega_x[0] * x)) * (eta_y * omega_y[0] * cos(omega_y[0] * y) + sin(omega_y[0] * y))
            kl_sum += rand_tensor_list[i] * np.sqrt(lambda_x[i]) * np.sqrt(lambda_y[i]) * np.sqrt(lambda_z[i]) * (eta_x * omega_x[i] * cos(omega_x[i] * x) + sin(omega_x[i] * x)) * (eta_y * omega_y[i] * cos(omega_y[i] * y) + sin(omega_y[i] * y)) * (eta_z * omega_z[i] * cos(omega_z[i] * z) + sin(omega_z[i] * z))

    # print("type of kl_sum = ", kl_sum.type(), ", size = ", kl_sum.shape)

    return kl_sum

def create_nu_tensor(args, list_of_coeff_tensors, x, y, z):
    pi = np.pi
    sin = torch.sin
    cos = torch.cos
    exp = torch.exp
    a1 = list_of_coeff_tensors[0]
    a2 = list_of_coeff_tensors[1]
    a3 = list_of_coeff_tensors[2]
    a4 = list_of_coeff_tensors[3]
    a5 = list_of_coeff_tensors[4]
    a6 = list_of_coeff_tensors[5]

    local_batch_size = a1.shape[0]
    if args.nsd == 2:
        x = x.repeat((local_batch_size,1,1,1))
        y = y.repeat((local_batch_size,1,1,1))
        z = z.repeat((local_batch_size,1,1,1))
    elif args.nsd == 3:
        x = x.repeat((local_batch_size,1,1,1,1))
        y = y.repeat((local_batch_size,1,1,1,1))
        z = z.repeat((local_batch_size,1,1,1,1))

    # print(a1)
    # print(a2)
#     nu_4d_tensor_gp = x**0
#     nu_4d_tensor_gp = sin(pi * x) * sin(pi * y)
    # nu_4d_tensor_gp = exp(x)
    kl_sum_flag = args.dataset.nu_calc_info.KL_sum
    if kl_sum_flag:
        nu_4d_tensor = exp(construct_KL_sum(args, x, y, z, [a1, a2, a3, a4, a5, a6]))
    else:
        nu_4d_tensor = exp(0 * x + a1*(cos(2*pi*x)/2 - 3/5)*(cos(2*pi*y)/2 - 3/5) + a2*(cos(4*pi*x)/2 - 3/5)*(cos(4*pi*y)/2 - 3/5) + a3*(cos(6*pi*x)/2 - 3/5)*(cos(6*pi*y)/2 - 3/5) + a4*(cos(8*pi*x)/2 - 3/5)*(cos(8*pi*y)/2 - 3/5) + a5*(cos(10*pi*x)/2 - 3/5)*(cos(10*pi*y)/2 - 3/5) + a6*(cos(12*pi*x)/2 - 3/5)*(cos(12*pi*y)/2 - 3/5))
        # nu_4d_tensor = a1 * sin(pi * x) * sin(pi * y)
    
    # forcing_4d_tensor = 0 * x

    # print("type of a1 = ", a1.type(), ", size = ", a1.shape)    
    # print("type of a6 = ", a6.type(), ", size = ", a6.shape)    
    # print("type of nu_4d_np = ", nu_4d_tensor.type(), ", size = ", nu_4d_tensor.shape)
    # print("type of forcing_4d_np = ", forcing_4d_tensor.type(), ", size = ", forcing_4d_tensor.shape)
    # forcing_4d_tensor = torch.FloatTensor(forcing_4d_tensor)

    # list_of_bc = calc_BC_poisson(args)

    # print("type of nu_4d = ", nu_4d_tensor.type())
    # np.savez("/data/bkhara/PDE_sampler/poisson-test-k-3d/data_check.npz", coeff=array_of_coeff_nu, nu=nu_4d_tensor.cpu().detach().numpy(), x=x.cpu().detach().numpy(), y=y.cpu().detach().numpy(), z=z.cpu().detach().numpy())
    # exit()

    return nu_4d_tensor

def create_forcing_tensor(args, list_of_coeff_tensors, x, y, z):
    pi = np.pi
    sin = torch.sin
    cos = torch.cos
    exp = torch.exp
    a1 = list_of_coeff_tensors[0]
    a2 = list_of_coeff_tensors[1]
    a3 = list_of_coeff_tensors[2]
    a4 = list_of_coeff_tensors[3]
    a5 = list_of_coeff_tensors[4]
    a6 = list_of_coeff_tensors[5]

    local_batch_size = a1.shape[0]
    if args.nsd == 2:
        x = x.repeat((local_batch_size,1,1,1))
        y = y.repeat((local_batch_size,1,1,1))
        z = z.repeat((local_batch_size,1,1,1))
    elif args.nsd == 3:
        x = x.repeat((local_batch_size,1,1,1))
        y = y.repeat((local_batch_size,1,1,1))
        z = z.repeat((local_batch_size,1,1,1))

    nu = 1.0
    # forcing_4d_tensor = nu*(2*a1*pi**2*cos(2*pi*x)*(cos(2*pi*y)/2 - 1/2) + 8*a2*pi**2*cos(4*pi*x)*(cos(4*pi*y)/2 - 1/2) + 18*a3*pi**2*cos(6*pi*x)*(cos(6*pi*y)/2 - 1/2) + 32*a4*pi**2*cos(8*pi*x)*(cos(8*pi*y)/2 - 1/2) + 50*a5*pi**2*cos(10*pi*x)*(cos(10*pi*y)/2 - 1/2) + 72*a6*pi**2*cos(12*pi*x)*(cos(12*pi*y)/2 - 1/2)) + nu*(2*a1*pi**2*cos(2*pi*y)*(cos(2*pi*x)/2 - 1/2) + 8*a2*pi**2*cos(4*pi*y)*(cos(4*pi*x)/2 - 1/2) + 18*a3*pi**2*cos(6*pi*y)*(cos(6*pi*x)/2 - 1/2) + 32*a4*pi**2*cos(8*pi*y)*(cos(8*pi*x)/2 - 1/2) + 50*a5*pi**2*cos(10*pi*y)*(cos(10*pi*x)/2 - 1/2) + 72*a6*pi**2*cos(12*pi*y)*(cos(12*pi*x)/2 - 1/2))
    # forcing_4d_tensor = nu*exp(a1*(cos(2*pi*x)/2 - 1/2)*(cos(2*pi*y)/2 - 1/2) + a2*(cos(4*pi*x)/2 - 1/2)*(cos(4*pi*y)/2 - 1/2) + a3*(cos(6*pi*x)/2 - 1/2)*(cos(6*pi*y)/2 - 1/2) + a4*(cos(8*pi*x)/2 - 1/2)*(cos(8*pi*y)/2 - 1/2) + a5*(cos(10*pi*x)/2 - 1/2)*(cos(10*pi*y)/2 - 1/2) + a6*(cos(12*pi*x)/2 - 1/2)*(cos(12*pi*y)/2 - 1/2))*(2*a1*pi**2*cos(2*pi*x)*(cos(2*pi*y)/2 - 1/2) + 8*a2*pi**2*cos(4*pi*x)*(cos(4*pi*y)/2 - 1/2) + 18*a3*pi**2*cos(6*pi*x)*(cos(6*pi*y)/2 - 1/2) + 32*a4*pi**2*cos(8*pi*x)*(cos(8*pi*y)/2 - 1/2) + 50*a5*pi**2*cos(10*pi*x)*(cos(10*pi*y)/2 - 1/2) + 72*a6*pi**2*cos(12*pi*x)*(cos(12*pi*y)/2 - 1/2)) - nu*exp(a1*(cos(2*pi*x)/2 - 1/2)*(cos(2*pi*y)/2 - 1/2) + a2*(cos(4*pi*x)/2 - 1/2)*(cos(4*pi*y)/2 - 1/2) + a3*(cos(6*pi*x)/2 - 1/2)*(cos(6*pi*y)/2 - 1/2) + a4*(cos(8*pi*x)/2 - 1/2)*(cos(8*pi*y)/2 - 1/2) + a5*(cos(10*pi*x)/2 - 1/2)*(cos(10*pi*y)/2 - 1/2) + a6*(cos(12*pi*x)/2 - 1/2)*(cos(12*pi*y)/2 - 1/2))*(a1*pi*sin(2*pi*y)*(cos(2*pi*x)/2 - 1/2) + 2*a2*pi*sin(4*pi*y)*(cos(4*pi*x)/2 - 1/2) + 3*a3*pi*sin(6*pi*y)*(cos(6*pi*x)/2 - 1/2) + 4*a4*pi*sin(8*pi*y)*(cos(8*pi*x)/2 - 1/2) + 5*a5*pi*sin(10*pi*y)*(cos(10*pi*x)/2 - 1/2) + 6*a6*pi*sin(12*pi*y)*(cos(12*pi*x)/2 - 1/2))**2 - nu*exp(a1*(cos(2*pi*x)/2 - 1/2)*(cos(2*pi*y)/2 - 1/2) + a2*(cos(4*pi*x)/2 - 1/2)*(cos(4*pi*y)/2 - 1/2) + a3*(cos(6*pi*x)/2 - 1/2)*(cos(6*pi*y)/2 - 1/2) + a4*(cos(8*pi*x)/2 - 1/2)*(cos(8*pi*y)/2 - 1/2) + a5*(cos(10*pi*x)/2 - 1/2)*(cos(10*pi*y)/2 - 1/2) + a6*(cos(12*pi*x)/2 - 1/2)*(cos(12*pi*y)/2 - 1/2))*(a1*pi*sin(2*pi*x)*(cos(2*pi*y)/2 - 1/2) + 2*a2*pi*sin(4*pi*x)*(cos(4*pi*y)/2 - 1/2) + 3*a3*pi*sin(6*pi*x)*(cos(6*pi*y)/2 - 1/2) + 4*a4*pi*sin(8*pi*x)*(cos(8*pi*y)/2 - 1/2) + 5*a5*pi*sin(10*pi*x)*(cos(10*pi*y)/2 - 1/2) + 6*a6*pi*sin(12*pi*x)*(cos(12*pi*y)/2 - 1/2))**2 + nu*exp(a1*(cos(2*pi*x)/2 - 1/2)*(cos(2*pi*y)/2 - 1/2) + a2*(cos(4*pi*x)/2 - 1/2)*(cos(4*pi*y)/2 - 1/2) + a3*(cos(6*pi*x)/2 - 1/2)*(cos(6*pi*y)/2 - 1/2) + a4*(cos(8*pi*x)/2 - 1/2)*(cos(8*pi*y)/2 - 1/2) + a5*(cos(10*pi*x)/2 - 1/2)*(cos(10*pi*y)/2 - 1/2) + a6*(cos(12*pi*x)/2 - 1/2)*(cos(12*pi*y)/2 - 1/2))*(2*a1*pi**2*cos(2*pi*y)*(cos(2*pi*x)/2 - 1/2) + 8*a2*pi**2*cos(4*pi*y)*(cos(4*pi*x)/2 - 1/2) + 18*a3*pi**2*cos(6*pi*y)*(cos(6*pi*x)/2 - 1/2) + 32*a4*pi**2*cos(8*pi*y)*(cos(8*pi*x)/2 - 1/2) + 50*a5*pi**2*cos(10*pi*y)*(cos(10*pi*x)/2 - 1/2) + 72*a6*pi**2*cos(12*pi*y)*(cos(12*pi*x)/2 - 1/2))
    forcing_4d_tensor = 2. * nu * pi**2 * sin(pi * x) * sin(pi * y)

    # print("type of a1 = ", a1.type(), ", size = ", a1.shape)    
    # print("type of a6 = ", a6.type(), ", size = ", a6.shape)    
    # print("type of nu_4d_np = ", nu_4d_tensor.type(), ", size = ", nu_4d_tensor.shape)
    # print("type of forcing_4d_np = ", forcing_4d_tensor.type(), ", size = ", forcing_4d_tensor.shape)
    # forcing_4d_tensor = torch.FloatTensor(forcing_4d_tensor)

    # list_of_bc = calc_BC_poisson(args)

    # print("type of nu_4d = ", nu_4d_tensor.type())
    # np.savez("/data/bkhara/PDE_sampler/poisson-test-k-3d/data_check.npz", coeff=array_of_coeff_nu, nu=nu_4d_tensor.cpu().detach().numpy(), x=x.cpu().detach().numpy(), y=y.cpu().detach().numpy(), z=z.cpu().detach().numpy())
    # exit()

    return forcing_4d_tensor

def create_forcing_tensor_reacdiff(args, list_of_coeff_tensors, x, y, z):
    pi = np.pi
    sin = torch.sin
    cos = torch.cos
    exp = torch.exp
    tanh = torch.tanh
    a1 = list_of_coeff_tensors[0]
    a2 = list_of_coeff_tensors[1]
    a3 = list_of_coeff_tensors[2]
    a4 = list_of_coeff_tensors[3]
    a5 = list_of_coeff_tensors[4]
    a6 = list_of_coeff_tensors[5]

    local_batch_size = a1.shape[0]
    if args.nsd == 2:
        x = x.repeat((local_batch_size,1,1,1))
        y = y.repeat((local_batch_size,1,1,1))
        z = z.repeat((local_batch_size,1,1,1))
    elif args.nsd == 3:
        x = x.repeat((local_batch_size,1,1,1))
        y = y.repeat((local_batch_size,1,1,1))
        z = z.repeat((local_batch_size,1,1,1))

    kappa = args.reaction_coeff_kappa
    # forcing_4d_tensor = nu*(2*a1*pi**2*cos(2*pi*x)*(cos(2*pi*y)/2 - 1/2) + 8*a2*pi**2*cos(4*pi*x)*(cos(4*pi*y)/2 - 1/2) + 18*a3*pi**2*cos(6*pi*x)*(cos(6*pi*y)/2 - 1/2) + 32*a4*pi**2*cos(8*pi*x)*(cos(8*pi*y)/2 - 1/2) + 50*a5*pi**2*cos(10*pi*x)*(cos(10*pi*y)/2 - 1/2) + 72*a6*pi**2*cos(12*pi*x)*(cos(12*pi*y)/2 - 1/2)) + nu*(2*a1*pi**2*cos(2*pi*y)*(cos(2*pi*x)/2 - 1/2) + 8*a2*pi**2*cos(4*pi*y)*(cos(4*pi*x)/2 - 1/2) + 18*a3*pi**2*cos(6*pi*y)*(cos(6*pi*x)/2 - 1/2) + 32*a4*pi**2*cos(8*pi*y)*(cos(8*pi*x)/2 - 1/2) + 50*a5*pi**2*cos(10*pi*y)*(cos(10*pi*x)/2 - 1/2) + 72*a6*pi**2*cos(12*pi*y)*(cos(12*pi*x)/2 - 1/2))
    # forcing_4d_tensor = nu*exp(a1*(cos(2*pi*x)/2 - 1/2)*(cos(2*pi*y)/2 - 1/2) + a2*(cos(4*pi*x)/2 - 1/2)*(cos(4*pi*y)/2 - 1/2) + a3*(cos(6*pi*x)/2 - 1/2)*(cos(6*pi*y)/2 - 1/2) + a4*(cos(8*pi*x)/2 - 1/2)*(cos(8*pi*y)/2 - 1/2) + a5*(cos(10*pi*x)/2 - 1/2)*(cos(10*pi*y)/2 - 1/2) + a6*(cos(12*pi*x)/2 - 1/2)*(cos(12*pi*y)/2 - 1/2))*(2*a1*pi**2*cos(2*pi*x)*(cos(2*pi*y)/2 - 1/2) + 8*a2*pi**2*cos(4*pi*x)*(cos(4*pi*y)/2 - 1/2) + 18*a3*pi**2*cos(6*pi*x)*(cos(6*pi*y)/2 - 1/2) + 32*a4*pi**2*cos(8*pi*x)*(cos(8*pi*y)/2 - 1/2) + 50*a5*pi**2*cos(10*pi*x)*(cos(10*pi*y)/2 - 1/2) + 72*a6*pi**2*cos(12*pi*x)*(cos(12*pi*y)/2 - 1/2)) - nu*exp(a1*(cos(2*pi*x)/2 - 1/2)*(cos(2*pi*y)/2 - 1/2) + a2*(cos(4*pi*x)/2 - 1/2)*(cos(4*pi*y)/2 - 1/2) + a3*(cos(6*pi*x)/2 - 1/2)*(cos(6*pi*y)/2 - 1/2) + a4*(cos(8*pi*x)/2 - 1/2)*(cos(8*pi*y)/2 - 1/2) + a5*(cos(10*pi*x)/2 - 1/2)*(cos(10*pi*y)/2 - 1/2) + a6*(cos(12*pi*x)/2 - 1/2)*(cos(12*pi*y)/2 - 1/2))*(a1*pi*sin(2*pi*y)*(cos(2*pi*x)/2 - 1/2) + 2*a2*pi*sin(4*pi*y)*(cos(4*pi*x)/2 - 1/2) + 3*a3*pi*sin(6*pi*y)*(cos(6*pi*x)/2 - 1/2) + 4*a4*pi*sin(8*pi*y)*(cos(8*pi*x)/2 - 1/2) + 5*a5*pi*sin(10*pi*y)*(cos(10*pi*x)/2 - 1/2) + 6*a6*pi*sin(12*pi*y)*(cos(12*pi*x)/2 - 1/2))**2 - nu*exp(a1*(cos(2*pi*x)/2 - 1/2)*(cos(2*pi*y)/2 - 1/2) + a2*(cos(4*pi*x)/2 - 1/2)*(cos(4*pi*y)/2 - 1/2) + a3*(cos(6*pi*x)/2 - 1/2)*(cos(6*pi*y)/2 - 1/2) + a4*(cos(8*pi*x)/2 - 1/2)*(cos(8*pi*y)/2 - 1/2) + a5*(cos(10*pi*x)/2 - 1/2)*(cos(10*pi*y)/2 - 1/2) + a6*(cos(12*pi*x)/2 - 1/2)*(cos(12*pi*y)/2 - 1/2))*(a1*pi*sin(2*pi*x)*(cos(2*pi*y)/2 - 1/2) + 2*a2*pi*sin(4*pi*x)*(cos(4*pi*y)/2 - 1/2) + 3*a3*pi*sin(6*pi*x)*(cos(6*pi*y)/2 - 1/2) + 4*a4*pi*sin(8*pi*x)*(cos(8*pi*y)/2 - 1/2) + 5*a5*pi*sin(10*pi*x)*(cos(10*pi*y)/2 - 1/2) + 6*a6*pi*sin(12*pi*x)*(cos(12*pi*y)/2 - 1/2))**2 + nu*exp(a1*(cos(2*pi*x)/2 - 1/2)*(cos(2*pi*y)/2 - 1/2) + a2*(cos(4*pi*x)/2 - 1/2)*(cos(4*pi*y)/2 - 1/2) + a3*(cos(6*pi*x)/2 - 1/2)*(cos(6*pi*y)/2 - 1/2) + a4*(cos(8*pi*x)/2 - 1/2)*(cos(8*pi*y)/2 - 1/2) + a5*(cos(10*pi*x)/2 - 1/2)*(cos(10*pi*y)/2 - 1/2) + a6*(cos(12*pi*x)/2 - 1/2)*(cos(12*pi*y)/2 - 1/2))*(2*a1*pi**2*cos(2*pi*y)*(cos(2*pi*x)/2 - 1/2) + 8*a2*pi**2*cos(4*pi*y)*(cos(4*pi*x)/2 - 1/2) + 18*a3*pi**2*cos(6*pi*y)*(cos(6*pi*x)/2 - 1/2) + 32*a4*pi**2*cos(8*pi*y)*(cos(8*pi*x)/2 - 1/2) + 50*a5*pi**2*cos(10*pi*y)*(cos(10*pi*x)/2 - 1/2) + 72*a6*pi**2*cos(12*pi*y)*(cos(12*pi*x)/2 - 1/2))
    forcing_4d_tensor = kappa**2*tanh(kappa*((x - 1/2)**2 + (y - 1/2)**2 - 1/16))*(2*x - 1)**2*(tanh(kappa*((x - 1/2)**2 + (y - 1/2)**2 - 1/16))**2 - 1) - kappa**2*(tanh(kappa*((x - 1/2)**2 + (y - 1/2)**2 - 1/16))/2 - 1/2) - 2*kappa*(tanh(kappa*((x - 1/2)**2 + (y - 1/2)**2 - 1/16))**2 - 1) + kappa**2*tanh(kappa*((x - 1/2)**2 + (y - 1/2)**2 - 1/16))*(2*y - 1)**2*(tanh(kappa*((x - 1/2)**2 + (y - 1/2)**2 - 1/16))**2 - 1)

    return forcing_4d_tensor

def create_generator_input(args, nu_tensor):
    output_dim = args.output_size

    if args.nsd == 2:
        nu_tensor_for_generator = (nu_tensor.squeeze(1)).reshape(-1, output_dim * output_dim)
    elif args.nsd == 3:
        nu_tensor_for_generator = (nu_tensor.squeeze(1)).reshape(-1, output_dim * output_dim * output_dim)

    return nu_tensor_for_generator

def post_process_gen_output(args, gen_output):
    return gen_output

def create_grid(args):
    nx = args.output_size
    ny = args.output_size
    nz = args.output_size

    # batch_size = args.optimization.batch_size
    # batch_size = 1

    if args.nsd == 2:
        x_2d, y_2d = grid(1, 1, nx, ny)
        
        # x = torch.FloatTensor(np.repeat(x_2d[np.newaxis, np.newaxis, :, :], batch_size, axis=0))
        # y = torch.FloatTensor(np.repeat(y_2d[np.newaxis, np.newaxis, :, :], batch_size, axis=0))
        # z = torch.zeros(x.shape)

        x = torch.FloatTensor(x_2d[np.newaxis, np.newaxis, :, :])
        y = torch.FloatTensor(y_2d[np.newaxis, np.newaxis, :, :])
        z = torch.zeros(x.shape)

    elif args.nsd == 3:
        x_3d, y_3d, z_3d = grid_3d(1, 1, 1, nx, ny, nz)
        
        # x = torch.FloatTensor(np.repeat(x_3d[np.newaxis, np.newaxis, :, :], batch_size, axis=0))
        # y = torch.FloatTensor(np.repeat(y_3d[np.newaxis, np.newaxis, :, :], batch_size, axis=0))
        # z = torch.FloatTensor(np.repeat(z_3d[np.newaxis, np.newaxis, :, :], batch_size, axis=0))

        x = torch.FloatTensor(x_3d[np.newaxis, np.newaxis, :, :])
        y = torch.FloatTensor(y_3d[np.newaxis, np.newaxis, :, :])
        z = torch.FloatTensor(z_3d[np.newaxis, np.newaxis, :, :])

    return x, y, z

def calc_ingredients_poisson_logK_backend(args):
    pi = np.pi
    sin = torch.sin
    cos = torch.cos
    exp = torch.exp

    nx = args.output_size
    ny = args.output_size
    nz = args.output_size

    if args.nsd == 2:
        x_2d, y_2d = grid(1, 1, nx, ny)
        
        x = torch.FloatTensor(np.repeat(x_2d[np.newaxis, np.newaxis, :, :], args.batch_size, axis=0))
        y = torch.FloatTensor(np.repeat(y_2d[np.newaxis, np.newaxis, :, :], args.batch_size, axis=0))

        a1 = torch.FloatTensor(np.zeros((args.batch_size, 1, ny, nx), dtype=np.float32))
        a2 = torch.FloatTensor(np.zeros((args.batch_size, 1, ny, nx), dtype=np.float32))
        a3 = torch.FloatTensor(np.zeros((args.batch_size, 1, ny, nx), dtype=np.float32))
        a4 = torch.FloatTensor(np.zeros((args.batch_size, 1, ny, nx), dtype=np.float32))
        a5 = torch.FloatTensor(np.zeros((args.batch_size, 1, ny, nx), dtype=np.float32))
        a6 = torch.FloatTensor(np.zeros((args.batch_size, 1, ny, nx), dtype=np.float32))
    elif args.nsd == 3:
        x_3d, y_3d, z_3d = grid_3d(1, 1, 1, nx, ny, nz)
        
        x = torch.FloatTensor(np.repeat(x_3d[np.newaxis, np.newaxis, :, :], args.batch_size, axis=0))
        y = torch.FloatTensor(np.repeat(y_3d[np.newaxis, np.newaxis, :, :], args.batch_size, axis=0))
        z = torch.FloatTensor(np.repeat(z_3d[np.newaxis, np.newaxis, :, :], args.batch_size, axis=0))

        a1 = torch.FloatTensor(np.zeros((args.batch_size, 1, nz, ny, nx), dtype=np.float32))
        a2 = torch.FloatTensor(np.zeros((args.batch_size, 1, nz, ny, nx), dtype=np.float32))
        a3 = torch.FloatTensor(np.zeros((args.batch_size, 1, nz, ny, nx), dtype=np.float32))
        a4 = torch.FloatTensor(np.zeros((args.batch_size, 1, nz, ny, nx), dtype=np.float32))
        a5 = torch.FloatTensor(np.zeros((args.batch_size, 1, nz, ny, nx), dtype=np.float32))
        a6 = torch.FloatTensor(np.zeros((args.batch_size, 1, nz, ny, nx), dtype=np.float32))

    array_of_coeff_nu, list_of_coeff_tensors_nu = calc_coefficients_nu_logK(args)

    a1 = list_of_coeff_tensors_nu[0]
    if (args.n_sum_nu > 1):
        a2 = list_of_coeff_tensors_nu[1]
    if (args.n_sum_nu > 2):
        a3 = list_of_coeff_tensors_nu[2]
    if (args.n_sum_nu > 3):
        a4 = list_of_coeff_tensors_nu[3]
    if (args.n_sum_nu > 4):
        a5 = list_of_coeff_tensors_nu[4]
    if (args.n_sum_nu > 5):
        a6 = list_of_coeff_tensors_nu[5]    

    # Select the nu and forcing accordingly
    # print("manufactured_sol =", args.manufactured_sol)
    # print("type of manufactured_sol = ", type(args.manufactured_sol))
    # print("Before nu & force selection")
    if args.KL_sum:
        nu_4d_tensor = exp(construct_KL_sum(args, x, y, [a1, a2, a3, a4, a5, a6]))
    else:
        nu_4d_tensor = exp(0 * x + a1*(cos(2*pi*x)/2 - 3/5)*(cos(2*pi*y)/2 - 3/5) + a2*(cos(4*pi*x)/2 - 3/5)*(cos(4*pi*y)/2 - 3/5) + a3*(cos(6*pi*x)/2 - 3/5)*(cos(6*pi*y)/2 - 3/5) + a4*(cos(8*pi*x)/2 - 3/5)*(cos(8*pi*y)/2 - 3/5) + a5*(cos(10*pi*x)/2 - 3/5)*(cos(10*pi*y)/2 - 3/5) + a6*(cos(12*pi*x)/2 - 3/5)*(cos(12*pi*y)/2 - 3/5))
    forcing_4d_tensor = 0 * x

    # print("type of a1 = ", a1.type(), ", size = ", a1.shape)    
    # print("type of a6 = ", a6.type(), ", size = ", a6.shape)    
    # print("type of nu_4d_np = ", nu_4d_tensor.type(), ", size = ", nu_4d_tensor.shape)
    # print("type of forcing_4d_np = ", forcing_4d_tensor.type(), ", size = ", forcing_4d_tensor.shape)
    # forcing_4d_tensor = torch.FloatTensor(forcing_4d_tensor)

    output_dim = args.output_size

    if args.nsd == 2:
        nu_tensor_for_generator = (nu_4d_tensor.squeeze(1)).reshape(-1, output_dim * output_dim)
    elif args.nsd == 3:
        nu_tensor_for_generator = (nu_4d_tensor.squeeze(1)).reshape(-1, output_dim * output_dim * output_dim)

    list_of_bc = calc_BC_poisson(args)

    # print("type of nu_4d = ", nu_4d_tensor.type())
    # np.savez("/data/bkhara/PDE_sampler/poisson-test-k-3d/data_check.npz", coeff=array_of_coeff_nu, nu=nu_4d_tensor.cpu().detach().numpy(), x=x.cpu().detach().numpy(), y=y.cpu().detach().numpy(), z=z.cpu().detach().numpy())
    # exit()


    return array_of_coeff_nu, nu_4d_tensor, forcing_4d_tensor, list_of_bc, nu_tensor_for_generator

def calc_ingredients_poisson_logK(args):
    array_of_coeff_nu, nu_4d_tensor, forcing_4d_tensor, list_of_bc, nu_tensor_for_generator = calc_ingredients_poisson_logK_backend(args)
    return nu_4d_tensor, forcing_4d_tensor, list_of_bc, nu_tensor_for_generator

def calc_BC_burgers(args):
    homogeneous_1d = np.zeros(args.output_size)
    homogeneous = torch.FloatTensor(np.repeat(homogeneous_1d[np.newaxis, np.newaxis, :], args.batch_size, axis=0))
    
    x_1d = np.linspace(0,0.2,args.output_size)
    y_1d = np.linspace(0,1.0,args.output_size)
    x = torch.FloatTensor(np.repeat(x_1d[np.newaxis, np.newaxis, :], args.batch_size, axis=0))
    y = torch.FloatTensor(np.repeat(y_1d[np.newaxis, np.newaxis, :], args.batch_size, axis=0))

    coeff_rand_list = np.random.rand(args.batch_size)*6 + 2.0
    coeff_repeated = np.repeat(coeff_rand_list.reshape(args.batch_size, 1), (args.output_size))
    coeff_4d_np = coeff_repeated.reshape(args.batch_size, 1, args.output_size)
    coeff_4d = torch.FloatTensor(coeff_4d_np)

    pi = np.pi
    cos = torch.cos
    bc_x_0 = 0.5 * (1.0 - cos(pi * coeff_4d * y))
    bc_x_1 = homogeneous
    bc_y_0 = homogeneous
    bc_y_1 = homogeneous

    # print("homogeneous = ", homogeneous.type(), ", size = ", homogeneous.shape)
    # print("bc_x_0 = ", bc_x_0.type(), ", size = ", bc_x_0.shape)
    
    list_of_bc = [bc_x_0, bc_x_1, bc_y_0, bc_y_1]
    gen_input = (bc_x_0.squeeze(1)).reshape(-1, args.output_size)
    return coeff_rand_list, list_of_bc, gen_input

def calc_ingredients_burgers_backend(args):
    pi = np.pi
    sin = torch.sin
    cos = torch.cos

    nx = args.output_size
    ny = args.output_size
    x_2d, y_2d = grid(0.2, 1, nx, ny)

    x = torch.FloatTensor(np.repeat(x_2d[np.newaxis, np.newaxis, :, :], args.batch_size, axis=0))
    y = torch.FloatTensor(np.repeat(y_2d[np.newaxis, np.newaxis, :, :], args.batch_size, axis=0))

    nu_4d_tensor = 0 * x
    forcing_4d_tensor = 0 * x

    # print("type of a1 = ", a1.type(), ", size = ", a1.shape)    
    # print("type of a6 = ", a6.type(), ", size = ", a6.shape)    
    # print("type of nu_4d_np = ", nu_4d_tensor.type(), ", size = ", nu_4d_tensor.shape)
    # print("type of forcing_4d_np = ", forcing_4d_tensor.type(), ", size = ", forcing_4d_tensor.shape)
    # forcing_4d_tensor = torch.FloatTensor(forcing_4d_tensor)

    coeff_rand_list, list_of_bc, gen_input = calc_BC_burgers(args)

    return coeff_rand_list, nu_4d_tensor, forcing_4d_tensor, list_of_bc, gen_input

def calc_ingredients_burgers(args):
    coeff_rand_list, nu_4d_tensor, forcing_4d_tensor, list_of_bc, gen_input = calc_ingredients_burgers_backend(args)
    return nu_4d_tensor, forcing_4d_tensor, list_of_bc, gen_input