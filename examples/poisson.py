import torch
from torch import nn
import numpy as np
from .fem import FEM
from .fdm import FDM
import time

from .gen_input_calc import create_grid, create_nu_tensor, get_fields_of_coefficients, create_forcing_tensor


class LinearPoisson(FEM,FDM):

	def __init__(self, args):
		self.loss_calc_type = args.pde.loss_calc_method.name
		if self.loss_calc_type == 'fem':
			FEM.__init__(self,args)
		elif self.loss_calc_type == 'fdm':
			FDM.__init__(self,args)
		self.manufactured = False
		if args.pde.name == "linear_poisson_manufactured":
			self.manufactured = True

	def loss(self, g, nu, forcing):
		if self.loss_calc_type == 'fem':
			if self.nsd == 2:
				if self.args.pde.loss_calc_method.resmin:
					return self.loss_2d_fem_resmin(g, nu, forcing)
				else:
					return self.loss_2d_fem(g, nu, forcing)
			elif self.nsd == 3:
				return self.loss_3d_fem(g, nu, forcing)
		elif self.loss_calc_type == 'fdm':
			if self.nsd == 2:
				return self.loss_2d(g, nu, forcing)
			elif self.nsd == 3:
				return self.loss_3d(g, nu, forcing)

	def compliance(self, g, nu, forcing):
		if self.loss_calc_type == 'fem':
			if self.nsd == 2:
				return self.compliance_2d_fem(g, nu, forcing)
			elif self.nsd == 3:
				return self.compliance_3d_fem(g, nu, forcing)


	def boundary_loss(self, g, list_of_bc, output_dim):
		if self.nsd == 2:
			return self.boundary_loss_2d(g, list_of_bc, output_dim)
		elif self.nsd == 3:
			return self.boundary_loss_3d(g, list_of_bc, output_dim)

	def loss_2d(self, g, nu, forcing):
		# print("g (pde) = ", g.type(), ", size = ", g.size())
		# print("g.shape = ", g.shape)
		start = time.time()   

		if self.args.boundary_opt == 1:
			############################################################
			g_x = self.derivative_x(self.pad(g))
			g_y = self.derivative_y(self.pad(g))
			if self.d2_calc_type == 0:
				g_xx = self.derivative_xx(self.pad_d2(g))
				g_yy = self.derivative_yy(self.pad_d2(g))
			elif self.d2_calc_type == 1:
				g_xx = self.derivative_x(self.pad(g_x))
				g_yy = self.derivative_y(self.pad(g_y))        
			g_laplacian = g_xx + g_yy
			# g_laplacian = self.calc_laplacian(self.pad_d2(g))
			nu_x = self.derivative_x(self.pad(nu))
			nu_y = self.derivative_y(self.pad(nu))
			############################################################
		else:
			############################################################
			p2d = (1, 1, 1, 1)
			nu_padded = nn.functional.pad(nu, p2d, 'replicate')
			if self.args.pdetype == 'linear_poisson_manufactured':
				g = nn.functional.pad(g, (1,1,1,1), 'constant', value=0)
			else:
				g = nn.functional.pad(g, (0,0,1,1), 'replicate')
				g = nn.functional.pad(g, (0,1,0,0), 'constant', value=0)
				g = nn.functional.pad(g, (1,0,0,0), 'constant', value=1)
			g_x = nn.functional.conv2d(g, self.sobelx)
			g_y = nn.functional.conv2d(g, self.sobely)
			g_xx = nn.functional.conv2d(g, self.sobelxx)
			g_yy = nn.functional.conv2d(g, self.sobelyy)
			g_laplacian = g_xx + g_yy
			nu_x = nn.functional.conv2d(nu_padded, self.sobelx)
			nu_y = nn.functional.conv2d(nu_padded, self.sobely)
			############################################################        


		# print("g   = ", g.type(), ", size = ", g.size())
		# print("g_x = ", g_x.type(), ", size = ", g_x.size())
		# print("g_y = ", g_y.type(), ", size = ", g_y.size())
		# print("g_xx = ", g_xx.type(), ", size = ", g_xx.size())
		# print("nu   = ", nu.type(), ", size = ", nu.size())
		# print("nu_x = ", nu_x.type(), ", size = ", nu_x.size())
		# print(g)
		# exit()

		grad_g_DOT_grad_nu = torch.mul(g_x, nu_x) + torch.mul(g_y, nu_y)
		res = forcing + (grad_g_DOT_grad_nu + torch.mul(nu, g_laplacian))

		end = time.time()
		# print("Time (FDM loss): ", end - start)

		res_flat = res.view(g.shape[0], -1)
		res_norm_1 = torch.norm(res_flat, p=1, dim=1) #/(output_dim*output_dim)
		res_norm_2 = torch.norm(res_flat, p=2, dim=1) #/(output_dim*output_dim)

		c1 = self.c1
		c2 = self.c2

		res_norm = c1 * res_norm_1 + c2 * res_norm_2

		# print("nu = ", nu)
		# print("nu_x = ", nu_x)
		# print("nu_y = ", nu_y)
		# print("g = ", g)
		# print("g_x = ", g_x)
		# print("g_y = ", g_y)
		# print("g_xx = ", g_xx)
		# print("g_yy = ", g_yy)
		# print("res_norm = ", res_norm)
		# exit()

		# print(res_norm)
		return res_norm

		# output_dim = g.shape[3]
		# neumann_lambda = g.shape[3]
		# x_b_right_neumann = g_x[:,:,:,-2] * neumann_lambda
		# neumann_norm = torch.norm(x_b_right_neumann, p=1, dim=2)

	def loss_3d(self, g, nu, forcing):
		# print("g (pde) = ", g.type(), ", size = ", g.size())
		# print("g.shape = ", g.shape)
		g_x = self.derivative_x(self.pad(g))
		g_y = self.derivative_y(self.pad(g))
		g_z = self.derivative_z(self.pad(g))

		if self.d2_calc_type == 0:
			g_xx = self.derivative_xx(self.pad_d2(g))
			g_yy = self.derivative_yy(self.pad_d2(g))
			g_zz = self.derivative_zz(self.pad_d2(g))
		elif self.d2_calc_type == 1:
			g_xx = self.derivative_x(self.pad(g_x))
			g_yy = self.derivative_y(self.pad(g_y))
			g_zz = self.derivative_z(self.pad(g_z))


		g_laplacian = g_xx + g_yy + g_zz
		# g_laplacian = self.calc_laplacian(self.pad_d2(g))

		nu_x = self.derivative_x(self.pad(nu))
		nu_y = self.derivative_y(self.pad(nu))
		nu_z = self.derivative_z(self.pad(nu))

		grad_g_DOT_grad_nu = torch.mul(g_x, nu_x) + torch.mul(g_y, nu_y) + torch.mul(g_z, nu_z)
		res = forcing + (grad_g_DOT_grad_nu + torch.mul(nu, g_laplacian))

		res_flat = res.view(g.shape[0], -1)
		res_norm_1 = torch.norm(res_flat, p=1, dim=1)
		res_norm_2 = torch.norm(res_flat, p=2, dim=1)

		c1 = self.c1
		c2 = self.c2

		res_norm = c1 * res_norm_1 + c2 * res_norm_2

		# np.savez("all_data.npz", g=g.cpu().detach().numpy(), nu=nu.cpu().detach().numpy(), 
		#     g_x=g_x.cpu().detach().numpy(),g_y=g_y.cpu().detach().numpy(), g_z=g_z.cpu().detach().numpy(),
		#     g_xx=g_xx.cpu().detach().numpy(), g_yy=g_yy.cpu().detach().numpy(), g_zz=g_zz.cpu().detach().numpy())

		return res_norm

		# output_dim = g.shape[3]
		# neumann_lambda = g.shape[3]
		# x_b_right_neumann = g_x[:,:,:,-2] * neumann_lambda
		# neumann_norm = torch.norm(x_b_right_neumann, p=1, dim=2)

	def loss_2d_fem(self, g, nu, forcing):
		# g = self.apply_bc(g)
		# nu = self.apply_padding_on(nu)
		# f = self.apply_padding_on(forcing)
		f = forcing

		stride = self.fem_basis_deg

		nu_gp = self.gauss_pt_evaluations(nu, stride=stride)
		f_gp = self.gauss_pt_evaluations(f, stride=stride)
		u_gp = self.gauss_pt_evaluations(g, stride=stride)
		u_x_gp = self.gauss_pt_evaluations_der_x(g, stride=stride)
		# u_xx_gp = self.gauss_pt_evaluations_der_xx(g, stride=stride)
		u_y_gp = self.gauss_pt_evaluations_der_y(g, stride=stride)
		# u_yy_gp = self.gauss_pt_evaluations_der_yy(g, stride=stride)

		transformation_jacobian = (0.5 * self.h)**2

		res_elmwise = transformation_jacobian * sum(
						self.gpw[i] * (0.5 * nu_gp[i] * (u_x_gp[i]**2 + u_y_gp[i]**2) - (u_gp[i] * f_gp[i])) 
						for i in range(self.ngp_total))

		# # Alternative way, without pythonic generator
		# res_elmwise = torch.zeros(g_gp_list[0].size()).to(device)
		# for i in range(self.ngp_total):
		#     res_elmwise += (0.25 * self.h ** 2) * self.gpw[i] * (nu_gp_list[i] * (g_x_gp_list[i]**2 + g_y_gp_list[i]**2) - 2. * (g_gp_list[i] * f_gp_list[i])) 

		res_flat = res_elmwise.view(g.shape[0], -1)
		loss = torch.sum(res_flat, 1)
		
		return loss

	def compliance_2d_fem(self, g, nu, forcing):
		# g = self.apply_bc(g)
		# nu = self.apply_padding_on(nu)
		# f = self.apply_padding_on(forcing)
		f = forcing

		stride = self.fem_basis_deg

		nu_gp = self.gauss_pt_evaluations(nu, stride=stride)
		f_gp = self.gauss_pt_evaluations(f, stride=stride)
		u_gp = self.gauss_pt_evaluations(g, stride=stride)
		u_x_gp = self.gauss_pt_evaluations_der_x(g, stride=stride)
		u_y_gp = self.gauss_pt_evaluations_der_y(g, stride=stride)

		transformation_jacobian = (0.5 * self.h)**2

		res_elmwise = transformation_jacobian * sum(
						self.gpw[i] * (0.5 * nu_gp[i] * (u_gp[i]**2) - (u_gp[i] * f_gp[i])) 
						for i in range(self.ngp_total))

		# # Alternative way, without pythonic generator
		# res_elmwise = torch.zeros(g_gp_list[0].size()).to(device)
		# for i in range(self.ngp_total):
		#     res_elmwise += (0.25 * self.h ** 2) * self.gpw[i] * (nu_gp_list[i] * (g_x_gp_list[i]**2 + g_y_gp_list[i]**2) - 2. * (g_gp_list[i] * f_gp_list[i])) 

		res_flat = res_elmwise.view(g.shape[0], -1)
		loss = torch.sum(res_flat, 1)
		
		return loss

	def loss_2d_fem_resmin(self, g, nu, forcing):
		# g = self.apply_bc(g)
		# nu = self.apply_padding_on(nu)
		# f = self.apply_padding_on(forcing)
		f = forcing

		stride = self.args.fem_basis_deg

		nu_gp = self.gauss_pt_evaluations(nu, stride=stride)
		f_gp = self.gauss_pt_evaluations(f, stride=stride)

		nu_x_gp = self.gauss_pt_evaluations_der_x(nu, stride=stride)
		nu_y_gp = self.gauss_pt_evaluations_der_y(nu, stride=stride)

		u_gp = self.gauss_pt_evaluations(g, stride=stride)
		u_x_gp = self.gauss_pt_evaluations_der_x(g, stride=stride)
		u_y_gp = self.gauss_pt_evaluations_der_y(g, stride=stride)
		u_xx_gp = self.gauss_pt_evaluations_der_xx(g, stride=stride)
		u_yy_gp = self.gauss_pt_evaluations_der_yy(g, stride=stride)        

		transformation_jacobian = (0.5 * self.h)**2

		res_elmwise = transformation_jacobian * sum(
						self.gpw[i] * ((nu_x_gp[i] * u_x_gp[i] + nu_y_gp[i] * u_y_gp[i]) + nu_gp[i] * (u_xx_gp[i] + u_yy_gp[i]) - f_gp[i])**2
						for i in range(self.ngp_total))

		# # Alternative way, without pythonic generator
		# res_elmwise = torch.zeros(g_gp_list[0].size()).to(device)
		# for i in range(self.ngp_total):
		#     res_elmwise += (0.25 * self.h ** 2) * self.gpw[i] * (nu_gp_list[i] * (g_x_gp_list[i]**2 + g_y_gp_list[i]**2) - 2. * (g_gp_list[i] * f_gp_list[i])) 

		res_flat = res_elmwise.view(g.shape[0], -1)
		loss = torch.sum(res_flat, 1)
		
		return loss

	def loss_3d_fem(self, g, nu, forcing):
		# print("In FEM loss")
		nx = self.nx
		ny = self.ny
		nelem = self.nelem
		
		# g = self.apply_bc(g)
		# nu = self.apply_padding_on(nu)
		# f = self.apply_padding_on(forcing)
		f = forcing

		stride = self.fem_basis_deg

		nu_gp = self.gauss_pt_evaluations(nu, stride=stride)
		f_gp = self.gauss_pt_evaluations(f, stride=stride)
		u_gp = self.gauss_pt_evaluations(g, stride=stride)
		u_x_gp = self.gauss_pt_evaluations_der_x(g, stride=stride)
		u_y_gp = self.gauss_pt_evaluations_der_y(g, stride=stride)
		u_z_gp = self.gauss_pt_evaluations_der_z(g, stride=stride)
		
		transformation_jacobian = (0.5 * self.h)**3

		res_elmwise = transformation_jacobian * sum(
						self.gpw[i] * (0.5 * nu_gp[i] * (u_x_gp[i]**2 + u_y_gp[i]**2 + u_z_gp[i]**2) - (u_gp[i] * f_gp[i])) 
						for i in range(self.ngp_total))

		res_flat = res_elmwise.view(g.shape[0], -1)
		loss = torch.sum(res_flat, 1)
		
		return loss

	def compliance_3d_fem(self, g, nu, forcing):
		# print("In FEM loss")
		nx = self.nx
		ny = self.ny
		nelem = self.nelem
		
		# g = self.apply_bc(g)
		# nu = self.apply_padding_on(nu)
		# f = self.apply_padding_on(forcing)
		f = forcing

		stride = self.fem_basis_deg

		nu_gp = self.gauss_pt_evaluations(nu, stride=stride)
		f_gp = self.gauss_pt_evaluations(f, stride=stride)
		u_gp = self.gauss_pt_evaluations(g, stride=stride)
		u_x_gp = self.gauss_pt_evaluations_der_x(g, stride=stride)
		u_y_gp = self.gauss_pt_evaluations_der_y(g, stride=stride)
		u_z_gp = self.gauss_pt_evaluations_der_z(g, stride=stride)
		
		transformation_jacobian = (0.5 * self.h)**3

		res_elmwise = transformation_jacobian * sum(
						self.gpw[i] * (0.5 * nu_gp[i] * (u_gp[i]**2) - (u_gp[i] * f_gp[i])) 
						for i in range(self.ngp_total))

		res_flat = res_elmwise.view(g.shape[0], -1)
		loss = torch.sum(res_flat, 1)
		
		return loss
	
	def loss_2d_fem_SS_1(self, g, nu, forcing):
		print("In FEM loss")
		nx = self.nx
		ny = self.ny
		nelem = self.nelem
		p2d = (1, 1, 1, 1)
		# print("g (pde) = ", g.type(), ", size = ", g.size())
		# print("g.shape = ", g.shape)
		# all_coeff_tensors = get_fields_of_coefficients(args, coeff_mini_batch, self.gpline_x.shape[0])
		# nu_4d_tensor_gp = create_nu_tensor(args, all_coeff_tensors, self.gpX_2d, self.gpY_2d)

		# nu_gp_0 = nn.functional.conv2d(nu, self.N_gp_0, stride=2)
		# nu_gp_1 = nn.functional.conv2d(nu, self.N_gp_1, stride=2)
		# nu_gp_2 = nn.functional.conv2d(nu, self.N_gp_2, stride=2)
		# nu_gp_3 = nn.functional.conv2d(nu, self.N_gp_3, stride=2)

		start = time.time()

		# nu_gp = torch.zeros((self.ngp_linear, args.batch_size, 1, nelem, nelem)).to(device)
		nu_gp = []
		nu_gp.append(nn.functional.conv2d(nu, self.N_gp_0, stride=1))
		nu_gp.append(nn.functional.conv2d(nu, self.N_gp_1, stride=1))
		nu_gp.append(nn.functional.conv2d(nu, self.N_gp_2, stride=1))
		nu_gp.append(nn.functional.conv2d(nu, self.N_gp_3, stride=1))

		kMat_gp = [[[] for _ in range(self.nbf)] for _ in range(self.nbf)]
		for i in range(self.nbf):
			for j in range(self.nbf):
				for igp in range(self.ngp_linear):
					kMat_gp[i][j].append(torch.nn.functional.pad((self.dN_x[igp,i] * self.dN_x[igp,j] + self.dN_y[igp,i] * self.dN_y[igp,j]) * nu_gp[igp], p2d, "constant", 0))
				kMat_gp[i][j] = torch.stack(kMat_gp[i][j], dim=0)
			kMat_gp[i] = torch.stack(kMat_gp[i], dim=0)
		kMat_gp = torch.stack(kMat_gp, dim=0)
		
		# cMat = [[torch.sum(*(kMat_gp[i][j])) for _ in range(self.nbf)] for _ in range(self.nbf)]
		
		cMat_2 = torch.sum(kMat_gp, dim=0)

		u_2 = torch.nn.functional.pad(g, p2d, "constant", 0)

		# print("cmat shape = ", cMat.shape)

		end = time.time()
		# print("Time (Elemental integration values): ", end - start)


		# for i in range(4):
		#     for j in range(4):
		#         self.cMat[i,j,:,:,:,:] = nn.functional.conv2d(self.kMat[i,j,:,:,:,:], self.strider, stride=2)

		lhfinal_2 = torch.zeros((args.batch_size, 1, ny, nx)).to(device)

		start = time.time()
		for j in range(ny):
			for i in range(nx):

				iel = i
				jel = j
				# print(i,",",j,",",iel,",",jel)
				lhfinal_2[:,:,j,i] += cMat_2[3, 0, :, :, jel, iel] * u_2[:, :, j, i]
				lhfinal_2[:,:,j,i] += cMat_2[3, 1, :, :, jel, iel] * u_2[:, :, j, i+1]
				lhfinal_2[:,:,j,i] += cMat_2[3, 2, :, :, jel, iel] * u_2[:, :, j+1, i]
				lhfinal_2[:,:,j,i] += cMat_2[3, 3, :, :, jel, iel] * u_2[:, :, j+1, i+1]
					
				iel = i+1
				jel = j
				# print(i,",",j,",",iel,",",jel)
				lhfinal_2[:,:,j,i] += cMat_2[2, 0, :, :, jel, iel] * u_2[:, :, j, i+1]
				lhfinal_2[:,:,j,i] += cMat_2[2, 1, :, :, jel, iel] * u_2[:, :, j, i+2]
				lhfinal_2[:,:,j,i] += cMat_2[2, 2, :, :, jel, iel] * u_2[:, :, j+1, i+1]
				lhfinal_2[:,:,j,i] += cMat_2[2, 3, :, :, jel, iel] * u_2[:, :, j+1, i+2]
					
				iel = i
				jel = j+1
				# print(i,",",j,",",iel,",",jel)
				lhfinal_2[:,:,j,i] += cMat_2[1, 0, :, :, jel, iel] * u_2[:, :, j+1, i]
				lhfinal_2[:,:,j,i] += cMat_2[1, 1, :, :, jel, iel] * u_2[:, :, j+1, i+1]
				lhfinal_2[:,:,j,i] += cMat_2[1, 2, :, :, jel, iel] * u_2[:, :, j+2, i]
				lhfinal_2[:,:,j,i] += cMat_2[1, 3, :, :, jel, iel] * u_2[:, :, j+2, i+1]
					
				iel = i+1
				jel = j+1
				# print(i,",",j,",",iel,",",jel)
				lhfinal_2[:,:,j,i] += cMat_2[0, 0, :, :, jel, iel] * u_2[:, :, j+1, i+1]
				lhfinal_2[:,:,j,i] += cMat_2[0, 1, :, :, jel, iel] * u_2[:, :, j+1, i+2]
				lhfinal_2[:,:,j,i] += cMat_2[0, 2, :, :, jel, iel] * u_2[:, :, j+2, i+1]
				lhfinal_2[:,:,j,i] += cMat_2[0, 3, :, :, jel, iel] * u_2[:, :, j+2, i+2]

		end = time.time()
		# print("Time (Assembly): ", end - start)

		res_flat = lhfinal_2.view(g.shape[0], -1)
		res_norm_1 = torch.norm(res_flat, p=1, dim=1) #/(output_dim*output_dim)
		res_norm_2 = torch.norm(res_flat, p=2, dim=1) #/(output_dim*output_dim)

		c1 = self.c1
		c2 = self.c2

		res_norm = c1 * res_norm_1 + c2 * res_norm_2

		return res_norm

	def loss_2d_fem_SS_0(self, g, coeff_mini_batch, forcing):
		print("In FEM loss")
		nx = self.nx
		ny = self.ny
		nelem = self.nelem
		# print("g (pde) = ", g.type(), ", size = ", g.size())
		# print("g.shape = ", g.shape)
		all_coeff_tensors = get_fields_of_coefficients(self.args, coeff_mini_batch, self.gpline_x.shape[0])
		nu_4d_tensor_gp = create_nu_tensor(self.args, all_coeff_tensors, self.gpX_2d, self.gpY_2d)
		
		for i in range(self.nbf):
			for j in range(self.nbf):
				self.kMat[i,j,:,:,:,:] = (self.dN[i,0,:,:] * self.dN[j,0,:,:] + self.dN[i,1,:,:] * self.dN[j,1,:,:]) * nu_4d_tensor_gp
		
		for i in range(4):
			for j in range(4):
				self.cMat[i,j,:,:,:,:] = nn.functional.conv2d(self.kMat[i,j,:,:,:,:], self.strider, stride=2)

		for j in range(ny):
			for i in range(nx):
				k1 = k2 = k3 = k4 = -1
				if i > 0 and j > 0:
					k1 = nelem * (j-1) + (i-1)
				if j > 0 and i < (nx - 1):
					k2 = nelem * (j-1) + (i)
				if i > 0 and j < (ny - 1):
					k3 = nelem * (j) + (i-1)
				if (i < (nx - 1) and j < (ny - 1)):
					k4 = nelem * (j) + (i)
					
				if k1 >= 0:
					iel = k1 % nelem
					jel = k1 // nelem            
		#             print(i,",",j,",",iel,",",jel)
					self.lhfinal[:,:,j,i] += self.cMat[3, 0, :, :, jel, iel] * g[:, :, j-1, i-1]
					self.lhfinal[:,:,j,i] += self.cMat[3, 1, :, :, jel, iel] * g[:, :, j-1, i]
					self.lhfinal[:,:,j,i] += self.cMat[3, 2, :, :, jel, iel] * g[:, :, j, i-1]
					self.lhfinal[:,:,j,i] += self.cMat[3, 3, :, :, jel, iel] * g[:, :, j, i]
					
				if k2 >= 0:
					iel = k2 % nelem
					jel = k2 // nelem            
		#             print(i,",",j,",",iel,",",jel)
					self.lhfinal[:,:,j,i] += self.cMat[2, 0, :, :, jel, iel] * g[:, :, j-1, i]
					self.lhfinal[:,:,j,i] += self.cMat[2, 1, :, :, jel, iel] * g[:, :, j-1, i+1]
					self.lhfinal[:,:,j,i] += self.cMat[2, 2, :, :, jel, iel] * g[:, :, j, i]
					self.lhfinal[:,:,j,i] += self.cMat[2, 3, :, :, jel, iel] * g[:, :, j, i+1]
					
				if k3 >= 0:
					iel = k3 % nelem
					jel = k3 // nelem            
		#             print(i,",",j,",",iel,",",jel)
					self.lhfinal[:,:,j,i] += self.cMat[1, 0, :, :, jel, iel] * g[:, :, j, i-1]
					self.lhfinal[:,:,j,i] += self.cMat[1, 1, :, :, jel, iel] * g[:, :, j, i]
					self.lhfinal[:,:,j,i] += self.cMat[1, 2, :, :, jel, iel] * g[:, :, j+1, i-1]
					self.lhfinal[:,:,j,i] += self.cMat[1, 3, :, :, jel, iel] * g[:, :, j+1, i]
					
				if k4 >= 0:
					iel = k4 % nelem
					jel = k4 // nelem
		#             print(i,",",j,",",iel,",",jel)
					self.lhfinal[:,:,j,i] += self.cMat[0, 0, :, :, jel, iel] * g[:, :, j, i]
					self.lhfinal[:,:,j,i] += self.cMat[0, 1, :, :, jel, iel] * g[:, :, j, i+1]
					self.lhfinal[:,:,j,i] += self.cMat[0, 2, :, :, jel, iel] * g[:, :, j+1, i]
					self.lhfinal[:,:,j,i] += self.cMat[0, 3, :, :, jel, iel] * g[:, :, j+1, i+1]

		res_flat = self.lhfinal.view(g.shape[0], -1)
		res_norm_1 = torch.norm(res_flat, p=1, dim=1) #/(output_dim*output_dim)
		res_norm_2 = torch.norm(res_flat, p=2, dim=1) #/(output_dim*output_dim)

		c1 = self.c1
		c2 = self.c2

		res_norm = c1 * res_norm_1 + c2 * res_norm_2

		return res_norm

	def boundary_loss_2d(self, g, list_of_bc, output_dim):     
		# print("g (bc) = ", g.type(), ", size = ", g.size())
		# print("sobelx = ", sobelx.type(), ", size = ", sobelx.size())
		# print("h_corr = ", h_corr.type(), ", size = ", h_corr.size())
		
		g_x = self.derivative_x(self.pad(g))
		g_y = self.derivative_y(self.pad(g))

		x_b_right_neumann = g_x[:,:,:,-1]
		x_b_bottom_neumann = g_x[:,:,0,:]
		x_b_top_neumann = g_x[:,:,-1,:]

		# x_b_x1 = g[:,:,:,-2]
		# x_b_y0 = g[:,:,2,:]
		# x_b_y1 = g[:,:,-2,:]
		# x_b = x_b_y0 + x_b_y1    
		# t_0 = g[:,:,:,2] - t_init.unsqueeze(1)

		x_b_x0 = g[:,:,:,0] - list_of_bc[0] #- t_init.unsqueeze(1)
		x_b_x1 = g[:,:,:,-1] - list_of_bc[1]
		x_b_y0 = g[:,:,0,:] - list_of_bc[2]
		x_b_y1 = g[:,:,-1,:] - list_of_bc[3]

		# ggtemp_0 = list_of_bc[0]
		# ggtemp_1 = list_of_bc[3]
		# print("ggtemp_0 = ", ggtemp_0.type(), ", size = ", ggtemp_0.shape)
		# print("ggtemp_1 = ", ggtemp_1.type(), ", size = ", ggtemp_1.shape)
		
		# return (dirichlet + neumann_right)
		dirichlet = (torch.norm(x_b_x0, dim=2) + torch.norm(x_b_x1, dim=2))

		neumann_coef_1 = args.ltr_c1 * args.ltr_neumann_cf
		neumann_coef_2 = args.ltr_c2 * args.ltr_neumann_cf
		neumann_norm_1 = torch.norm(x_b_bottom_neumann, p=1, dim=2) + torch.norm(x_b_top_neumann, p=1, dim=2)
		neumann_norm_2 = torch.norm(x_b_bottom_neumann, p=2, dim=2) + torch.norm(x_b_top_neumann, p=2, dim=2)
		
		b_loss = dirichlet + (neumann_coef_1 * neumann_norm_1 + neumann_coef_2 * neumann_norm_2)
		return b_loss
		# return (dirichlet)

	def boundary_loss_3d(self, g, list_of_bc, output_dim):     
			# print("g (bc) = ", g.type(), ", size = ", g.size())
			# print("sobelx = ", sobelx.type(), ", size = ", sobelx.size())
			# print("h_corr = ", h_corr.type(), ", size = ", h_corr.size())
			g_x = self.derivative_x(self.pad(g))
			g_y = self.derivative_y(self.pad(g))
			g_z = self.derivative_z(self.pad(g))

			x_b_x0 = g[:,:,:,:, 0] - list_of_bc[0]
			x_b_x1 = g[:,:,:,:,-1] - list_of_bc[1]

			dirichlet = (torch.norm(x_b_x0, dim=2) + torch.norm(x_b_x1, dim=2))
			
			b_loss = dirichlet #+ (neumann_coef_1 * neumann_norm_1 + neumann_coef_2 * neumann_norm_2)
			return b_loss
			# return (dirichlet)

	def prepare_inputs(self, coeff_mini_batch):
		if not self.manufactured:
			all_coeff_tensors = get_fields_of_coefficients(self.args, coeff_mini_batch, self.args.output_size)
			# print(all_coeff_tensors)
			# print(all_coeff_tensors.shape)

			forcing_4d_tensor = 0 * self.x
			nu_4d_tensor = create_nu_tensor(self.args, all_coeff_tensors, self.x, self.y, self.z)
			# gen_input = create_generator_input(self.args, nu_4d_tensor)
			# list_of_bc = calc_BC_poisson(args)
			gen_input = nu_4d_tensor
			return gen_input, nu_4d_tensor, forcing_4d_tensor
		else:
			all_coeff_tensors = get_fields_of_coefficients(self.args, coeff_mini_batch, self.args.output_size)
			forcing_4d_tensor = create_forcing_tensor(self.args, all_coeff_tensors, self.x, self.y, self.z)
			nu_4d_tensor = (self.x)**0. # all 1
			gen_input = forcing_4d_tensor
			return gen_input, nu_4d_tensor, forcing_4d_tensor

	def apply_bc(self, g):
		if not self.manufactured:
			if self.nsd == 2:
				if self.fem_basis_deg == 1:
					g = nn.functional.pad(g, (0,0,1,1), 'replicate')
					g = nn.functional.pad(g, (1,0,0,0), 'constant', value=1)
					g = nn.functional.pad(g, (0,1,0,0), 'constant', value=0)
				elif self.fem_basis_deg == 2:
					g = nn.functional.pad(g, (0,0,1,1), 'replicate')
					g = nn.functional.pad(g, (1,0,0,0), 'constant', value=1)
					g = nn.functional.pad(g, (0,1,0,1), 'replicate')
					g = nn.functional.pad(g, (0,1,0,0), 'constant', value=0)                    
				elif self.fem_basis_deg == 3:
					g = nn.functional.pad(g, (1,1,1,1), 'replicate')
					g = nn.functional.pad(g, (0,0,1,1), 'replicate')
					g = nn.functional.pad(g, (0,1,0,0), 'constant', value=0)
					g = nn.functional.pad(g, (1,0,0,0), 'constant', value=1)
			elif self.nsd == 3:
				if self.fem_basis_deg == 1:
					g = nn.functional.pad(g, (0,0,1,1,1,1), 'replicate')
					g = nn.functional.pad(g, (0,1,0,0,0,0), 'constant', value=0)
					g = nn.functional.pad(g, (1,0,0,0,0,0), 'constant', value=1)
			return g
		else:
			if self.fem_basis_deg == 1:
				g = nn.functional.pad(g, (1,1,1,1), 'constant', value=0)
			elif self.fem_basis_deg == 2:
				g = nn.functional.pad(g, (1,0,1,0), 'constant', value=0)
				g = nn.functional.pad(g, (0,1,0,1), 'replicate')
				g = nn.functional.pad(g, (0,1,0,1), 'constant', value=0)            
			elif self.fem_basis_deg == 3:
				g = nn.functional.pad(g, (1,1,1,1), 'replicate')
				g = nn.functional.pad(g, (1,1,1,1), 'replicate', value=0)
			return g             

	def apply_padding_on(self, field):
		if self.nsd == 2:
			if self.fem_basis_deg == 1:
				field = nn.functional.pad(field, (1, 1, 1, 1), 'replicate')
			elif self.fem_basis_deg == 2:
				field = nn.functional.pad(field, (1,1,1,1), 'replicate')
				field = nn.functional.pad(field, (0,1,0,1), 'replicate') 
			elif self.fem_basis_deg == 3:
				field = nn.functional.pad(field, (2, 2, 2, 2), 'replicate')
			return field
		elif self.nsd == 3:
			if self.fem_basis_deg == 1:
				p3d = (1, 1, 1, 1, 1, 1)
				field = nn.functional.pad(field, p3d, 'replicate')
			return field
					


	def calc_exact_solution(self, coefficients):
		if self.manufactured:
			x, y, z = create_grid(self.args)        
			all_coeff_tensors = get_fields_of_coefficients(self.args, coefficients, self.args.output_size)

			a1 = all_coeff_tensors[0]
			a2 = all_coeff_tensors[1]
			a3 = all_coeff_tensors[2]
			a4 = all_coeff_tensors[3]
			a5 = all_coeff_tensors[4]
			a6 = all_coeff_tensors[5]

			pi = np.pi
			sin = torch.sin
			cos = torch.cos
			exp = torch.exp

			# u_exact_tensor = a1*(cos(2*pi*x)/2 - 1/2)*(cos(2*pi*y)/2 - 1/2) + a2*(cos(4*pi*x)/2 - 1/2)*(cos(4*pi*y)/2 - 1/2) + a3*(cos(6*pi*x)/2 - 1/2)*(cos(6*pi*y)/2 - 1/2) + a4*(cos(8*pi*x)/2 - 1/2)*(cos(8*pi*y)/2 - 1/2) + a5*(cos(10*pi*x)/2 - 1/2)*(cos(10*pi*y)/2 - 1/2) + a6*(cos(12*pi*x)/2 - 1/2)*(cos(12*pi*y)/2 - 1/2)
			# u_exact_tensor = exp(a1*(cos(2*pi*x)/2 - 1/2)*(cos(2*pi*y)/2 - 1/2) + a2*(cos(4*pi*x)/2 - 1/2)*(cos(4*pi*y)/2 - 1/2) + a3*(cos(6*pi*x)/2 - 1/2)*(cos(6*pi*y)/2 - 1/2) + a4*(cos(8*pi*x)/2 - 1/2)*(cos(8*pi*y)/2 - 1/2) + a5*(cos(10*pi*x)/2 - 1/2)*(cos(10*pi*y)/2 - 1/2) + a6*(cos(12*pi*x)/2 - 1/2)*(cos(12*pi*y)/2 - 1/2))
			u_exact_tensor = sin(pi * x) * sin(pi * y)

			if self.args.fem_basis_deg == 1:
				u_exact_tensor = nn.functional.pad(u_exact_tensor, (1,1,1,1), 'replicate')    
			elif self.args.fem_basis_deg == 2:
				u_exact_tensor = nn.functional.pad(u_exact_tensor, (1,1,1,1), 'replicate')            
				u_exact_tensor = nn.functional.pad(u_exact_tensor, (0,1,0,1), 'replicate')            
			elif self.args.fem_basis_deg == 3:
				u_exact_tensor = nn.functional.pad(u_exact_tensor, (2,2,2,2), 'replicate')        
			return u_exact_tensor
		else:
			raise NotImplementedError()


class NonlinearPoisson(FEM,FDM):
	def __init__(self, args):
		super().__init__(args)
		if args.loss_calc_type == 'fem':
			FEM.__init__(args)
		else:
			FDM.__init__(args)

	def loss(self, g, nu, forcing):
		if self.loss_calc_type == 'fem':
			if self.nsd == 2:
				return self.loss_2d_fem(g, nu, forcing)
		elif self.loss_calc_type == 'fdm':
			raise NotImplementedError()
			if self.nsd == 2:
				return self.loss_2d(g, nu, forcing)
			elif self.nsd == 3:
				return self.loss_3d(g, nu, forcing)

	def boundary_loss(self, g, list_of_bc, output_dim):
		raise NotImplementedError()
		if self.nsd == 2:
			return self.boundary_loss_2d(g, list_of_bc, output_dim)
		elif self.nsd == 3:
			return self.boundary_loss_3d(g, list_of_bc, output_dim)

	def loss_2d_fem(self, g, nu, forcing):
		# print("In FEM loss")
		nx = self.nx
		ny = self.ny
		nelem = self.nelem
		# print("g (pde) = ", g.type(), ", size = ", g.size())
		# print("g.shape = ", g.shape)

		start = time.time()
		
		if args.boundary_opt == 0:
			p2d = (1, 1, 1, 1)
			nu = nn.functional.pad(nu, p2d, 'replicate')
			g = nn.functional.pad(g, (0,0,1,1), 'replicate')
			g = nn.functional.pad(g, (0,1,0,0), 'constant', value=0)
			g = nn.functional.pad(g, (1,0,0,0), 'constant', value=1)

		# print("g = ", g)
		# exit()

		# print("size of nu = ", nu.shape)
		# print("size of g = ", g.shape)

		# nu_gp = torch.zeros((self.ngp_linear, args.batch_size, 1, nelem, nelem)).to(device)
		g_gp_0 = nn.functional.conv2d(g, self.N_gp_0, stride=1)
		g_gp_1 = nn.functional.conv2d(g, self.N_gp_1, stride=1)
		g_gp_2 = nn.functional.conv2d(g, self.N_gp_2, stride=1)
		g_gp_3 = nn.functional.conv2d(g, self.N_gp_3, stride=1)

		nu_gp_0 = nn.functional.conv2d(nu, self.N_gp_0, stride=1)
		nu_gp_1 = nn.functional.conv2d(nu, self.N_gp_1, stride=1)
		nu_gp_2 = nn.functional.conv2d(nu, self.N_gp_2, stride=1)
		nu_gp_3 = nn.functional.conv2d(nu, self.N_gp_3, stride=1)

		g_x_gp_0 = nn.functional.conv2d(g, self.dN_x_gp_0, stride=1)
		g_x_gp_1 = nn.functional.conv2d(g, self.dN_x_gp_1, stride=1)
		g_x_gp_2 = nn.functional.conv2d(g, self.dN_x_gp_2, stride=1)
		g_x_gp_3 = nn.functional.conv2d(g, self.dN_x_gp_3, stride=1)

		g_y_gp_0 = nn.functional.conv2d(g, self.dN_y_gp_0, stride=1)
		g_y_gp_1 = nn.functional.conv2d(g, self.dN_y_gp_1, stride=1)
		g_y_gp_2 = nn.functional.conv2d(g, self.dN_y_gp_2, stride=1)
		g_y_gp_3 = nn.functional.conv2d(g, self.dN_y_gp_3, stride=1)

		# nu_x_gp_0 = nn.functional.conv2d(nu, self.dN_x_gp_0, stride=1)
		# nu_x_gp_1 = nn.functional.conv2d(nu, self.dN_x_gp_1, stride=1)
		# nu_x_gp_2 = nn.functional.conv2d(nu, self.dN_x_gp_2, stride=1)
		# nu_x_gp_3 = nn.functional.conv2d(nu, self.dN_x_gp_3, stride=1)
		# nu_y_gp_0 = nn.functional.conv2d(nu, self.dN_y_gp_0, stride=1)
		# nu_y_gp_1 = nn.functional.conv2d(nu, self.dN_y_gp_1, stride=1)
		# nu_y_gp_2 = nn.functional.conv2d(nu, self.dN_y_gp_2, stride=1)
		# nu_y_gp_3 = nn.functional.conv2d(nu, self.dN_y_gp_3, stride=1)


		# int_g_x_elmwise = g_x_gp_0**2 + g_x_gp_1**2 + g_x_gp_2**2 + g_x_gp_3**2
		# int_g_y_elmwise = g_y_gp_0**2 + g_y_gp_1**2 + g_y_gp_2**2 + g_y_gp_3**2

		# integral_nu_elmwise = (0.25 * self.h ** 2) *  (self.gpw[0] * nu_gp_0 \
		#                                              + self.gpw[1] * nu_gp_1 \
		#                                              + self.gpw[2] * nu_gp_2 \
		#                                              + self.gpw[3] * nu_gp_3)

		# integral_nu_batch = torch.sum(integral_nu_elmwise.view(g.shape[0], -1), 1)
		# print("Integral nu = ", integral_nu_batch)

		# integral_dnu_elmwise = (0.25 * self.h ** 2) * (self.gpw[0] * (nu_x_gp_0**2 + nu_y_gp_0**2) \
		#                             + self.gpw[1] * (nu_x_gp_1**2 + nu_y_gp_1**2) \
		#                             + self.gpw[2] * (nu_x_gp_2**2 + nu_y_gp_2**2) \
		#                             + self.gpw[3] * (nu_x_gp_3**2 + nu_y_gp_3**2))

		# integral_dnu_batch = torch.sum(integral_dnu_elmwise.view(g.shape[0], -1), 1)
		# print("Integral dnu_mag = ", integral_dnu_batch)
		# exit()



		res_elmwise = (0.25 * self.h ** 2) * (self.gpw[0] * nu_gp_0 * (1. + g_gp_0**2) * (g_x_gp_0**2 + g_y_gp_0**2) \
											+ self.gpw[1] * nu_gp_1 * (1. + g_gp_1**2) * (g_x_gp_1**2 + g_y_gp_1**2) \
											+ self.gpw[2] * nu_gp_2 * (1. + g_gp_2**2) * (g_x_gp_2**2 + g_y_gp_2**2) \
											+ self.gpw[3] * nu_gp_3 * (1. + g_gp_3**2) * (g_x_gp_3**2 + g_y_gp_3**2))

		# print("size of res = ", res_elmwise.shape)
		# exit()

		res_flat = res_elmwise.view(g.shape[0], -1)
		loss = torch.sum(res_flat, 1)
		# print(loss)        
		
		return loss
