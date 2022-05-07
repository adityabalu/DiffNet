import torch
import numpy as np
from torch import nn
from .base import PDE
from .cuboid_mesh import CuboidMesh

def gauss_pt_eval(tensor, N, nsd=2, stride=1):
    if nsd == 1:
        conv_gp = nn.functional.conv1d
    elif nsd == 2:
        conv_gp = nn.functional.conv2d
    elif nsd == 3:
        conv_gp = nn.functional.conv3d

    result_list = []
    for i in range(len(N)):
        result_list.append(conv_gp(tensor, N[i], stride=stride))
    return torch.cat(result_list, 1)


class DiffNetFEM(PDE):
    """docstring for DiffNetFEM"""
    def __init__(self, network, dataset, **kwargs):
        super(DiffNetFEM, self).__init__(network, dataset, **kwargs)
        self.ngp_1d = kwargs.get('ngp_1d', 2)
        self.fem_basis_deg = kwargs.get('fem_basis_deg', 1)

        # Gauss quadrature setup
        if self.fem_basis_deg == 1:
            ngp_1d = 2
        elif self.fem_basis_deg == 2:
            ngp_1d = 3
        elif self.fem_basis_deg == 3:
            ngp_1d = 3

        if ngp_1d > self.ngp_1d:
            self.ngp_1d = ngp_1d
        
        self.ngp_total = ngp_total = self.ngp_1d**self.nsd
        self.gpx_1d, self.gpw_1d = self.gauss_guadrature_scheme(self.ngp_1d)

        self.nelemX = nelemX = int((self.domain_sizeX - 1)/self.fem_basis_deg)
        self.nelemY = nelemY = int((self.domain_sizeY - 1)/self.fem_basis_deg)
        if self.nsd == 3:
            self.nelemZ = nelemZ = int((self.domain_sizeZ - 1)/self.fem_basis_deg)
        self.nelem = nelem = int((self.domain_size - 1)/self.fem_basis_deg) # for backward compatibility (uses the X dir value)
        self.hx = self.domain_lengthX / self.nelemX
        self.hy = self.domain_lengthY / self.nelemY
        if self.nsd == 3:
            self.hz = self.domain_lengthZ / self.nelemZ
        self.h = self.domain_length / self.nelem # for backward compatibility (uses the X dir value)

        # Basis functions setup
        if self.fem_basis_deg == 1:
            self.nbf_1d = nbf_1d = 2
            self.nbf_total = nbf_total = self.nbf_1d**self.nsd
            
            self.bf_1d = lambda x: np.array([0.5*(1.-x), 0.5*(1.+x)])
            self.bf_1d_der = lambda x: np.array([0.5*(0.-1.), 0.5*(0.+1.)])
            self.bf_1d_der2 = lambda x: np.array([0.0, 0.0])

        elif self.fem_basis_deg == 2:
            assert (self.domain_size- 1)%2 == 0
            self.nbf_1d = nbf_1d = 3
            self.nbf_total = nbf_total = self.nbf_1d**self.nsd

            self.bf_1d = lambda x: np.array([
                                        0.5 * x * (x-1.),
                                        (1. - x**2),
                                        0.5 * x * (x+1.)
                                        ], dtype=np.float)
            self.bf_1d_der = lambda x: np.array([
                                        0.5 * (2.*x-1.),
                                        (- 2.*x),
                                        0.5 * (2.*x+1.)
                                        ], dtype=np.float)
            self.bf_1d_der2 = lambda x: np.array([
                                        0.5 * (2.),
                                        (- 2.),
                                        0.5 * (2.)
                                        ], dtype=np.float)
        
        elif self.fem_basis_deg == 3:
            assert (self.domain_size- 1)%3 == 0
            self.nbf_1d = nbf_1d = 4
            self.nbf_total = nbf_total = self.nbf_1d**self.nsd

            self.bf_1d = lambda x: np.array([
                                        (-9. / 16.) * (x**3- x**2 - (1. / 9.) * x + (1. / 9.)),
                                        (27. / 16.) * (x**3 - (1. / 3.) * x**2 - x + (1. / 3.)),
                                        (-27. / 16.) * (x**3 + (1. / 3.) * x**2 - x - (1. / 3.)),
                                        (9. / 16.) * (x**3 + x**2 - (1. / 9.) * x - (1. / 9.))
                                        ], dtype=np.float)
            self.bf_1d_der = lambda x: np.array([
                                        (-9. / 16.) * (3 * x**2 - 2 * x - (1. / 9.)),
                                        (27. / 16.) * (3 * x**2 - (2. / 3.) * x - 1),
                                        (-27. / 16.) * (3 * x**2 + (2. / 3.) * x - 1),
                                        (9. / 16.) * (3 * x**2 + 2 * x - (1. / 9.))
                                        ], dtype=np.float)

            self.bf_1d_der2 = lambda x: np.array([
                                        (-9. / 16.) * (6. * x - 2.),
                                        (27. / 16.) * (6. * x - (2. / 3.)),
                                        (-27. / 16.) * (6. * x + (2. / 3.)),
                                        (9. / 16.) * (6. * x + 2.)
                                        ], dtype=np.float)

    def gauss_guadrature_scheme(self, ngp_1d):
        if ngp_1d == 1:
            gpx_1d = np.array([0.])
            gpw_1d = np.array([2.])
        elif ngp_1d == 2:
            gpx_1d = np.array([-0.5773502691896258, 0.5773502691896258])
            gpw_1d = gpw = np.array([1., 1.])
        elif ngp_1d == 3:
            gpx_1d = np.array([-0.774596669, 0., +0.774596669])
            gpw_1d = np.array([5./9., 8./9., 5./9.])
        elif ngp_1d == 4:
            gpx_1d = np.array([-0.861136, -0.339981, +0.339981, +0.861136])
            gpw_1d = np.array([0.347855, 0.652145, 0.652145, 0.347855])
        return gpx_1d, gpw_1d

    def gauss_pt_evaluation(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.N_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_surf(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.N_gp_surf, nsd=(self.nsd-1), stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der_x(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.dN_x_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der_y(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.dN_y_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der_z(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.dN_z_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_x(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_x_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_y(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_y_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_z(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_z_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_xy(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_xy_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_yz(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_yz_gp, nsd=self.nsd, stride=(self.nbf_1d-1))

    def gauss_pt_evaluation_der2_zx(self, tensor, stride=1):
        return gauss_pt_eval(tensor, self.d2N_zx_gp, nsd=self.nsd, stride=(self.nbf_1d-1))
   


class DiffNet2DFEM(DiffNetFEM):
    """docstring for DiffNet2DFEM"""
    def __init__(self, network, dataset, **kwargs):
        super(DiffNet2DFEM, self).__init__(network, dataset, **kwargs)
        assert self.nsd==2
        self.gpw = torch.zeros(self.ngp_total)
        self.N_gp = nn.ParameterList() 
        self.dN_x_gp = nn.ParameterList()
        self.dN_y_gp = nn.ParameterList()
        self.d2N_x_gp = nn.ParameterList()
        self.d2N_y_gp = nn.ParameterList() 
        self.d2N_xy_gp = nn.ParameterList()
        self.Nvalues = torch.ones((1,self.nbf_total,self.ngp_total,1,1))
        self.dN_x_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1))
        self.dN_y_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1))
        self.d2N_x_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1))
        self.d2N_y_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1))
        self.d2N_xy_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1))
        for jgp in range(self.ngp_1d):
            for igp in range(self.ngp_1d):
                N_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                dN_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                dN_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                d2N_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                d2N_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                d2N_xy_gp = torch.zeros((self.nbf_1d, self.nbf_1d))

                IGP = self.ngp_1d * jgp + igp # tensor product id or the linear id of the gauss point
                self.gpw[IGP] = self.gpw_1d[igp] * self.gpw_1d[jgp]
                for jbf in range(self.nbf_1d):
                    for ibf in range(self.nbf_1d):
                        IBF = self.nbf_1d * jbf + ibf
                        N_gp[jbf,ibf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf]
                        dN_x_gp[jbf,ibf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * (2 / self.hx)
                        dN_y_gp[jbf,ibf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * (2 / self.hy)
                        d2N_x_gp[jbf,ibf] = self.bf_1d_der2(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * (2 / self.hx)**2
                        d2N_y_gp[jbf,ibf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der2(self.gpx_1d[jgp])[jbf] * (2 / self.hy)**2
                        d2N_xy_gp[jbf,ibf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * (2 / self.hx) * (2 / self.hy)
                        self.Nvalues[0,IBF,IGP,:,:] = N_gp[jbf,ibf]
                        self.dN_x_values[0,IBF,IGP,:,:] = dN_x_gp[jbf,ibf]
                        self.dN_y_values[0,IBF,IGP,:,:] = dN_y_gp[jbf,ibf]
                        self.d2N_x_values[0,IBF,IGP,:,:] = d2N_x_gp[jbf,ibf]
                        self.d2N_y_values[0,IBF,IGP,:,:] = d2N_y_gp[jbf,ibf]
                        self.d2N_xy_values[0,IBF,IGP,:,:] = d2N_xy_gp[jbf,ibf]
                self.N_gp.append(nn.Parameter(N_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.dN_x_gp.append(nn.Parameter(dN_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.dN_y_gp.append(nn.Parameter(dN_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.d2N_x_gp.append(nn.Parameter(d2N_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.d2N_y_gp.append(nn.Parameter(d2N_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                self.d2N_xy_gp.append(nn.Parameter(d2N_xy_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))

        x = np.linspace(0,self.domain_lengthX,self.domain_sizeX)
        y = np.linspace(0,self.domain_lengthY,self.domain_sizeY)
        xx, yy = np.meshgrid(x,y)
        self.xx = torch.FloatTensor(xx)
        self.yy = torch.FloatTensor(yy)
        self.xgp = self.gauss_pt_evaluation(self.xx.unsqueeze(0).unsqueeze(0))
        self.ygp = self.gauss_pt_evaluation(self.yy.unsqueeze(0).unsqueeze(0))
        self.xiigp = torch.ones_like(self.xgp)
        self.etagp = torch.ones_like(self.ygp)
        for jgp in range(self.ngp_1d):
            for igp in range(self.ngp_1d):
                IGP = self.ngp_1d * jgp + igp # tensor product id or the linear id of the gauss point
                self.xiigp[0,IGP,:,:] = torch.ones_like(self.xiigp[0,IGP,:,:])*self.gpx_1d[igp]
                self.etagp[0,IGP,:,:] = torch.ones_like(self.etagp[0,IGP,:,:])*self.gpx_1d[jgp]

        # SURFACE BASES
        self.gpw_surf = torch.zeros(self.ngp_1d)
        self.N_gp_surf = nn.ParameterList()
        self.dN_x_gp_surf = nn.ParameterList()
        self.dN_y_gp_surf = nn.ParameterList()
        self.Nvalues_surf = torch.ones((1,self.nbf_1d,self.ngp_1d,1))
        self.dN_x_values_surf = torch.ones((1,self.nbf_1d,self.ngp_1d,1))
        self.dN_y_values_surf = torch.ones((1,self.nbf_1d,self.ngp_1d,1))
        for igp in range(self.ngp_1d):
            s_N_gp = torch.zeros((self.nbf_1d))
            s_dN_x_gp = torch.zeros((self.nbf_1d))
            s_dN_y_gp = torch.zeros((self.nbf_1d))

            IGP = igp # tensor product id or the linear id of the gauss point
            self.gpw_surf[IGP] = self.gpw_1d[igp]
            for ibf in range(self.nbf_1d):
                IBF = ibf
                s_N_gp   [ibf] = self.bf_1d(self.gpx_1d[igp])[ibf]
                s_dN_x_gp[ibf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * (2 / self.hx)
                s_dN_y_gp[ibf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * (2 / self.hy)
                self.Nvalues_surf    [0,IBF,IGP,:] = s_N_gp   [ibf]
                self.dN_x_values_surf[0,IBF,IGP,:] = s_dN_x_gp[ibf]
                self.dN_y_values_surf[0,IBF,IGP,:] = s_dN_y_gp[ibf]
            self.N_gp_surf.append(nn.Parameter(s_N_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
            self.dN_x_gp_surf.append(nn.Parameter(s_dN_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
            self.dN_y_gp_surf.append(nn.Parameter(s_dN_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
        # print("N_gp = ", self.N_gp)
        # print("N_gp_surf = ", self.N_gp_surf)
        # for i in range(4):
        #     print("N_gp[{}] = \n".format(i), self.N_gp[i])
        # for i in range(2):
        #     print("N_gp_surf[{}] = \n".format(i), self.N_gp_surf[i])
        # print("Nvalues_surf = \n", self.Nvalues_surf)

        # print("xgp = ", self.xgp)
        # print("ygp = ", self.ygp)
        # print("xiigp = ", self.xiigp)
        # print("etagp = ", self.etagp)
        # # print("self.bf_1d(self.xiigp.squeeze()) = ", self.bf_1d(self.xiigp.squeeze()))
        # print("Nvalues = \n", self.Nvalues)
        # exit()

    def calc_l2_err_old(self, u_sol):
        cn = lambda j,n: [j,j+1,j+n,(j+1)+n]
        N = lambda x,y: (1./4.)*np.array([(1-x)*(1-y), (1+x)*(1-y), (1-x)*(1+y), (1+x)*(1+y)])
        transform = lambda a,b,x: ((a+b)/2. + (b-a)/2.*x)

        ngp = 4
        gpx = np.array([-0.577350269189626, 0.577350269189626, -0.577350269189626, 0.577350269189626])
        gpy = np.array([-0.577350269189626, -0.577350269189626, 0.577350269189626, 0.577350269189626])
        gpw = np.ones(4)

        nnodex = self.domain_size
        nnodey = self.domain_size
        nelmx = self.domain_size - 1
        nelmy = self.domain_size - 1
        hx = (1. / nelmx)
        hy = (1. / nelmy)
        J = (hx/2.)*(hy/2.)
        
        x = np.linspace(0,1,self.domain_size)
        y = np.linspace(0,1,self.domain_size)
        # u_sol = self.u_curr.numpy()

        usolnorm = 0.
        uexnorm = 0.
        l2_err = 0.
        for j in range(nelmy):
            for i in range(nelmx):
                local_u = (u_sol[j:j+2,i:i+2].reshape(4,1)).squeeze()
                # print("local_u = ", local_u)
                for igp in range(ngp):
                    basis = N(gpx[igp], gpy[igp])
                    # print("basis = ", basis)
                    xp = transform(x[i],x[i+1],gpx[igp])
                    yp = transform(y[j],y[j+1],gpy[igp])

                    u1 = np.dot(local_u, basis)
                    u2 = self.exact_solution(xp,yp)

                    # print("(xp,yp,uex,usol) = ", xp,yp,u2,u1)
                    # print("u1 = ", u1)
                    # print("u2 = ", u2)
                    # print("gpw(igp) = ", gpw[igp])
                    # print("J = ", J)
                    l2_err += (u1 - u2)**2 * J
                    usolnorm += u1**2*J
                    uexnorm += u2**2*J

        l2_err = np.sqrt(l2_err)
        usolnorm = np.sqrt(usolnorm)
        uexnorm = np.sqrt(uexnorm)

        # assumes that self.u_exact has been assigned as a torch tensor in the child class
        u_ex = self.u_exact #.squeeze().detach().cpu().numpy()

        print("J = ", J)
        print("usol.shape =", u_sol.shape)
        print("uex.shape =", u_ex.shape)
        print("||u_sol||, ||uex|| = ", usolnorm, uexnorm)
        print("||e||_{{L2}} = ", l2_err)
        # by taking vector norm
        print("||e|| (vector-norm) = ", np.linalg.norm(u_ex - u_sol, 'fro')/nnodex)

    def calc_l2_err(self, u_sol):
        # assumes that self.u_exact has been assigned as a torch tensor in the child class
        u_ex = torch.FloatTensor(self.u_exact)

        u_gp = self.gauss_pt_evaluation(u_sol)
        u_ex_gp = self.exact_solution(self.xgp, self.ygp).type_as(u_sol)
        e_gp = u_gp - u_ex_gp

        gpw = self.gpw.type_as(u_sol)
        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*self.hx)*(0.5*self.hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        e2 = e_gp**2*JxW
        e2_elmwise = torch.sum(e2, 1)
        eL2 = torch.sqrt(torch.sum(e2_elmwise))

        u2 = u_gp**2*JxW
        u2_elmwise = torch.sum(u2, 1)
        uL2 = torch.sqrt(torch.sum(u2_elmwise))

        u_ex2 = u_ex_gp**2*JxW
        u_ex2_elmwise = torch.sum(u_ex2, 1)
        u_exL2 = torch.sqrt(torch.sum(u_ex2_elmwise))

        print("J = ", trnsfrm_jac)
        print("usol.shape =", u_sol.shape)
        print("uex.shape =", u_ex.shape)
        print("||u_sol||, ||uex|| = ", uL2, u_exL2)
        print("||e||_{{L2}} = ", eL2)
        # by taking vector norm
        print("||e|| (vector-norm) = ", torch.norm(u_ex - u_sol, 'fro')/np.sqrt(self.domain_sizeX*self.domain_sizeY))


class DiffNet3DFEM(DiffNetFEM):
    """docstring for DiffNet2DFEM"""
    def __init__(self, network, dataset, **kwargs):
        super(DiffNet3DFEM, self).__init__(network, dataset, **kwargs)
        assert self.nsd==3
        self.gpw = torch.zeros(self.ngp_total)
        self.N_gp = nn.ParameterList() 
        self.dN_x_gp = nn.ParameterList()
        self.dN_y_gp = nn.ParameterList()
        self.dN_z_gp = nn.ParameterList()
        self.d2N_x_gp = nn.ParameterList()
        self.d2N_y_gp = nn.ParameterList() 
        self.d2N_z_gp = nn.ParameterList() 
        self.d2N_xy_gp = nn.ParameterList()
        self.d2N_yz_gp = nn.ParameterList()
        self.d2N_zx_gp = nn.ParameterList()
        self.Nvalues = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.dN_x_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.dN_y_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.dN_z_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.d2N_x_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.d2N_y_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        self.d2N_z_values = torch.ones((1,self.nbf_total,self.ngp_total,1,1,1))
        for kgp in range(self.ngp_1d):
            for jgp in range(self.ngp_1d):
                for igp in range(self.ngp_1d):
                    N_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    dN_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    dN_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    dN_z_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_z_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_xy_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_yz_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))
                    d2N_zx_gp = torch.zeros((self.nbf_1d, self.nbf_1d, self.nbf_1d))

                    IGP = kgp * self.ngp_1d**2 + jgp * self.ngp_1d + igp # tensor product id or the linear id of the gauss point
                    self.gpw[IGP] = self.gpw_1d[igp] * self.gpw_1d[jgp] * self.gpw_1d[kgp]

                    for kbf in range(self.nbf_1d):
                        for jbf in range(self.nbf_1d):
                            for ibf in range(self.nbf_1d):
                                IBF = kbf * self.nbf_1d**2 + jbf * self.nbf_1d + ibf
                                N_gp[kbf,jbf,ibf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf]
                                dN_x_gp[kbf,jbf,ibf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.hx)
                                dN_y_gp[kbf,jbf,ibf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.hy)
                                dN_z_gp[kbf,jbf,ibf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d_der(self.gpx_1d[kgp])[kbf] * (2 / self.hz)
                                d2N_x_gp[ibf,jbf,kbf] = self.bf_1d_der2(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.hx)**2
                                d2N_y_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der2(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.hy)**2
                                d2N_z_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d_der2(self.gpx_1d[kgp])[kbf] * (2 / self.hz)**2
                                d2N_xy_gp[ibf,jbf,kbf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * self.bf_1d(self.gpx_1d[kgp])[kbf] * (2 / self.hx) * (2 / self.hy)
                                d2N_yz_gp[ibf,jbf,kbf] = self.bf_1d(self.gpx_1d[igp])[ibf] * self.bf_1d_der(self.gpx_1d[jgp])[jbf] * self.bf_1d_der(self.gpx_1d[kgp])[kbf] * (2 / self.hy) * (2 / self.hz)
                                d2N_zx_gp[ibf,jbf,kbf] = self.bf_1d_der(self.gpx_1d[igp])[ibf] * self.bf_1d(self.gpx_1d[jgp])[jbf] * self.bf_1d_der(self.gpx_1d[kgp])[kbf] * (2 / self.hz) * (2 / self.hx)
                                self.Nvalues[0,IBF,IGP,:,:,:] = N_gp[kbf,jbf,ibf]
                                self.dN_x_values[0,IBF,IGP,:,:,:] = dN_x_gp[kbf,jbf,ibf]
                                self.dN_y_values[0,IBF,IGP,:,:,:] = dN_y_gp[kbf,jbf,ibf]
                                self.dN_z_values[0,IBF,IGP,:,:,:] = dN_z_gp[kbf,jbf,ibf]
                                self.d2N_x_values[0,IBF,IGP,:,:,:] = d2N_x_gp[kbf,jbf,ibf]
                                self.d2N_y_values[0,IBF,IGP,:,:,:] = d2N_y_gp[kbf,jbf,ibf]
                                self.d2N_z_values[0,IBF,IGP,:,:,:] = d2N_z_gp[kbf,jbf,ibf]

                    self.N_gp.append(nn.Parameter(N_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.dN_x_gp.append(nn.Parameter(dN_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.dN_y_gp.append(nn.Parameter(dN_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.dN_z_gp.append(nn.Parameter(dN_z_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_x_gp.append(nn.Parameter(d2N_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_y_gp.append(nn.Parameter(d2N_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_z_gp.append(nn.Parameter(d2N_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_xy_gp.append(nn.Parameter(d2N_xy_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_yz_gp.append(nn.Parameter(d2N_yz_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))
                    self.d2N_zx_gp.append(nn.Parameter(d2N_zx_gp.unsqueeze(0).unsqueeze(1), requires_grad=False))

        x = np.linspace(0,self.domain_lengthX,self.domain_sizeX)
        y = np.linspace(0,self.domain_lengthY,self.domain_sizeY)
        z = np.linspace(0,self.domain_lengthZ,self.domain_sizeZ)
        # xx, yy, zz = np.meshgrid(x,y,z)
        xx, yy, zz = CuboidMesh.meshgrid_3d(x,y,z)
        self.xx = torch.FloatTensor(xx)
        self.yy = torch.FloatTensor(yy)
        self.zz = torch.FloatTensor(zz)
        self.xgp = self.gauss_pt_evaluation(self.xx.unsqueeze(0).unsqueeze(0))
        self.ygp = self.gauss_pt_evaluation(self.yy.unsqueeze(0).unsqueeze(0))
        self.zgp = self.gauss_pt_evaluation(self.zz.unsqueeze(0).unsqueeze(0))

        # print("xgp = ", self.xgp)
        # print("ygp = ", self.ygp)
        # print("xiigp = ", self.xiigp)
        # print("etagp = ", self.etagp)
        # # print("self.bf_1d(self.xiigp.squeeze()) = ", self.bf_1d(self.xiigp.squeeze()))
        # print("Nvalues = \n", self.Nvalues)
        # for i in range(8):
        #     print("N_gp[{}] = \n".format(i), self.N_gp[i])
        # for i in range(8):
        #     print("dN_x_gp[{}] = \n".format(i), self.dN_x_gp[i])
        # for i in range(8):
        #     print("dN_y_gp[{}] = \n".format(i), self.dN_y_gp[i])
        # for i in range(8):
        #     print("dN_z_gp[{}] = \n".format(i), self.dN_z_gp[i])

    def calc_l2_err_old(self, u_sol):
        N = lambda x,y,z: (1./8.)*np.array([
            (1-x)*(1-y)*(1-z),
            (1+x)*(1-y)*(1-z),
            (1-x)*(1+y)*(1-z),
            (1+x)*(1+y)*(1-z),
            (1-x)*(1-y)*(1+z),
            (1+x)*(1-y)*(1+z),
            (1-x)*(1+y)*(1+z),
            (1+x)*(1+y)*(1+z)
            ])
        transform = lambda a,b,x: ((a+b)/2. + (b-a)/2.*x)

        ngp = 8
        gpx_1d = self.gpx_1d
        gpw_1d = self.gpw_1d
        gpx = np.tile(gpx_1d,4)
        gpy = np.tile(np.repeat(gpx_1d,2),2)
        gpz = np.repeat(gpx_1d,4)
        gpw = np.ones(8)

        nnodex = self.domain_size
        nnodey = self.domain_size
        nelmx = self.domain_size - 1
        nelmy = self.domain_size - 1
        nelmz = self.domain_size - 1
        hx = (1. / nelmx)
        hy = (1. / nelmy)
        hz = (1. / nelmz)
        J = (hx/2.)*(hy/2.)*(hz/2.)

        x = np.linspace(0,1,self.domain_size)
        y = np.linspace(0,1,self.domain_size)
        z = np.linspace(0,1,self.domain_size)
        # u_sol = self.u_curr.numpy()

        usolnorm = 0.
        uexnorm = 0.
        l2_err = 0.
        for k in range(nelmz):
            for j in range(nelmy):
                for i in range(nelmx):
                    local_u = (u_sol[k:k+2,j:j+2,i:i+2].reshape(8,1)).squeeze()
                    # print("local_u = ", local_u)
                    for igp in range(ngp):
                        basis = N(gpx[igp], gpy[igp], gpz[igp])
                        # print("basis = ", basis)
                        xp = transform(x[i],x[i+1],gpx[igp])
                        yp = transform(y[j],y[j+1],gpy[igp])
                        zp = transform(z[k],z[k+1],gpz[igp])

                        u1 = np.dot(local_u, basis)
                        u2 = self.exact_solution(xp,yp,zp)

                        # print("(xp,yp,uex,usol) = ", xp,yp,u2,u1)
                        # print("u1 = ", u1)
                        # print("u2 = ", u2)
                        # print("gpw(igp) = ", gpw[igp])
                        # print("J = ", J)
                        l2_err += (u1 - u2)**2 * J
                        usolnorm += u1**2*J
                        uexnorm += u2**2*J

        l2_err = np.sqrt(l2_err)
        usolnorm = np.sqrt(usolnorm)
        uexnorm = np.sqrt(uexnorm)

        # assumes that self.u_exact has been assigned as a torch tensor in the child class
        u_ex = self.u_exact.squeeze().detach().cpu().numpy()

        print("J = ", J)
        print("usol.shape =", u_sol.shape)
        print("uex.shape =", u_ex.shape)
        print("||u_sol||, ||uex|| = ", usolnorm, uexnorm)
        print("||e||_{{L2}} = ", l2_err)
        # by taking vector norm
        print("||e|| (vector-norm) = ", np.linalg.norm(u_ex.reshape(-1,1) - u_sol.reshape(-1,1), 'fro')/(1.*nnodex)**1.5)

    def calc_l2_err(self, u_sol):
        # assumes that self.u_exact has been assigned as a torch tensor in the child class
        u_ex = torch.FloatTensor(self.u_exact)

        u_gp = self.gauss_pt_evaluation(u_sol)
        u_ex_gp = self.exact_solution(self.xgp, self.ygp, self.zgp).type_as(u_sol)
        e_gp = u_gp - u_ex_gp

        gpw = self.gpw.type_as(u_sol)
        # DERIVE NECESSARY VALUES
        trnsfrm_jac = (0.5*self.hx)*(0.5*self.hy)*(0.5*self.hz)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        e2 = e_gp**2*JxW
        e2_elmwise = torch.sum(e2, 1)
        eL2 = torch.sqrt(torch.sum(e2_elmwise))

        u2 = u_gp**2*JxW
        u2_elmwise = torch.sum(u2, 1)
        uL2 = torch.sqrt(torch.sum(u2_elmwise))

        u_ex2 = u_ex_gp**2*JxW
        u_ex2_elmwise = torch.sum(u_ex2, 1)
        u_exL2 = torch.sqrt(torch.sum(u_ex2_elmwise))

        print("J = ", trnsfrm_jac)
        print("usol.shape =", u_sol.shape)
        print("uex.shape =", u_ex.shape)
        print("||u_sol||, ||uex|| = ", uL2, u_exL2)
        print("||e||_{{L2}} = ", eL2)
        # by taking vector norm
        print("||e|| (vector-norm) = ", torch.norm(u_ex - u_sol, 'fro')/np.sqrt(self.domain_sizeX*self.domain_sizeY*self.domain_sizeZ))
