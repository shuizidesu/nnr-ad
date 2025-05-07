import torch
import time
import matplotlib.pyplot as plt
from torch.func import jacrev

import scipy.io as sio

from nonlinear_ode_solver.solver_integrator import NewmarkBetaMethod


'''Config and parameter'''
# System parameter
device = 'cpu'
# Parameters for the newmark-beta method
solve_steps = 1280
# Parameters for the newton iteration
incremental_tolerance = 1e-12
residual_tolerance = 1e-12
max_epoch = 200

'''Define the system equation'''
class Rotor40DOFHertz(torch.nn.Module):
    def __init__(self):
        super(Rotor40DOFHertz, self).__init__()
        matlab_data = sio.loadmat('../data/40DOF_MCJK_Fg.mat')
        self.M = torch.tensor(matlab_data['M'], device=device, dtype=torch.double)
        self.C = torch.tensor(matlab_data['C'], device=device, dtype=torch.double)
        self.K = torch.tensor(matlab_data['K'], device=device, dtype=torch.double)
        self.J = torch.tensor(matlab_data['J'], device=device, dtype=torch.double)
        self.Fg = torch.tensor(matlab_data['Fg'], device=device, dtype=torch.double)

        self.lmda = 1.2

        self.Kb = 1e8
        self.ri_normal = torch.tensor([0.014, 0.022, 0.014], device=device, dtype=torch.double)
        self.ro_normal = torch.tensor([0.029, 0.037, 0.029], device=device, dtype=torch.double)
        self.ri_inter = torch.tensor([0.014], device=device, dtype=torch.double)
        self.ro_inter = torch.tensor([0.029], device=device, dtype=torch.double)

        self.bearX = [0, 14, 12]
        self.bearY = [0+20, 14+20, 12+20]

        self.delta0 = 1e-7
        self.Nb = 16

        self.Md = torch.tensor([3.5157, 7.401, 3.701], device=device, dtype=torch.double)
        self.Delta0 = torch.tensor([5e-6, 5e-6, 1e-5], device=device, dtype=torch.double)

    def initialize_parameters(self, omega1):
        self.omega1 = omega1
        self.C_J = self.C + self.omega1 * self.J

    def force(self, current_time):
        Qt = torch.zeros([40, 1], device=device, dtype=torch.double)

        Qt[2] = torch.sin(self.omega1*current_time) * self.omega1 ** 2 * self.Delta0[0] * self.Md[0]
        Qt[10] = torch.sin(self.omega1*current_time) * self.omega1 ** 2 * self.Delta0[1] * self.Md[1]
        Qt[16] = torch.sin(self.omega1*self.lmda*current_time) * self.omega1 ** 2 * self.Delta0[2] * self.Md[2] * self.lmda ** 2

        Qt[2+20] = torch.cos(self.omega1*current_time) * self.omega1 ** 2 * self.Delta0[0] * self.Md[0]
        Qt[10+20] = torch.cos(self.omega1*current_time) * self.omega1 ** 2 * self.Delta0[1] * self.Md[1]
        Qt[16+20] = torch.cos(self.omega1*self.lmda*current_time) * self.omega1 ** 2 * self.Delta0[2] * self.Md[2] * self.lmda ** 2

        Qt = Qt - self.Fg

        return Qt

    def nonlinearity(self, x, current_time):
        nonlinear = torch.zeros([40, 1], device=device, dtype=torch.double)

        for i in range(3):
            omega_c = self.omega1 * self.ri_normal[i] / (self.ri_normal[i] + self.ro_normal[i])

            theta = (2*torch.pi/
                     self.Nb*torch.linspace(0, self.Nb-1, self.Nb, device=device, dtype=torch.double).unsqueeze(1)+
                     omega_c*current_time)

            delta = ((x[self.bearX[i]]).unsqueeze(1) * torch.cos(theta) +
                     (x[self.bearY[i]]).unsqueeze(1) * torch.sin(theta)) - self.delta0

            delta = torch.where(delta < 0, torch.tensor(0.0, device=device, dtype=torch.double), delta)

            FX = (self.Kb * delta ** (10 / 9) * torch.cos(theta)).sum()
            FY = (self.Kb * delta ** (10 / 9) * torch.sin(theta)).sum()

            nonlinear[self.bearX[i]] = -FX
            nonlinear[self.bearY[i]] = -FY


        omega_c = (self.omega1 * self.ri_inter[0] + self.omega1 * self.ro_inter[0] * self.lmda) / (self.ro_inter[0] + self.ri_inter[0])

        theta = (2 * torch.pi /
                 self.Nb * torch.linspace(0, self.Nb - 1, self.Nb, device=device, dtype=torch.double).unsqueeze(1) +
                 omega_c * current_time)

        delta = ((x[8] - x[18]).unsqueeze(1) * torch.cos(theta) +
                 (x[28] - x[38]).unsqueeze(1) * torch.sin(theta)) - self.delta0

        delta = torch.where(delta < 0, torch.tensor(0.0, device=device, dtype=torch.double), delta)

        FX = (self.Kb * delta ** (10 / 9) * torch.cos(theta)).sum()
        FY = (self.Kb * delta ** (10 / 9) * torch.sin(theta)).sum()

        nonlinear[8] = -FX
        nonlinear[18] = FX

        nonlinear[28] = -FY
        nonlinear[38] = FY

        return nonlinear

    def prepare_initial_ddx0(self, x0, dx0, omega1, current_time):
        self.initialize_parameters(omega1)
        Qt = self.force(current_time)
        return  torch.linalg.solve(self.M, Qt + self.nonlinearity(x0, current_time) - self.K @ x0 - self.C_J @ dx0)

    def calculate_residual(self, x, dx, ddx, current_time):
        Qt = self.force(current_time)
        residual_vector = self.M @ ddx + self.C_J @ dx + self.K @ x - self.nonlinearity(x, current_time) - Qt
        return residual_vector


# Give the initial status
my_system = Rotor40DOFHertz()
omega1 = torch.tensor(3760.0, device=device, dtype=torch.double)
x0 = torch.zeros([40, 1], device=device, dtype=torch.double)
dx0 = torch.zeros([40, 1], device=device, dtype=torch.double)
current_time = torch.tensor(0.0, device=device, dtype=torch.double)
ddx0 = my_system.prepare_initial_ddx0(x0, dx0, omega1, current_time)

'''Solve!'''
x = torch.zeros([40, solve_steps+1], device=device, dtype=torch.double)
dx = torch.zeros([40, solve_steps+1], device=device, dtype=torch.double)
ddx = torch.zeros([40, solve_steps+1], device=device, dtype=torch.double)
time_stamp = torch.zeros([solve_steps+1, 1], device=device, dtype=torch.double)
x[:, 0:1] = x0
dx[:, 0:1] = dx0
ddx[:, 0:1] = ddx0
x1 = torch.zeros([40, 1], device=device, dtype=torch.double, requires_grad=True)

delta_t = 2.0 * torch.pi / omega1 / 128.0
numerical_solver =  NewmarkBetaMethod(gamma=0.5, delta=0.25, delta_t=delta_t)
numerical_solver.initialize_system(my_system)
start_time = time.time()
for step in range(1, solve_steps+1):
    current_time += delta_t
    epoch = 0
    incremental = torch.inf
    residual = torch.inf
    print(f'===== Step={step} start =====')
    while not ((epoch>max_epoch) or (incremental<incremental_tolerance) or (residual<residual_tolerance)):
        jacobian = jacrev(numerical_solver.calculate_residual, argnums=0)(x1, x0, dx0, ddx0, current_time).squeeze()

        residual_equation = numerical_solver.calculate_residual(x1, x0, dx0, ddx0, current_time)
        delta_x1 = torch.linalg.solve(jacobian, residual_equation)
        x1 = x1-delta_x1

        incremental = torch.norm(delta_x1)
        residual = torch.norm(residual_equation)
        epoch += 1

        print(f'Epoch={epoch}, incremental={incremental:.5e}, residual={residual:.5e}')

    print(f'===== Step={step} done, current time={current_time:.5e} =====')
    time_stamp[step] = current_time

    ddx1 = numerical_solver.calculate_ddx1(x0, dx0, ddx0, x1)
    dx1 = numerical_solver.calculate_dx1(dx0, ddx0, ddx1)

    x[:, step:step+1] = x1
    dx[:, step:step+1] = dx1
    ddx[:, step:step+1] = ddx1

    x0 = x1.detach()
    dx0 = dx1.detach()
    ddx0 = ddx1.detach()

end_time = time.time()
print(f'===== Time cost={end_time-start_time:.5e} =====')

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 6), dpi=200)

ax[0, 0].plot(time_stamp.cpu().detach().numpy(), x[0, :].cpu().detach().numpy())
ax[0, 1].plot(time_stamp.cpu().detach().numpy(), x[5, :].cpu().detach().numpy())
ax[1, 0].plot(time_stamp.cpu().detach().numpy(), x[10, :].cpu().detach().numpy())
ax[1, 1].plot(time_stamp.cpu().detach().numpy(), x[35, :].cpu().detach().numpy())
plt.show()
