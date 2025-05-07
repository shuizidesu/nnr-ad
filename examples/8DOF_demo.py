import torch
import time
import matplotlib.pyplot as plt
from torch.func import jacrev

from nonlinear_ode_solver.solver_integrator import NewmarkBetaMethod

torch.set_default_dtype(torch.double)


'''Config and parameter'''
# System parameter
device = 'cpu'
# Parameters for the newmark-beta method
delta_t = 0.01
solve_steps = 10000-1
# Parameters for the newton iteration
incremental_tolerance = 1e-12
residual_tolerance = 1e-12
max_epoch = 200

'''Define the system equation'''
class Rotor8DOFHertz(torch.nn.Module):
    def __init__(self):
        super(Rotor8DOFHertz, self).__init__()
        self.lmda = torch.tensor(1.25, device=device, dtype=torch.double)
        self.e1 = torch.tensor(1e-6, device=device, dtype=torch.double)
        self.e2 = torch.tensor(5e-7, device=device, dtype=torch.double)
        self.c1, self.c2, self.c3 = 655.0, 655.0, 655.0
        self.k1, self.k2, self.k3 = 6e7, 6e7, 6e7
        self.l1 = 0.9188
        self.l2 = 1.1122
        self.l3 = 0.5120
        self.l4 = 0.6243
        self.l5 = 0.0995
        self.m = torch.tensor([97.37, 108.3], device=device, dtype=torch.double)
        self.ri = torch.tensor(58e-3, device=device, dtype=torch.double)
        self.ro = torch.tensor(66e-3, device=device, dtype=torch.double)
        self.Nb = 8
        self.Jd = torch.tensor([1.8454, 2.0060], device=device, dtype=torch.double)
        self.Jp = torch.tensor([3.6907, 4.0119], device=device, dtype=torch.double)
        self.Kb = torch.tensor(1e8, device=device, dtype=torch.double)
        self.g = torch.tensor(9.8, device=device, dtype=torch.double)
        self.l = self.l1 + self.l2
        self.delta0 = torch.tensor(5e-6, device=device, dtype=torch.double)

    def initialize_parameters(self, omega1):
        self.omega1 = omega1
        self.omega2 = omega1 * self.lmda
        self.omega_c = (self.ri + self.lmda * self.ro) / (self.ri + self.ro)
        self.E1 = self.e1 / self.delta0
        self.E2 = self.e2 / self.delta0

        ita1 = self.Jp[0] / self.Jd[0]
        ita2 = self.Jp[1] / self.Jd[1]
        L1 = self.l1 / self.l

        L2 = self.l2 / self.l
        self.L2 = L2

        L3 = self.l3 / self.l

        L4 = self.l4 / self.l
        self.L4 = L4

        L5 = self.l5 / self.l
        self.L5 = L5

        C1 = self.c1 / (self.m[0] * self.omega1)
        C2 = self.c2 / (self.m[0] * self.omega1)
        C3 = self.c1 * self.l1 * self.l / (self.Jd[0] * self.omega1)
        C4 = self.c2 * self.l2 * self.l / (self.Jd[0] * self.omega1)
        C5 = self.c3 / (self.m[1] * self.omega1)
        C6 = self.c3 * self.l3 * self.l / (self.Jd[1] * self.omega1)
        K1 = self.k1 / (self.m[0] * self.omega1 ** 2)
        K2 = self.k2 / (self.m[0] * self.omega1 ** 2)
        K3 = self.k1 * self.l1 * self.l / (self.Jd[0] * self.omega1 ** 2)
        K4 = self.k2 * self.l2 * self.l / (self.Jd[0] * self.omega1 ** 2)
        K5 = self.k3 / (self.m[1] * self.omega1 ** 2)
        K6 = self.k3 * self.l3 * self.l / (self.Jd[1] * self.omega1 ** 2)

        self.M = torch.eye(8, device=device, dtype=torch.double)
        self.C = torch.tensor([
            [C1+C2, 0,      0,            C2*L2-C1*L1,  0,    0,   0,           0],
            [0,     C1+C2,  C1*L1-C2*L2,  0,            0,    0,   0,           0],
            [0,     C3-C4,  C3*L1+C4*L2,  ita1,         0,    0,   0,           0],
            [C4-C3, 0,      -ita1,        C3*L1+C4*L2,  0,    0,   0,           0],
            [0,     0,      0,            0,            C5,   0,   0,           -C5*L3],
            [0,     0,      0,            0,            0,    C5,  C5*L3,       0],
            [0,     0,      0,            0,            0,    C6,  C6*L3,       self.lmda*ita2],
            [0,     0,      0,            0,            -C6,  0,   -self.lmda*ita2,  C6*L3]
        ], device=device, dtype=torch.double)
        self.K = torch.tensor([
            [K1+K2, 0,      0,            K2*L2-K1*L1,  0,    0,   0,          0],
            [0,     K1+K2,  K1*L1-K2*L2,  0,            0,    0,   0,          0],
            [0,     K3-K4,  K3*L1+K4*L2,  0,            0,    0,   0,          0],
            [K4-K3, 0,      0,            K3*L1+K4*L2,  0,    0,   0,          0],
            [0,     0,      0,            0,            K5,   0,   0,          -K5*L3],
            [0,     0,      0,            0,            0,    K5,  K5*L3,      0],
            [0,     0,      0,            0,            0,    K6,  K6*L3,      0],
            [0,     0,      0,            0,            -K6,  0,   0,          K6*L3]
        ], device=device, dtype=torch.double)

    def force(self, current_time):
        Qt = torch.zeros([8, 1], device=device, dtype=torch.double)

        Qt[0] = self.E1*torch.cos(current_time)-self.g/(self.omega1**2*self.delta0)
        Qt[1] = self.E1*torch.sin(current_time)

        Qt[4] = self.lmda**2*self.E2*torch.cos(self.lmda*current_time)-self.g/(self.omega1**2*self.delta0)
        Qt[5] = self.lmda**2*self.E2*torch.sin(self.lmda*current_time)

        return Qt

    def nonlinearity(self, x, current_time):
        nonlinear = torch.zeros([8, 1], device=device, dtype=torch.double)

        theta = (2*torch.pi/
                 self.Nb*torch.linspace(0, self.Nb-1, self.Nb, device=device, dtype=torch.double).unsqueeze(1)+
                 self.omega_c*current_time)
        delta = ((x[0] + x[3] * (self.L2 - self.L5)- x[4] - x[7] * self.L4).unsqueeze(1) * torch.cos(theta) +
                 (x[1] - x[2] * (self.L2 - self.L5) - x[5] + x[6] * self.L4).unsqueeze(1) * torch.sin(theta)) - 1.0

        delta = torch.where(delta < 0.0, torch.tensor(0.0, device=device, dtype=torch.double), delta)

        FX = (self.Kb * self.delta0 ** (1 / 9) * delta ** (10 / 9) * torch.cos(theta)).sum()
        FY = (self.Kb * self.delta0 ** (1 / 9) * delta ** (10 / 9) * torch.sin(theta)).sum()

        nonlinear[0] = FX / (self.m[0] * self.omega1 ** 2)
        nonlinear[1] = FY / (self.m[0] * self.omega1 ** 2)
        nonlinear[2] = -FY * self.l * (self.l2 - self.l5) / (self.Jd[0] * self.omega1 ** 2)
        nonlinear[3] = FX * self.l * (self.l2 - self.l5) / (self.Jd[0] * self.omega1 ** 2)

        nonlinear[4] = -FX / (self.m[1] * self.omega1 ** 2)
        nonlinear[5] = -FY / (self.m[1] * self.omega1 ** 2)
        nonlinear[6] = FY * self.l * self.l4 / (self.Jd[1] * self.omega1 ** 2)
        nonlinear[7] = -FX * self.l * self.l4 / (self.Jd[1] * self.omega1 ** 2)

        return nonlinear

    def prepare_initial_ddx0(self, x0, dx0, omega1, current_time):
        self.initialize_parameters(omega1)
        Qt = self.force(current_time)
        return  torch.linalg.solve(self.M, Qt - self.nonlinearity(x0, current_time) - self.K @ x0 - self.C @ dx0)

    def calculate_residual(self, x, dx, ddx, current_time):
        Qt = self.force(current_time)
        residual_vector = self.M @ ddx + self.C @ dx + self.K @ x + self.nonlinearity(x, current_time) - Qt
        return residual_vector


# Give the initial status
my_system = Rotor8DOFHertz()
omega1 = torch.tensor(800.0, device=device, dtype=torch.double)

x0 = torch.zeros([8, 1], device=device, dtype=torch.double)
dx0 = torch.zeros([8, 1], device=device, dtype=torch.double)
current_time = torch.tensor(0.0, device=device, dtype=torch.double)
ddx0 = my_system.prepare_initial_ddx0(x0, dx0, omega1, current_time)

'''Solve!'''
x = torch.zeros([8, solve_steps+1], device=device, dtype=torch.double)
dx = torch.zeros([8, solve_steps+1], device=device, dtype=torch.double)
ddx = torch.zeros([8, solve_steps+1], device=device, dtype=torch.double)
time_stamp = torch.zeros([solve_steps+1, 1], device=device, dtype=torch.double)
x[:, 0:1] = x0
dx[:, 0:1] = dx0
ddx[:, 0:1] = ddx0
x1 = torch.zeros([8, 1], device=device, dtype=torch.double)
numerical_solver = NewmarkBetaMethod(gamma=0.5, delta=0.25, delta_t=delta_t)
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

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(15, 6), dpi=200)

ax[0, 0].plot(time_stamp.cpu().detach().numpy(), x[2, :].cpu().detach().numpy())
ax[0, 0].plot(time_stamp.cpu().detach().numpy(), dx[2, :].cpu().detach().numpy())
ax[0, 1].plot(time_stamp.cpu().detach().numpy(), x[3, :].cpu().detach().numpy())
ax[0, 1].plot(time_stamp.cpu().detach().numpy(), dx[3, :].cpu().detach().numpy())
plt.show()
