import torch


'''Define the newmark-beta method'''
class NewmarkBetaMethod(torch.nn.Module):
    def __init__(self, gamma=0.5, delta=0.25, delta_t=0.01):
        super(NewmarkBetaMethod, self).__init__()
        # Parameters for the newmark-beta method
        self.gamma = gamma
        self.delta = delta
        self.delta_t = delta_t
        self.a0 = 1 / (delta * delta_t ** 2)
        self.a2 = 1 / (delta * delta_t)
        self.a3 = 1 / (2 * delta) - 1
        self.a6 = delta_t * (1 - gamma)
        self.a7 = gamma * delta_t

    def initialize_system(self, system):
        self.system = system

    def calculate_ddx1(self, x0, dx0, ddx0, x1):
        return self.a0*(x1-x0)-self.a2*dx0-self.a3*ddx0

    def calculate_dx1(self, dx0, ddx0, ddx1):
        return self.a6*ddx0+self.a7*ddx1+dx0

    def calculate_residual_sum(self, x1, x0 ,dx0, ddx0, current_time):
        ddx1 = self.calculate_ddx1(x0, dx0, ddx0, x1)
        dx1 = self.calculate_dx1(dx0, ddx0, ddx1)
        residual_equation_sum = self.system.calculate_residual(x1, dx1, ddx1, current_time).sum(0)
        return residual_equation_sum

    def calculate_residual(self, x1, x0 ,dx0, ddx0, current_time):
        ddx1 = self.calculate_ddx1(x0, dx0, ddx0, x1)
        dx1 = self.calculate_dx1(dx0, ddx0, ddx1)
        residual_equation = self.system.calculate_residual(x1, dx1, ddx1, current_time)
        return residual_equation


'''Define the generalized-alpha method'''
class GeneralizedAlphaMethod(torch.nn.Module):
    def __init__(self, rho=0.5, delta_t=0.01):
        super(GeneralizedAlphaMethod, self).__init__()
        # Parameters for the generalized-alpha method
        alpha_m = (2*rho-1)/(rho+1)
        alpha_f = rho/(rho+1)
        self.gamma = 0.5-alpha_m+alpha_f
        self.beta = 0.25*(1-alpha_m+alpha_f)**2
        self.delta_t = delta_t

    def initialize_system(self, system):
        self.system = system

    def calculate_ddx1(self, x0, dx0, ddx0, x1):
        return 1.0/(self.beta*self.delta_t**2)*(x1-x0)-dx0/(self.beta*self.delta_t)-(1.0/2.0/self.beta-1.0)*ddx0

    def calculate_dx1(self, x0, dx0, ddx0, x1):
        return (self.gamma/self.beta/self.delta_t*(x1-x0)-(self.gamma/self.beta-1.0)*dx0 -
                (self.gamma/2.0/self.beta-1.0)*ddx0*self.delta_t)

    def calculate_residual_sum(self, x1, x0 ,dx0, ddx0, current_time):
        ddx1 = self.calculate_ddx1(x0, dx0, ddx0, x1)
        dx1 = self.calculate_dx1(x0, dx0, ddx0, x1)
        residual_equation_sum = self.system.calculate_residual(x1, dx1, ddx1, current_time).sum(0)
        return residual_equation_sum

    def calculate_residual(self, x1, x0 ,dx0, ddx0, current_time):
        ddx1 = self.calculate_ddx1(x0, dx0, ddx0, x1)
        dx1 = self.calculate_dx1(x0, dx0, ddx0, x1)
        residual_equation = self.system.calculate_residual(x1, dx1, ddx1, current_time)
        return residual_equation
