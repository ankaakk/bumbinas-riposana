from main_func import *
import time

start_time = time.time()

# func = lambda x: -np.sqrt(50**2 - (x-50)**2) + 70         # convex circle
# func = lambda x: np.sqrt(100**2 - (x)**2)-20              # concave circle
# func = lambda x: -50 * np.sin(x*np.pi/100) + 60           # sinudoid
# func = lambda x: np.cosh(0.05*(x-50)) + 50                # very flat catenary
# func = lambda x: 60 - 0.5*x                               # flat
# func = lambda x: 15* np.log(-x+101)                       # log, requires contact_flag
func = lambda x: 0.025*(x-50)**2+5                        # parabola
# func = lambda x: 0.005*(x-50)**2 + 50                     # flat parabola
# func = lambda x: -0.5*x+70                                # another flat
# func = lambda x: -0.9*(0.2*x-3)**3 + 5*(0.2*x-3)**2 +15   # cubic parabola, requires contact_flag
# func = lambda x: 100*np.exp(-0.05*x)                      # exponent
# func = lambda x: 1.5e-5*(x-50)**4                         # fourth power
# func = lambda x: 50

'''default values'''
mukin  = 0.2                                     # dimensionless; coefficient of kinetic friction
mustat = 0.3                                     # dimensionless; coefficient of static friction
crf = 0.01                                      # dimensionless; coefficient of rolling friction/resistance; ~b/r
r_particle = 2                                   # m, particle radius
icoef = 2/5     
method= 'RK23'

solve(func, do_animation=False, tmax=30)

print("--- %s seconds ---" % (time.time() - start_time))