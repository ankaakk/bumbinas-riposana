import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import scipy.integrate as sci



func = lambda x: -0.6*x+80          # surface equation

### CONSTANTS
g = 9.81                  # m/s^2
mukin = 0.0               # dimensionless; coefficient of kinetic friction
xlim, ylim = 100, 100     # m; simulation box size


def get_surface(func):
    global xlim
    
    Npts = 101          # N points
    surface = np.zeros((Npts,3))
    surface[:,0] = np.linspace(0, xlim, Npts)
    surface[:,1] = func(surface[:,0])

    for i in range(Npts-1):
        surface[i,2] = np.arctan((surface[i+1,1] - surface[i,1]) / (surface[i+1,0] - surface[i,0]))     # the last theta is 0.0
        # theta with the x axis;    positive for increasing function, negative for decreading;     in radians

    return surface


def Derivatives(state, t):
    global g, mukin         # use the constants defined at the top
    
    x  = state[0]           # unpack state array
    y  = state[1]
    vx = state[2]
    vy = state[3]

    # get a theta from the 3rd column of the surface array, corresponding to the largest x in surface that is <= x in this calulation
    theta = surface[np.max(np.where(surface[:,0]<=x)),2]

    # mucrit = g * np.sin(theta)                  # critical mu for v=const, i.e., F_friction = F_g along the slope

    gravitya = np.asarray([0.0, -g])                                                                   # a = acceleration
    normala = np.asarray([-g * np.cos(theta)* np.sin(theta), g * np.cos(theta) * np.cos(theta)])       # a due to normal reaction force
    
    # mukin set to 0 (at the top) because friction is not working properly
    frictionmod = mukin * np.linalg.norm(normala)                                                      # length of a_friction vector
    frictiona = np.asarray([-frictionmod * np.cos(theta), -frictionmod * np.sin(theta)])

    accel = gravitya + normala + frictiona

    statedot = np.asarray([vx, vy, accel[0], accel[1]])                  # vx, vy, ax, ay
    return statedot


### CALCULATIONS

surface = get_surface(func)                     # func is defined at the very top
theta0 = np.arctan((surface[1,1]-surface[0,1])/(surface[1,0]-surface[0,0]))     # initial slope
H = max(surface[:,1]-min(surface[:,1]))
T = np.sqrt(2*H/(g*(1-np.cos(theta0)**2)))      # estimate sliding time for a linear surface

ntimes = 100                                    # amount of time points
tmax = T                                        # in seconds
tout = np.linspace(0, tmax, ntimes)            
dt = tmax/ntimes                                # used in animtion for fps
stateinitial = np.asarray([surface[0,0], surface[0,1], 0.0, 0.0])       # x, y, vx, vy
stateout = sci.odeint(Derivatives, y0=stateinitial, t=tout)             # numerical integration happens here


### PLOTTING

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])         # fill the whole figure with ax
ax.set_axis_off()
fig.set_facecolor('black')              # facecolor = background color

ax.set_xlim([0, xlim])
ax.set_ylim([0, ylim])
# ax.plot(surface[:,0], surface[:,1] ,c='w')                                     # a single line of surface
poly = patches.Polygon(((0,0), *zip(surface[:,0], surface[:,1]), (xlim, 0)),     # filled surface 
                       color='#d3d3d3')
ax.add_patch(poly)

time_template = 'time = %.1f s'         # clock in the figure
vel_template = 'speed = %.1f m/s'
disp_template = 'dispalacement = %.1f m'
time_text = ax.text(0.03, 0.15, '', transform=ax.transAxes)
vel_text  = ax.text(0.03, 0.10, '', transform=ax.transAxes)
disp_text = ax.text(0.03, 0.05, '', transform=ax.transAxes)

scat = ax.scatter(stateinitial[0], stateinitial[1], s=150, c='#e5383b')     # the sliding particle


### ANIMATION

def animate(i):
    scat.set_offsets((stateout[i,0],stateout[i,1]))                           # shift the particle
    time_text.set_text(time_template % (tout[i]))                             # clock in the figure
    vel_text.set_text(vel_template % np.sqrt(stateout[i,2]**2 + stateout[i,3]**2))
    disp_text.set_text(disp_template % np.sqrt((stateout[i,0]-stateout[0,0])**2 + (stateout[i,1]-stateout[0,1])**2))
    return (scat,)


ani = animation.FuncAnimation(fig, animate, repeat=True, frames=ntimes)
writer = animation.PillowWriter(fps=1/dt,                                     # 1s of gif = 1s of model
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('1_sem_darbs/stable.gif', writer=writer)




'''
to do:
fix friction
add static friction
normal reacion force must depend on centripetal force
add a physical radius for the particle
visualize the particle with a patch with texture to see the rolling
check if the input surface func is inside the simulation box
raise exception (?) if the surface is going uphill from the start
stop the simulation when the particle reaches xlim
check if particle is in contact with the surface (allow a trampoline)
add rolling
'''