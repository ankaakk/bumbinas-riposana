import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import scipy.integrate as sci
plt.rcParams['animation.html'] = 'jshtml'


# func = lambda x: -np.sqrt(50**2 - (x-50)**2) + 70         # convex circle
# func = lambda x: np.sqrt(100**2 - (x)**2)-20              # concave circle
func = lambda x: -50 * np.sin(x*np.pi/100) + 60           # sinudoid
# func = lambda x: np.cosh(0.1*(x-50)) + 50                 # catenary
# func = lambda x: 60 - 0.5*x                               # flat
# func = lambda x: 15* np.log(-x+101)                       # log
# func = lambda x: 0.025*(x-50)**2+5                        # parabola
# func = lambda x: -0.5*x+70                                # another flat
# func = lambda x: -0.9*(0.2*x-3)**3 + 5*(0.2*x-3)**2       # cubic parabola

g = 9.81                                         # m/s^2
mukin  = 0.3                                     # dimensionless; coefficient of kinetic friction
mustat = 0.5                                      # dimensionless; coefficient of static friction
# print(f'Max incline for rolling without slipping: {np.rad2deg(np.arctan(7/2*mustat)):.1f} deg.')
xlim, ylim = 100, 100                            # m, siulation box size
r_particle = 2                                   # m, particle radius





def get_surface(func):
    global xlim                                  # use constants defined at the top
    
    Npts = 1001                                   # number of points
    x = np.linspace(0.0, xlim, Npts)
    y = func(x)
    dydx = np.gradient(y, x)
    d2ydx2 = np.gradient(dydx, x)
    angle = np.arctan(dydx)                      # in radians; positive if increasing
    for d in d2ydx2:
        if d == 0:
            radius = np.inf
        else:
            radius = (1+dydx**2)**1.5 / (d2ydx2)         # radius of curvature; positive if concave up (min)

    surface = np.ones((Npts,6))                 # x, y, dydx, d2ydx2, angle, radius
    surface[:,0] = x
    surface[:,1] = y
    surface[:,2] = dydx
    surface[:,3] = d2ydx2
    surface[:,4] = angle
    surface[:,5] = radius

    '''trajectory is always at a distance r_particle from the surface (along normal)
    if the surface has non-zero curvature, 
    the trajectory will be inside and hence have a smaller radius of curvature at every point,
    the slope angles are also slightly different'''
    # for the trajectory:
    x = x - r_particle * np.sin(angle)
    y = y + r_particle * np.cos(angle)

    dydx = np.gradient(y, x)
    d2ydx2 = np.gradient(dydx, x)
    angle = np.arctan(dydx)            
    for d in d2ydx2:
        if d == 0:
            radius = np.inf
        else:
            radius = (1+dydx**2)**1.5 / (d2ydx2)        

    trajectory = np.ones((Npts, 6))
    trajectory[:,0] = x
    trajectory[:,1] = y
    trajectory[:,2] = dydx
    trajectory[:,3] = d2ydx2
    trajectory[:,4] = angle
    trajectory[:,5] = radius

    return surface, trajectory


def get_stateinitial(i_init=0):                                     # i=0 for left corner, i=-1 for right corner
    global traj
    x_init = traj[i_init,0]
    y_init = traj[i_init,1]
    angle_init = traj[i_init, 4]
    stateinitial = np.asarray([x_init, y_init, 0.0, 0.0, 0.0, 0.0])
    return stateinitial


def normal(v, angle, radius):                                  # normal reaction force
    global g

    if radius == np.inf:
        centripetala = 0.0                                     # if curvature radius in infinite, surface is flat   
    else:           
        centripetala = v**2 / radius                           # radius is positive if concave up (function min)
 
    normal_mod = np.abs(g * np.cos(angle) + centripetala)      # centripetala contains a sign (from the radius sign)
 
    if normal_mod < 0.0:                                       # if required centripetala is larger than gravity can create,
        normal_mod = 0.0                                       # the ball looses contact with surface, so no normal force
        global contact
        contact = False

    return normal_mod


def friction(angle, centripetala, normal_mod):                  # only static friction for rolling without slipping        
    global g, mustat
    friction_max = mustat * normal_mod
    friction_mod = np.abs(2/7 * (g) * np.sin(angle))      # required for rolling without slipping
    if friction_mod > friction_max:
        friction_mod = friction_max
    return friction_mod                                 # abs value


def Derivatives(t, state):
    global g, r_particle
    
    x  = state[0]
    y  = state[1]
    vx = state[2]
    vy = state[3]
    v  = np.linalg.norm([vx, vy])
    phi = state[4]                                  # rotational angle
    phidot = state[5]                               # rotational velocity

    index = np.argmin(np.abs(traj[:,0]-x))          # index of x trajectory array which is closest to x of cm
    angle  = traj[index, 4]                         # slope at this x
    radius = traj[index, 5]

    vdir = np.sign(vx)                              # direction of movement, positive to the right
    sdir = np.sign(angle)                           # direction of slope, positive upward
     
    if radius == np.inf:
        centripetala = 0.0                                     # if curvature radius in infinite, surface is flat   
    else:           
        centripetala = v**2 / radius                           # radius is positive if concave up (function min)

    gravitya = np.asarray([0.0, -g])                                                        # a = acceleration
    
    '''no reaction and friction if lost contact with the surface
    does not attempt to calculate the reaction force once has lost contact => no bounces
    contact set to False in def normal() in this case'''
    if contact:
        normal_mod = normal(v=v, angle=angle, radius=radius)                                    # modulus of acceleration by normal reaction force
        friction_mod = friction(angle=angle, centripetala=centripetala, normal_mod=normal_mod) 
    else: 
        normal_mod = 0
        friction_mod = 0

    normala = np.asarray([-normal_mod * np.sin(angle), normal_mod * np.cos(angle)])
        
    '''only static friction for rolling without slipping'''
    friction_dir = friction_mod * sdir
    frictiona = np.asarray([friction_dir * np.cos(angle), friction_dir * np.sin(angle)])
    
    accel = gravitya + normala + frictiona

    phiddot = (5  * friction_dir) / (2 * r_particle)

    statedot = np.asarray([vx, vy, accel[0], accel[1], phidot, phiddot]) 
    energy.append(g*y + v**2/2 + 2/5*r_particle**2*phidot**2/2)             # will be on the screen

    '''for the case the integrator crashes'''
    if not np.all(np.isfinite(statedot)):
        print("BAD STATE at t =", t)
        print("state =", state)
        raise RuntimeError("Non-finite derivative detected")

    return statedot



surface, traj = get_surface(func)

fps = 15                                                                           # frames per second
tmax = 20.0                                                                        # seconds
ntimes = int(tmax * fps-1)                                                       # amount of time points
tout = np.linspace(0, tmax, ntimes)         
energy = []

contact = True
stateinitial = get_stateinitial(i_init=100)
method='Radau'                # RK45, RK23, DOP853, BDF, Radau, LSODA
stateout = sci.solve_ivp(Derivatives, y0=stateinitial, t_span=(0.0, tmax), t_eval=tout, method=method)
print(stateout.message)
energy = np.asarray(energy)







def animate(i):
    global xlim, ylim

    cx, cy = (stateout.y[0,i],stateout.y[1,i])
    circ.set_center((cx, cy))
    extent = [cx - r_particle, cx + r_particle, cy - r_particle, cy + r_particle]
    transform = mpl.transforms.Affine2D().rotate_around(cx, cy, stateout.y[4,i]) + ax.transData
    im.set_extent(extent)
    im.set_transform(transform)
    im.set_clip_path(circ)
    
    time_text.set_text(time_template % (tout[i]))                                   # clock in the figure
    vel_text.set_text(vel_template % (np.sqrt(stateout.y[2,i]**2 + stateout.y[3,i]**2)*np.sign(stateout.y[2,i])))
    disp_text.set_text(disp_template % np.sqrt((stateout.y[0,i]-stateout.y[0,0])**2 + (stateout.y[1,i]-stateout.y[1,0])**2))
    energy_text.set_text(energy_template % (energy[i]-energy[0]))

    return (circ)


fig = plt.figure(figsize=(7,7))          
ax = fig.add_axes([0, 0, 1, 1])                                                     # fill the whole figure with ax
ax.set_axis_off()                                      
fig.set_facecolor('black')                                                          # facecolor = background color
ax.set_facecolor('black')

ax.set_xlim([0,xlim])
ax.set_ylim([0,ylim])
ax.set_aspect('equal')
poly = patches.Polygon(((0,0), *zip(surface[:,0], surface[:,1]), (xlim, 0)),    
                       facecolor='#d3d3d3', edgecolor='none')                    # a polygon for surface  
ax.add_patch(poly)
# ax.plot(traj[:,0], traj[:,1], ls='--', c='orange')
time_template   = 'time = %.1f s'                                                     # clock in the animation
vel_template    = 'speed = %.1f m/s'
disp_template   = 'displacement = %.1f m'
energy_template = 'Δenergy = %.3f J/kg'
time_text   = ax.text(0.03, 0.20, '', transform=ax.transAxes, fontdict={'fontsize': 16})
vel_text    = ax.text(0.03, 0.15, '', transform=ax.transAxes, fontdict={'fontsize': 16})
disp_text   = ax.text(0.03, 0.10, '', transform=ax.transAxes, fontdict={'fontsize': 16})
energy_text = ax.text(0.03, 0.05, '', transform=ax.transAxes, fontdict={'fontsize': 16})

cx, cy = stateinitial[0], stateinitial[1]
extent = [cx - r_particle, cx + r_particle, cy - r_particle, cy + r_particle]
circ = patches.Circle((cx, cy), radius=r_particle, facecolor='#e5383b', edgecolor='none', zorder=2)
ax.add_patch(circ)
image = plt.imread('bumbinas-riposana/double-spiral.png')
im = ax.imshow(image, extent=extent, origin='lower', zorder=2)
im.set_clip_path(circ)



ani = animation.FuncAnimation(fig, animate, repeat=True, frames=ntimes)
writer = animation.PillowWriter(fps=fps,                                           # 1s of gif = 1s of model
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('bumbinas-riposana/animation.gif', writer=writer)
print('Successfully created an animation.')




'''
to do:

PHYSICS:
add kinetic friction
add rolling friction
fix if N<0 : N=0
return reaction force if bounces (if needed?)


GUI:
GUI inputs need validation, it also needs to look nicer 
display loading progress --> show animation in gui window  
plot the setup before submitting the parameters to see if they are sensible
for get_stateinitial(), let the user choose starting position as a fraction of xlim, 
    , convert to i (0 to 1000) for get_stateinitial()
after inputting mustat, display max angle for rolling without slipping: f'{np.rad2deg(np.arctan(7/2*mustat)):.1f} deg.'
    , and max angle in the input function f'{np.rad2deg(np.max(np.abs(surface[:,4]))):.1f} deg.'

    
SIMULATION ADJUSTMENTS BASED ON INPUT:
adjust xlim if func is a circle with radius < xlim
if func is not concave, stop simulation/animation when x > xlim (leaves the simulation box)


ANIMATION:
plot trajectory in animate()
additional info like method and fps, on the screen or in animation.PillowWriter(metadata=...)


vectorize radius
Reduce timestep/increase solver accuracy
Limit slope in trajectory: clip dy/dx or use a smoother function.
Add a small floor to normal force to avoid division by zero.
Smooth friction transition using Stribeck-like function.
'''
