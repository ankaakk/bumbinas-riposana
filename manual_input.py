import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import scienceplots
import scipy.integrate as sci
import time 

plt.style.use(['science', 'grid', 'notebook'])
plt.rcParams['animation.html'] = 'html5'
start_time = time.time()

# ----------------------------------------------------------------------------------------------------------

g = 9.81                                         # m/s^2
mukin  = 0.2                                     # dimensionless; coefficient of kinetic friction
mustat = 0.5                                      # dimensionless; coefficient of static friction
xlim, ylim = 100, 100                            # m, siulation box size
r_particle = 2                                   # m, particle radius
icoef = 2/5                                      # coefficient before mr^2 in moment of inertia

# ----------------------------------------------------------------------------------------------------------

'''adjust xlim if func is a circle'''
# func = lambda x: -np.sqrt(50**2 - (x-50)**2) + 70         # convex circle
# func = lambda x: np.sqrt(100**2 - (x)**2)-20              # concave circle
# func = lambda x: -50 * np.sin(x*np.pi/100) + 60           # sinudoid
# func = lambda x: np.cosh(0.05*(x-50)) + 50                # very flat catenary
# func = lambda x: 60 - 0.5*x                               # flat
# func = lambda x: 15* np.log(-x+101)                       # log
func = lambda x: 0.025*(x-50)**2+5                        # parabola
# func = lambda x: 0.005*(x-50)**2 + 50                     # flat parabola
# func = lambda x: -0.5*x+70                                # another flat
# func = lambda x: -0.9*(0.2*x-3)**3 + 5*(0.2*x-3)**2 +15   # cubic parabola
# func = lambda x: 100*np.exp(-0.05*x)                      # exponent
# func = lambda x: 50                               


# ----------------------------------------------------------------------------------------------------------


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
            radius = (1+dydx**2)**1.5 / (d2ydx2)  

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
    v_init = 0.
    stateinitial = np.asarray([x_init, y_init, v_init*np.cos(angle_init), v_init*np.sin(angle_init), 0.0, 0.0, 0.0])
    return stateinitial


def normal(angle, cpetal):                                  # normal reaction force
    global g
    normal_mod = g * np.cos(angle) + cpetal      # centripetala contains a sign (from the radius sign)
    return normal_mod


def sigmoid_friction(vslip, vnorm, vtol, angle, normal_mod):
    global g, mukin, mustat
    fkin = mukin * normal_mod
    # print(fkin)
    fstatmax = mustat * normal_mod
    fstat = np.abs(2/7 * (g) * np.sin(angle))
    if fstat > fstatmax:
        fstat = fstatmax
    f = -2*(fstat - fkin) / (1 + np.exp(-vslip/vnorm)) + 2*fstat - fkin
    if np.abs(vslip) < vtol:
        f = fstat
    return f


def Derivatives(t, state):
    global g, r_particle
    
    x  = state[0]
    y  = state[1]
    vx = state[2]
    vy = state[3]
    v  = np.linalg.norm([vx, vy])
    phi = state[4]                                             # rotational angle
    phidot = state[5]                                          # rotational velocity
    workdot = state[6]                                         # work done by friction if slipping, per unit mass
        
    index = np.argmin(np.abs(traj[:,0]-x))                     # index of x trajectory array which is closest to x of cm
    angle  = traj[index, 4]                                    # slope at this x
    radius = traj[index, 5]

     
    if radius == np.inf:
        centripetala = 0.0                                     # if curvature radius in infinite, surface is flat   
    else:           
        centripetala = v**2 / radius                           # radius is positive if concave up (function min)


    vtan = vx * np.cos(angle) + vy * np.sin(angle)             # mag same as v, but has sign along the slope
    vslip = vtan + (phidot*r_particle)                         # has a sign along the slope
    vslipabs = np.abs(vslip)
    vslipdir = np.sign(vslip)


    '''a sigmoid function to avoid uncontinuous derivative change in the friction force (integrators don't like it)
    pnorm meaning: fstat-f(vtol) = pnorm*(fstat-fkin); mathematically 0 < pnorm < 1, physically pnorm ~= 0.95  
    effectively vtol (a small number) is where the transition from fkin to fstat starts
    vnorm is a normalizatiion parameters that controls the width of sigmoid
    the choice of vtol and pnorm is quite random
    the function is still manually grounded to fstat in def friction, but works better with the sigmoid'''
    vswitch = 1e-1
    pnorm = 0.95                                             # fraction of fstat-fkin
    vnorm = vswitch / np.log((1+pnorm)/(1-pnorm))            # np.log() is ln
    vtol = 1e-2                                              # below this vslip is considered 0 (in def friction)
  

    gravitya = np.asarray([0.0, -g])                        # a = acceleration

    normal_mod = normal(angle=angle, cpetal=centripetala)                              
    normala = np.asarray([-normal_mod * np.sin(angle), normal_mod * np.cos(angle)])
    
    friction_mod = sigmoid_friction(vslip=vslipabs, vnorm=vnorm, vtol=vtol, angle=angle, normal_mod=normal_mod) 
    friction_signed = friction_mod * -vslipdir
    frictiona = np.asarray([friction_signed * np.cos(angle), friction_signed * np.sin(angle)])


    
    accel = gravitya + normala + frictiona


    phiddot = (friction_signed) / (icoef*r_particle)                # positive counterclockwise

    workdot = -friction_signed * vslip                              # stored positive

    statedot = np.asarray([vx, vy, accel[0], accel[1], phidot, phiddot, workdot]) 
    return statedot


# ----------------------------------------------------------------------------------------------------------


surface, traj = get_surface(func)

fps = 20                                                                           # frames per second
tmax = 40                                                                          # seconds
ntimes = int(tmax * fps-1)                                                         # amount of time points
tout = np.linspace(0, tmax, ntimes)         
dt = tout[1]-tout[0]

stateinitial = get_stateinitial(i_init=0)
method='RK23'                                                                      # RK45, RK23, DOP853, BDF, Radau (fails), LSODA
stateout = sci.solve_ivp(Derivatives, y0=stateinitial, t_span=(0.0, tmax), t_eval=tout, method=method, max_step=dt)
y = stateout.y[1]
phidot = stateout.y[5]
v = np.asarray([np.linalg.norm([stateout.y[2,i], stateout.y[3,i]]) for i in range(len(stateout.t))])
energy = g*y + v**2/2 + icoef*r_particle**2*phidot**2/2

print(stateout.message)
print(ntimes, np.shape(stateout.y))



# ----------------------------------------------------------------------------------------------------------



def animate(i):
    global xlim, ylim

    cx, cy = (stateout.y[0,i],stateout.y[1,i])
    circ.set_center((cx, cy))
    extent = [cx - r_particle, cx + r_particle, cy - r_particle, cy + r_particle]
    transform = mpl.transforms.Affine2D().rotate_around(cx, cy, stateout.y[4,i]) + ax.transData
    im.set_extent(extent)
    im.set_transform(transform)
    im.set_clip_path(circ)

    line.set_data(stateout.y[0,:i], stateout.y[1,:i])

    time_text.set_text(time_template % (tout[i]))                                  
    vel_text.set_text(vel_template % (np.sqrt(stateout.y[2,i]**2 + stateout.y[3,i]**2)*np.sign(stateout.y[2,i])))
    angvel_text.set_text(angvel_template % (-stateout.y[5,i]*r_particle))
    disp_text.set_text(disp_template % np.sqrt((stateout.y[0,i]-stateout.y[0,0])**2 + (stateout.y[1,i]-stateout.y[1,0])**2))
    energy_text.set_text(energy_template % ((energy[i]+stateout.y[6,i]-energy[0])/energy[0]*100))

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
                    facecolor='#d3d3d3', edgecolor='none')                        # a polygon for surface  
ax.add_patch(poly)
ax.plot(traj[:,0], traj[:,1], lw=1, ls='--', c='coral')                             # trajectory along the surface
line, = ax.plot([], [], 'o-', lw=1, c='crimson', ms=4, alpha=0.8)                   # actual path

time_template      = 't   = %.1f s'                                                 
vel_template       = 'vT = %.1f m/s'
angvel_template    = 'vR = %.1f m/s'
disp_template      = 'd   = %.1f m'
energy_template    = 'ΔE = %.2f%%'
time_text      = ax.text(0.03, 0.25, '', transform=ax.transAxes, fontdict={'fontsize': 14})
vel_text       = ax.text(0.03, 0.20, '', transform=ax.transAxes, fontdict={'fontsize': 14})
angvel_text    = ax.text(0.03, 0.15, '', transform=ax.transAxes, fontdict={'fontsize': 14})
disp_text      = ax.text(0.03, 0.10, '', transform=ax.transAxes, fontdict={'fontsize': 14})
energy_text    = ax.text(0.03, 0.05, '', transform=ax.transAxes, fontdict={'fontsize': 14})

cx, cy = stateinitial[0], stateinitial[1]
extent = [cx - r_particle, cx + r_particle, cy - r_particle, cy + r_particle]
circ = patches.Circle((cx, cy), radius=r_particle, facecolor='crimson', edgecolor='none', zorder=2)
ax.add_patch(circ)
image = plt.imread('double-spiral.png')
im = ax.imshow(image, extent=extent, origin='lower', zorder=2)
im.set_clip_path(circ)



ani = animation.FuncAnimation(fig, animate, repeat=True, frames=ntimes)
writer = animation.PillowWriter(fps=fps,                                           # 1s of gif = 1s of model
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('animation.gif', writer=writer)


print("--- %s seconds ---" % (time.time() - start_time))




'''
to do:

PHYSICS:
add rolling friction


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
additional info like method and fps, on the screen or in animation.PillowWriter(metadata=...)


vectorize radius
Reduce timestep/increase solver accuracy
Limit slope in trajectory: clip dy/dx or use a smoother function.
Add a small floor to normal force to avoid division by zero.
Smooth friction transition using Stribeck-like function.
'''
