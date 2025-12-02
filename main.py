import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import scipy.integrate as sci
from tkinter import *
from tkinter import messagebox
import sympy as sp

#func = lambda x: -50 * np.sin(x*np.pi/100) + 60
# func = lambda x: 60 - 0.5*x
# func = lambda x: 15* np.log(-x+101)
# func = lambda x: 0.025*(x-50)**2+5 
# func = lambda x: -0.5*x+70
# func = lambda x: -0.9*(0.2*x-3)**3 + 5*(0.2*x-3)**2

g = 9.81                                         # m/s^2
### GUI
#main window
master = Tk()
master.title('User input') 
master.geometry('400x300') #size


#labels for inpit
Label(master, text='Function ').grid(row=0) 
Label(master, text='Static ').grid(row=1) 
Label(master, text='Kinetic ').grid(row=2)
Label(master, text='xlim').grid(row=3)
Label(master, text='ylim ').grid(row=4)
Label(master, text='Radius ').grid(row=5)

#entry boxes
function = Entry(master) #-50 * sin(x*pi/100) + 60
mu_static = Entry(master) 
mu_kinetic = Entry(master)
xlimit = Entry(master)
ylimit = Entry(master)
radius = Entry(master)

#place entry boxes next to inputs
function.grid(row=0, column=1)
mu_static.grid(row=1, column=1) 
mu_kinetic.grid(row=2, column=1)
xlimit.grid(row=3, column=1)
ylimit.grid(row=4, column=1)
radius.grid(row=5, column=1)

def save_changes():
    '''
    reads inputs from entry boxes, checks if all values have been
    input, converts to float, closes window
    '''
    global mustat, mukin, xlim, ylim, r_particle, func


    func_str = function.get()  
    mustat = mu_static.get()
    mukin = mu_kinetic.get()
    xlim = xlimit.get()
    ylim = ylimit.get()
    r_particle = radius.get()

    x = sp.Symbol('x')
    f = sp.Function('f')(x)
    func_sp = sp.sympify(func_str)              #string to sympify expression
    func = sp.lambdify(x, func_sp, 'numpy')     #to numerical expression 

    if not mukin or not mustat or not xlim or not ylim or not r_particle or not func_str:
        messagebox.showerror('Error','Need all values') 
        return
    else:
        messagebox.showinfo('Saved', 'Values saved') 

    mustat = float(mustat) 
    mukin = float(mukin)
    xlim = float(xlim)
    ylim = float(ylim)
    r_particle = float(r_particle)

    master.destroy() 

def confirm_exit():
    '''
    asks for exit comfirmation
    '''
    response = messagebox.askyesnocancel('Confrim choices') 
    if response:
        save_changes()        

confirm_button = Button(master, text='Confirm', command=confirm_exit) 
confirm_button.grid(row=6,column=1)

mainloop()  #infinite loop, keeps window open

def get_surface(func):
    global xlim                                  # use constants defined at the top
    
    Npts = 1001                                   # number of points
    x = np.linspace(0.0, xlim, Npts)
    y = func(x)
    dydx = np.gradient(y, x)
    d2ydx2 = np.gradient(dydx, x)
    # d2ydx2[np.where(d2ydx2 < 1e-5)] = 0          # set to zero if too small (to avoid division by zero in radius)
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
    x = x
    y = y + r_particle / np.cos(angle)
    dydx = np.gradient(y, x)
    d2ydx2 = np.gradient(dydx, x)
    # d2ydx2_t[np.where(d2ydx2_t < 1e-7)] = 0  
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


def get_stateinitial(i_init):                                     # i=0 for left corner, i=-1 for right corner
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

    return normal_mod


def friction(v, dir, angle, normal_mod):                    # dir (direction) is passed as np.sign(vx) in def Derivatives()
    global g, mukin, mustat

    grav_along = np.abs(g * np.sin(angle))                  # gravity along the slope (absolute value)
    friction_max_kin  = mukin  * np.abs(normal_mod)
    friction_max_stat = mustat * np.abs(normal_mod)
    
    if v < 1e-5:                                            # static case
        friction_try = grav_along                           # try friction equal to the opposite moving forces
        if friction_try > friction_max_stat:                # if friction required to compensate gravity is larger than max allowed,
            friction_mod = friction_max_stat                # then set to max allowed
        else:     
            friction_mod = friction_try                     # else leave friction as it is
        if angle < 0.0:     
            friction_mod *= -1                              # if slope is downward, get a negative mod (needed for projections)
     
    else:                                                   # use kinetic friction if speed is not 0
        friction_mod = friction_max_kin * dir * (-1)        # set friction direction opposite to velocity direction

    return friction_mod                                     # contains the sign


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

    gravitya = np.asarray([0.0, -g])                                                        # a = acceleration
    
    normal_mod = normal(v=v, angle=angle, radius=radius)                                    # modulus of acceleration by normal reaction force
    normala = np.asarray([-normal_mod * np.sin(angle), normal_mod * np.cos(angle)])
    
    friction_mod = friction(v=v, dir=np.sign(vx), angle=angle, normal_mod=normal_mod)                 
    frictiona = np.asarray([friction_mod * np.cos(angle), friction_mod * np.sin(angle)])
    
    accel = gravitya + normala + frictiona

    ''' a draft of rotation'''
    phiddot = 5 * friction_mod / 2 / r_particle     * (np.sign(vx))             

    statedot = np.asarray([vx, vy, accel[0], accel[1], phidot, phiddot]) 
    return statedot


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
    
    return (circ)






surface, traj = get_surface(func)

'''with the sin function, the integrator crashes after 12s because of the uncontinuous transition between friction regimes'''
fps = 20                                                                         # frames per second
tmax = 10.0                                                                        # seconds
ntimes = int(tmax * fps)                                                           # amount of time points
tout = np.linspace(0, tmax, ntimes)         

stateinitial = get_stateinitial(0)
stateout = sci.solve_ivp(Derivatives, y0=stateinitial, t_span=(0.0, tmax), t_eval=tout, method='BDF')
print(stateout.message)
print(ntimes, np.shape(stateout.y))



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
time_template = 'time = %.1f s'                                                     # clock in the animation
vel_template = 'speed = %.1f m/s'
disp_template = 'displacement = %.1f m'
time_text = ax.text(0.03, 0.15, '', transform=ax.transAxes, fontdict={'fontsize': 16})
vel_text  = ax.text(0.03, 0.10, '', transform=ax.transAxes, fontdict={'fontsize': 16})
disp_text = ax.text(0.03, 0.05, '', transform=ax.transAxes, fontdict={'fontsize': 16})

cx, cy = stateinitial[0], stateinitial[1]
extent = [cx - r_particle, cx + r_particle, cy - r_particle, cy + r_particle]
circ = patches.Circle((cx, cy), radius=r_particle, facecolor='#e5383b', edgecolor='none', zorder=2)
ax.add_patch(circ)
image = plt.imread('double-spiral.png')
im = ax.imshow(image, extent=extent, origin='lower', zorder=2)
im.set_clip_path(circ)



ani = animation.FuncAnimation(fig, animate, repeat=True, frames=ntimes)
writer = animation.PillowWriter(fps=fps,                                           # 1s of gif = 1s of model
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('animation.gif', writer=writer)




'''
to do:

PHYSICS:
rotational friction -- negligable?
Stribeck friction?
stickiness == friction?
fix rotation - rolling or slipping -- tourqe
vectorize radius


Reduce timestep/increase solver accuracy
Limit slope in trajectory: clip dy/dx or use a smoother function.
Add a small floor to normal force to avoid division by zero.
Smooth friction transition using Stribeck-like function.

GUI:
GUI inputs need validation, it also needs to look nicer 
display loading progress --> show animation in gui window  
plot the setup before submitting the parameters to see if they are sensible
in get_stateinitial(), choose i=0 or i=-1 based on the function (0 if goes downhill, -1 if uphill)
'''
