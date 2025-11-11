import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import scipy.integrate as sci
import time
from tkinter import *
from tkinter import messagebox
start_time = time.time()


func = lambda x: 0.025*(x-50)**2+5               # equation of surface
# func = lambda x: -0.9*(0.2*x-3)**3 + 5*(0.2*x-3)**2

### CONSTANTS
g = 9.81                                         # m/s^2
''' mukin must be smaller than mustat '''

### GUI
#main window
master = Tk()
master.title('User input') 
master.geometry('400x300') #size

#labels for inpit
Label(master, text='Static ').grid(row=1) 
Label(master, text='Kinetic ').grid(row=2)
Label(master, text='xlim').grid(row=3)
Label(master, text='ylim ').grid(row=4)
Label(master, text='Radius ').grid(row=5)

#entry boxes
mu_static = Entry(master) 
mu_kinetic = Entry(master)
xlimit = Entry(master)
ylimit = Entry(master)
radius = Entry(master)

#place entry boxes next to inputs
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
    global mustat, mukin, xlim, ylim, r_particle

    mustat = mu_static.get()
    mukin = mu_kinetic.get()
    xlim = xlimit.get()
    ylim = ylimit.get()
    r_particle = radius.get()


    if not mukin or not mustat or not xlim or not ylim or not r_particle:
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
    angle = np.arctan(dydx)                      # in radians; positive if increasing
    radius = (1+dydx**2)**1.5 / (d2ydx2)         # radius of curvature; positive if concave up (min)

    surface = np.zeros((Npts,6))                 # x, y, dydx, d2ydx2, angle, radius
    surface[:,0] = x
    surface[:,1] = y
    surface[:,2] = dydx
    surface[:,3] = d2ydx2
    surface[:,4] = angle
    surface[:,5] = radius

    # plt.plot(surface[:,0], surface[:,1])
    return surface


def normal(v, angle, radius):
    global g

    if radius == np.inf:
        centripetala = 0.0                             # if curvature radius in indinite, surface is flat   
    else:  
        centripetala = v**2 / radius                   # radius is positive if concave up (function min)
 
    normal_mod = g * np.cos(angle) + centripetala      # centripetala contains a sign (from the radius sign)
 
    if normal_mod < 0.0:                               # if required centripetala is larger than gravity can create,
        normal_mod = 0.0                               # the ball looses contact with surface, so no normal force

    return normal_mod


def friction(vx, angle, normal_mod):
    global g, mukin, mustat

    grav_along = np.abs(g * np.sin(angle))                  # gravity along the slope (absolute value)
    friction_max_kin = mukin * normal_mod
    friction_max_stat = mustat * normal_mod
    
    if vx == 0.0:                                           # vx, not v, because it contains the sign
        friction_try = + grav_along                         # try friction equal to the opposite moving forces
        if friction_try > friction_max_stat:                # if friction required to compensate gravity is larger than max allowed,
            friction_mod = friction_max_stat                # then set to max allowed
        else:     
            friction_mod = friction_try                     # else leave friction as it is
        if angle < 0.0:     
            friction_mod *= -1                              # if slope is downward, get a negative mod (needed for projections)
     
    else:                                                   # use kinetic friction if speed is not 0
        friction_mod = friction_max_kin * np.sign(vx) * (-1)    # set friction direction opposite to velocity direction

    return friction_mod                                     # contains the sign


def Derivatives(state, t):
    global g, r_particle
    
    x  = state[0]
    y  = state[1]
    vx = state[2]
    vy = state[3]
    v = np.linalg.norm([vx, vy])
    phi = state[4]                      # rotation angle
    phidot = state[5]                   # rotation velocity

    ''' get angle and radius from the surface array, corresponding to the largest x in surface that is <= x in this calculation '''
    index = np.max(np.where(surface[:,0]<=x))
    angle  = surface[index, 4]
    radius = surface[index, 5]

    ''' recall that angle is negative for a downward slope, hence the signs in force components '''
    gravitya = np.asarray([0.0, -g])                                                        # a = acceleration
    
    normal_mod = normal(v=v, angle=angle, radius=radius)                                    # modulus of acceleration by normal reaction force
    normala = np.asarray([-normal_mod * np.sin(angle), normal_mod * np.cos(angle)])
    
    friction_mod = friction(vx=vx, angle=angle, normal_mod=normal_mod)                 
    frictiona = np.asarray([friction_mod * np.cos(angle), friction_mod * np.sin(angle)])
    
    accel = gravitya + normala + frictiona

    # phidot = v / r_particle * (-np.sign(vx))                                        # rotational velocity
    ''' moment of inertia of a solid sphere: 2/5 * m * R**2
    rot. acceleration = torque / mom. of in. = force * R / mom. of in. = m*a * R / (2/5 * m * R**2)'''
    phiddot = 5 * friction_mod / 2 / r_particle     * (np.sign(vx))                # cancelled mass and radius

    statedot = np.asarray([vx, vy, accel[0], accel[1], phidot, phiddot])            
    return statedot





### CALCULATIONS

surface = get_surface(func)
theta0 = np.arctan((surface[1,1]-surface[0,1])/(surface[1,0]-surface[0,0]))
H = max(surface[:,1]-min(surface[:,1]))
T = np.sqrt(2*H/(g*(1-np.cos(theta0)**2)))

ntimes = 100                                                                        # amount of time points
tmax = 5.0                                                                          # seconds
tout = np.linspace(0, tmax, ntimes)         
dt = tmax/ntimes          
angle_init = surface[0,4]  
stateinitial = np.asarray([surface[0,0]-r_particle*np.sin(angle_init),              # 0, x
                            surface[0,1]+r_particle*np.cos(angle_init),             # 1, y
                            0.0,                                                    # 2, vx
                            0.0,                                                    # 3, vy
                            0.0,                                                    # 4, phidot
                            0.0,                                                    # 5, phiddot
                            ])                   # x, y, vx, vy
stateout = sci.odeint(Derivatives, y0=stateinitial, t=tout, full_output=0)                         # numerical integration here





### PLOTTING            

fig = plt.figure(figsize=(7,7))          
ax = fig.add_axes([0, 0, 1, 1])                                                     # fill the whole figure with ax
ax.set_axis_off()                                      
fig.set_facecolor('black')                                                          # facecolor = background color
ax.set_facecolor('black')

ax.set_xlim([-1,xlim])
ax.set_ylim([-1,ylim])
ax.set_aspect('equal')
# ax.plot(surface[:,0], surface[:,1] ,c='w')                                        # a line for surface
poly = patches.Polygon(((0,0), *zip(surface[:,0], surface[:,1]), (xlim, 0)),    
                       color='#d3d3d3')#, edgecolor='#FFFFFF', lw=3)              # a polygon for surface  
ax.add_patch(poly)

time_template = 'time = %.1f s'                                                     # clock in the animation
vel_template = 'speed = %.1f m/s'
disp_template = 'displacement = %.1f m'
time_text = ax.text(0.03, 0.15, '', transform=ax.transAxes)
vel_text  = ax.text(0.03, 0.10, '', transform=ax.transAxes)
disp_text = ax.text(0.03, 0.05, '', transform=ax.transAxes)

# scat = ax.scatter(stateinitial[0], stateinitial[1], s=150, c='#e5383b')           # the sliding particle
image = plt.imread('double-spiral.png')

cx, cy = stateinitial[0], stateinitial[1]
extent = [cx - r_particle, cx + r_particle, cy - r_particle, cy + r_particle]
circ = patches.Circle((cx, cy), radius=r_particle, facecolor='#e5383b')
ax.add_patch(circ)
im = ax.imshow(image, extent=extent, origin='lower', zorder=2)
im.set_clip_path(circ)





### ANIMATION

def animate(i):
    # scat.set_offsets((stateout[i,0],stateout[i,1]))                                 # move the particle

    cx, cy = (stateout[i,0],stateout[i,1])
    circ.set_center((cx, cy))
    extent = [cx - r_particle, cx + r_particle, cy - r_particle, cy + r_particle]
    transform = mpl.transforms.Affine2D().rotate_around(cx, cy, stateout[i,4]) + ax.transData
    im.set_extent(extent)
    im.set_transform(transform)
    im.set_clip_path(circ)
    
    time_text.set_text(time_template % (tout[i]))                                   # clock in the figure
    vel_text.set_text(vel_template % (np.sqrt(stateout[i,2]**2 + stateout[i,3]**2)*np.sign(stateout[i,2])))
    disp_text.set_text(disp_template % np.sqrt((stateout[i,0]-stateout[0,0])**2 + (stateout[i,1]-stateout[0,1])**2))
    
    return (im)


ani = animation.FuncAnimation(fig, animate, repeat=True, frames=ntimes)#, interval=300)
writer = animation.PillowWriter(fps=1/dt,                                           # 1s of gif = 1s of model
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('animation69.gif', writer=writer)


print("--- %s seconds ---" % (time.time() - start_time))




'''
to do:
!!! consider contact point or CM position for calulations in Derivatives(), not only in stateinitial
GUI inputs need validation, it also needs to look nicer 
make a safer vx==0 in friction()
check if the input surface func is inside the simulation box
raise exception (?) if the surface is going uphill from the start
stop the simulation when the particle reaches xlim
check if particle is in contact with the surface
'''
