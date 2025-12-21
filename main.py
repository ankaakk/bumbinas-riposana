import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import scienceplots
import scipy.integrate as sci
import time 
from tkinter import *
from tkinter import messagebox
import sympy as sp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


plt.style.use(['science', 'grid', 'notebook'])
plt.rcParams['animation.html'] = 'html5'
start_time = time.time()
g = 9.81                                         # m/s^2
xlim, ylim = 100, 100                            # m, siulation box size
icoef = 2/5



master = Tk()
master.title('User input') 
master.geometry('1200x900') #size

current_function = None
coeff_entries = {}

def function_select(f_type):

    global current_function

    current_function = f_type
    coefficients()

f_frame = Frame(master)
f_frame.pack(pady=5)

Button(f_frame, text='Linear', command=lambda: function_select('linear')).pack(side='left', padx=5)
Button(f_frame, text='Quadratic', command=lambda: function_select('quadratic')).pack(side='left', padx=5)
Button(f_frame, text='Convex circle', command=lambda: function_select('convex')).pack(side='left', padx=5)
Button(f_frame, text='Concave circle', command=lambda: function_select('concave')).pack(side='left', padx=5)
Button(f_frame, text='Sinudoid', command=lambda: function_select('sinusoid')).pack(side='left', padx=5)
Button(f_frame, text='Log', command=lambda: function_select('log')).pack(side='left', padx=5)
Button(f_frame, text='Exponent', command=lambda: function_select('exponent')).pack(side='left', padx=5)
Button(f_frame, text='Catenary', command=lambda: function_select('catenary')).pack(side='left', padx=5)

coeff_frame = Frame(master)
coeff_frame.pack(pady=5)

def add_coeff(coeff):
    Label(coeff_frame, text=coeff).pack()
    entry = Entry(coeff_frame)
    entry.pack()
    coeff_entries[coeff] = entry

def coefficients():
    for i in coeff_frame.winfo_children():
        i.destroy()
    coeff_entries.clear()

    match current_function:
        case 'linear':
            add_coeff('a')
            add_coeff('b')
        case 'quadratic':
            add_coeff('a')
            add_coeff('b')
            add_coeff('c')
        case 'convex': ###
            add_coeff('a')
            add_coeff('b')
            add_coeff('d')
        case 'concave': ###
            add_coeff('a')
        case 'sinusoid':
            add_coeff('a')
            add_coeff('b')
        case 'log':
            add_coeff('a')
            add_coeff('b')
        case 'exponent':
            add_coeff('a')
            add_coeff('b')
        case 'catenary': ###
            add_coeff('a')


def get_function():

    try:
        coeff = {a: float(b.get()) for a, b in coeff_entries.items()}
    except ValueError:
        return None

    match current_function:
        case 'linear':
            return lambda x: coeff['a'] * x + coeff['b']
        case 'quadratic':
            return lambda x: coeff['a'] * x**2 + coeff['b'] * x + coeff['c']
        case 'convex':
            return lambda x: -np.sqrt(50**2 - (x-50)**2) + 70 ###
        case 'concave':
            return lambda x: np.sqrt(100**2 - (x)**2)-20   ###
        case 'sinusoid':
            return lambda x: -coeff['a'] * np.sin(x*np.pi/100) + coeff['b'] 
        case 'log':
            return lambda x: coeff['a']* np.log(-x+coeff['b'])
        case 'exponent':
            return lambda x: coeff['a']*np.exp(coeff['b']*x)
        case 'catenary':
            return lambda x: np.cosh(0.05*(x-50)) + 50 ###

#Label(f_frame, text='Static ').grid(row=1) 
#Label(f_frame, text='Kinetic ').grid(row=2)
#Label(f_frame, text='Radius ').grid(row=3)

params_frame = Frame(master)
params_frame.pack(pady=5)

#function = Entry(master) 
Label(params_frame, text='Static ').pack(side='left') 
Label(params_frame, text='Kinetic ').pack(side='left')
Label(params_frame, text='Radius ').pack(side='left')

params_frame_entries = Frame(master)
params_frame_entries.pack(pady=5)

mu_static = Entry(params_frame_entries)
mu_static.pack(side='left', padx=5)
mu_kinetic = Entry(params_frame_entries)
mu_kinetic.pack(side='left', padx=5)
radius = Entry(params_frame_entries)
radius.pack(side='left', padx=5)

#place entry boxes next to inputs
#function.grid(row=0, column=1)
#mu_static.grid(row=1, column=1) 
#mu_kinetic.grid(row=2, column=1)
#radius.grid(row=3, column=1)

#def temp_plot():
    
plot_frame = Frame(master)
plot_frame.pack(pady=10)


#fig, ax = plt.subplots(figsize=(5,4))
fig = Figure(figsize=(8, 5))
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack()

def update_plot():
    f = get_function() 

    try: 
        f = get_function()
    except ValueError:
        messagebox.showerror("Error", "Invalid coefficients")
        return

    x = np.linspace(0, xlim, 500)   #could input x and y as choosable for i
    y = f(x)
    
    

    ax.clear()
    ax.plot(x, y)
    ax.grid(True)
    canvas.draw()

Button(master, text="Plot preview", command=update_plot).pack(pady=5)

def save_changes():
    '''
    reads inputs from entry boxes, checks if all values have been
    input, converts to float, closes window
    '''
    global mustat, mukin, r_particle, func


    #func_str = function.get()
    func = get_function()
    mustat = mu_static.get()
    mukin = mu_kinetic.get()
    r_particle = radius.get()

    #x = sp.Symbol('x')
    #f = sp.Function('f')(x)
    #func_sp = sp.sympify(func_str)              #string to sympify expression
    #func = sp.lambdify(x, func_sp, 'numpy')      #to numerical expression

    if not mukin or not mustat or not xlim or not ylim or not r_particle:
        messagebox.showerror('Error','Need all values') 
        return
    else:
        messagebox.showinfo('Saved', 'Values saved') 

    mustat = float(mustat) 
    mukin = float(mukin)
    r_particle = float(r_particle)

    master.destroy() 

def confirm_exit():
    '''
    asks for exit comfirmation
    '''
    response = messagebox.askyesnocancel('Confrim choices') 
    if response:
        save_changes()        

confirm_button = Button(master, text='Confirm', command=confirm_exit).pack(pady=5) 
#confirm_button.grid(row=6,column=1)

mainloop()  #infinite loop, keeps window open

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
