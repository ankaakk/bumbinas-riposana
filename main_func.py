import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import scienceplots
import scipy.integrate as sci

plt.style.use(['science', 'grid', 'notebook'])
plt.rcParams['animation.html'] = 'html5'


def solve(func, do_animation=False, do_plots=True, tmax=20, fps=100, mukin=0.2, mustat=0.3, crf=0.01, r_particle=2., icoef=2/5, method='RK23'):
    def get_surface(func, xlim, r_particle):
        
        Npts = 10001                                   # number of points
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

        '''trajectory is always at a distance r_particle from the surface'''
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
        x_init = traj[i_init,0]
        y_init = traj[i_init,1]
        angle_init = traj[i_init, 4]
        v_init = 0.
        stateinitial = np.asarray([x_init, y_init, v_init*np.cos(angle_init), v_init*np.sin(angle_init), 0.0, 0.0, 0.0, 0.0])
        return stateinitial


    def normal(angle, cpetal):                                  # normal reaction force
        normal_mod = g * np.cos(angle) + cpetal      # centripetala contains a sign (from the radius sign)
        if normal_mod < 0.0:
            normal_mod = 0.0
        return normal_mod


    def prepare_friction(angle, normal_mod):
        fkin = mukin * normal_mod
        fstatmax = mustat * normal_mod
        fstat = np.abs(1/(icoef+1) * (g) * np.sin(angle)) * np.sign(normal_mod)     # sign to set to 0 if no contact
        if fstat > fstatmax:
            fstat = fstatmax
        return fstat, fkin          # f0, f1 for sigmoid


    def sigmoid(f0, f1, v):
        '''f0 is friction for v=0, f1 is friction for v!=0'''
        vswitch = 1e-1
        pnorm = 0.95                                             # fraction of fstat-fkin
        vnorm = vswitch / np.log((1+pnorm)/(1-pnorm))            # np.log() is ln
        vtol = 1e-2                                              # below this vslip is considered 0 (in def friction)
        
        f = -2*(f0 - f1) / (1 + np.exp(-v/vnorm)) + 2*f0 - f1
        # if np.abs(v) < vtol:                                   # manually ground to f0 for smal v
            # f = f0
        return f


    def Derivatives(t, state, r_particle, mukin, mustat, crf, icoef, traj, g):
        g = 9.81
        b = r_particle*crf/np.sqrt(1+crf**2)

        x  = state[0]
        y  = state[1]
        vx = state[2]
        vy = state[3]
        v  = np.linalg.norm([vx, vy])
        phi = state[4]                                  # rotational angle
        phidot = state[5]                               # rotational velocity
        workdot = state[6]                              # work done by friction if slipping, per unit mass

        index = np.argmin(np.abs(traj[:,0]-x))          # index of x trajectory array which is closest to x of cm
        angle  = traj[index, 4]                         # slope at this x
        radius = traj[index, 5]
        
        if radius == np.inf:
            centripetala = 0.0                                     # if curvature radius in infinite, surface is flat   
        else:           
            centripetala = v**2 / radius                           # radius is positive if concave up (function min)


        vtan = vx * np.cos(angle) + vy * np.sin(angle)      # mag same as v, but has sign along the slope
        # vdir = np.sign(vtan)                                # positive to the right
        vslip = vtan + (phidot*r_particle)                  # has a sign along the slope
        vslipabs = np.abs(vslip)
        vslipdir = np.sign(vslip)


        gravitya = np.asarray([0.0, -g])                     # a = acceleration

        normal_mod = normal(angle=angle, cpetal=centripetala)                              
        normala = np.asarray([-normal_mod * np.sin(angle), normal_mod * np.cos(angle)])

        dfriction_mod = sigmoid(*prepare_friction(angle, normal_mod), v=vslipabs)
        dfriction_signed = dfriction_mod * -vslipdir

        rfriction_mod = sigmoid(f0=normal_mod*crf, f1=0.0, v=vslipabs)
        rfriction_signed = rfriction_mod * -vslipdir
        
        # friction_mod = dfriction_mod + rfriction_mod
        friction_signed = dfriction_signed + -rfriction_signed
        frictiona = np.asarray([friction_signed * np.cos(angle), friction_signed * np.sin(angle)])
        
        accel = gravitya + normala + frictiona


        phiddot = (dfriction_signed) / (icoef*r_particle)               # positive counterclockwise
        phiddot += (rfriction_signed) / (icoef*r_particle)               # positive counterclockwise
        phiddot += (normal_mod*b) / (icoef * r_particle**2) * (-np.sign(phidot))


        dworkdot = (dfriction_mod+rfriction_mod) * vslipabs
        rworkdot = normal_mod * b * np.abs(phidot)


        statedot = np.asarray([vx, vy, accel[0], accel[1], phidot, phiddot, dworkdot, rworkdot]) 
        return statedot

    # ----------------------------------------------------------------------------------------------------------

    g = 9.81
    xlim, ylim = 100, 100                            # m, siulation box size
    surface, traj = get_surface(func, xlim, r_particle)

    ntimes = int(tmax * fps-1)                                                         # amount of time points
    tout = np.linspace(0, tmax, ntimes)         
    dt = tout[1]-tout[0]

    stateinitial = get_stateinitial(i_init=0)                                                                    # RK45, RK23, DOP853, BDF, Radau (fails), LSODA
    stateout = sci.solve_ivp(Derivatives, y0=stateinitial,
                            t_span=(0.0, tmax), t_eval=tout, method=method, max_step=dt,
                            args=(r_particle, mukin, mustat, crf, icoef, traj, g))
    y = stateout.y[1]
    phidot = stateout.y[5]
    v = np.asarray([np.linalg.norm([stateout.y[2,i], stateout.y[3,i]]) for i in range(len(stateout.t))])
    energy = g*y + v**2/2 + icoef*r_particle**2*phidot**2/2

    print(stateout.message)
    print(ntimes, np.shape(stateout.y))

# ----------------------------------------------------------------------------------------------------------

    if do_plots:
        ''' final position'''
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.plot(surface[:,0],surface[:,1], 'k-')
        ax.plot(traj[:,0],traj[:,1], 'k--')
        ax.scatter(stateout.y[0,-1],stateout.y[1,-1], c='k')
        ax.set_title(f'Final Position, method={method}, {tmax}s, {fps} fps')
        ax.set_xlabel(r'$x$, [m]')
        ax.set_ylabel(r'$y$, [m]')
        plt.savefig('graphs/final_pos.png')


        '''work and energy'''
        fig, axs = plt.subplots(1, 2, figsize=(12,5), tight_layout=True)
        ax = axs[0]
        ax.plot(stateout.t, energy/energy[0], label='ball')
        ax.plot(stateout.t, (energy+stateout.y[6]+stateout.y[7])/energy[0], label='system')
        ax.axhline(1, c='red', label='init', zorder=1)
        ax.set_title('energy')
        ax.set_xlabel(r'$t$, [s]')
        ax.set_ylabel(r'$E$ / $E_0$')
        ax.legend()

        ax = axs[1]
        ax.plot(stateout.t, stateout.y[6]/energy[0], c='orange', ls='-.', label='dry')
        ax.plot(stateout.t, stateout.y[7]/energy[0], c='tomato', ls='-.', label='roll')
        ax.plot(stateout.t, (stateout.y[7]+stateout.y[6])/energy[0], c='darkred', label='total')
        ax.set_title('work')
        ax.set_xlabel(r'$t$, [s]')
        ax.set_ylabel(r'$W$ / $E_0$')
        ax.legend()
        plt.savefig('graphs/energy.png')


        '''trans and rot velocities'''
        plt.figure()
        vt = np.sqrt(stateout.y[2]**2 + stateout.y[3]**2)*np.sign(stateout.y[2])
        vr = -stateout.y[5]*r_particle
        plt.plot(stateout.t, vt, label='trans')
        plt.plot(stateout.t, vr, label='ang')
        plt.plot(stateout.t, vt-vr, label='difference', ls='--')
        plt.axhline(0, color='k', zorder=0)
        plt.axvline(stateout.t[1], c='k', zorder=0)
        plt.legend()
        plt.title('Translational vs Angular velocity')
        plt.xlabel(r'$t$, [s]')
        plt.ylabel(r'$v$, [m/s]')
        plt.savefig('graphs/velocities.png')

        print('Plots are ready.')

# ----------------------------------------------------------------------------------------------------------

    if do_animation:
        def animate(i):
            global xlim, ylim

            cx, cy = (sol_y_ani[0,i], sol_y_ani[1,i])
            circ.set_center((cx, cy))
            extent = [cx - r_particle, cx + r_particle, cy - r_particle, cy + r_particle]
            transform = mpl.transforms.Affine2D().rotate_around(cx, cy, sol_y_ani[4,i]) + ax.transData
            im.set_extent(extent)
            im.set_transform(transform)
            im.set_clip_path(circ)

            line.set_data(sol_y_ani[0,:i], sol_y_ani[1,:i])

            time_text.set_text(time_template % (sol_t_ani[i]))                                   # clock in the figure
            vel_text.set_text(vel_template % (np.sqrt(sol_y_ani[2,i]**2 + sol_y_ani[3,i]**2)*np.sign(sol_y_ani[2,i])))
            angvel_text.set_text(angvel_template % (-sol_y_ani[5,i]*r_particle))
            disp_text.set_text(disp_template % np.sqrt((sol_y_ani[0,i]-sol_y_ani[0,0])**2 + (sol_y_ani[1,i]-sol_y_ani[1,0])**2))
            energy_text.set_text(energy_template % ((energy_ani[i]+sol_y_ani[6,i]+sol_y_ani[7,i]-energy_ani[0])/energy_ani[0]*100))

            return (circ)

        index_arr = np.where(stateout.y[0]>=xlim)[0]        # unpack function output
        if not len(index_arr)==0:                           # if leaves the box
            index = index_arr[0]                            # first point outside the box
            tmax = stateout.t[index]
            sol_t = stateout.t[:index]
            sol_y = stateout.y[:,:index]
            energy_red = energy[:index]
        else:                                               # if stays within the box, no changes
            sol_t = stateout.t
            sol_y = stateout.y
            energy_red = energy

        fps_ani = 20                        # originally stateout has a larger fps, no need to fraw that much
        n_ani = int(fps/fps_ani)
        sol_t_ani = sol_t[::n_ani]          # for example, take every fifth point from solution
        sol_y_ani = sol_y[:, ::n_ani]
        energy_ani = energy_red[::n_ani]

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
        ax.plot(traj[:,0], traj[:,1], lw=1, ls='--', c='coral')                                    # trajectory along the surface
        line, = ax.plot([], [], 'o-', lw=1, c='crimson', ms=4, alpha=0.8)                                        # actual path

        time_template      = 't   = %.1f s'                                                     # clock in the animation
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

        ani = animation.FuncAnimation(fig, animate, repeat=True, frames=int(fps_ani*tmax))
        writer = animation.PillowWriter(fps=fps_ani,                                           # 1s of gif = 1s of model
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save('animation.gif', writer=writer)
        print('Animation is ready.')

    return stateout, energy, surface, traj