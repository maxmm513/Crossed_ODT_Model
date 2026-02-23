import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import scipy.constants as Const

def RotateBeam(I, angle_deg, reshape=False):
    arr = rotate(I, angle=angle_deg, reshape=reshape, order=1, mode='nearest')
    return arr

def Astig_Gaussian2D(w0y, w0z, Power, x0, y0, phi_deg, gridExtent, lamb=1064e-9, res=1000):
    
    x = np.linspace(0, gridExtent[0], res)
    y = np.linspace(0, gridExtent[1], res)
    X, Y = np.meshgrid(x, y)

    #convert phi from degrees to radians
    phi = phi_deg * np.pi/180    

    # Rayleigh ranges
    x_RY = np.pi * w0y**2 / lamb
    x_RZ = np.pi * w0z**2 / lamb
    
    Xrot = (X-x0)*np.cos(phi) + (Y-y0)*np.sin(phi)
    Yrot = -(X-x0)*np.sin(phi) + (Y-y0)*np.cos(phi)

    # beam widths
    wY = w0y * np.sqrt(1 + Xrot**2 / x_RY**2)
    wZ = w0z * np.sqrt(1 + Xrot**2 / x_RZ**2)

    # intensity distribution
    I = 2 * Power / (np.pi * wY * wZ) * np.exp(-2 * Yrot**2 / wY**2)
    # I = np.sqrt(2/np.pi) * Power * wZ/wY * np.exp(-2*Y**2/wY**2)

    return I, X, Y

def Astig_Gaussian1D(w0y, w0z, Power, phi_deg, X, x0, lamb=1064e-9):
    
    phi = phi_deg*np.pi/180
    
    # Rayleigh ranges
    x_RY = np.pi * w0y**2 / lamb
    x_RZ = np.pi * w0z**2 / lamb

    # beam widths
    wY = w0y * np.sqrt(1 + ((X-x0)*np.cos(phi))**2 / x_RY**2)
    wZ = w0z * np.sqrt(1 + ((X-x0)**2 *np.cos(phi))**2 / x_RZ**2)
    
    # intensity
    I = 2*Power/(np.pi * wY * wZ) * np.exp(-2*(X-x0)**2*np.sin(phi)**2 / wY**2)
    return I
    

def Rotated_Astig_Gaussian2D(w0y, w0z, Power, angle_deg, x0, y0, gridExtent, res=1000):
    
    # unrotated beam
    I, X, Y = Astig_Gaussian2D(w0y, w0z, Power, x0, y0, angle_deg, gridExtent, res=res)

    # rotate by angle_deg
    I_rot = RotateBeam(I, angle_deg)

    return I_rot, X, Y

# calculate U_dip for given intensity distribution
def DipolePotential(I):
    
    gamma = 2*np.pi*5.8724e6 # Hz
    freq_0 = 446.789e12 # Hz, transition frequency for Li
    omega_0 = 2*np.pi*freq_0
    freq_IR = Const.c / (1064e-9) # frequency of IR beams
    omega_IR = 2*np.pi*freq_IR
    
    U0 = -3*np.pi*Const.c**2/(2*omega_0**3) * (gamma/(omega_0-omega_IR) + gamma/(omega_0+omega_IR))
    
    U = U0*I
    
    return U

# waist vs. power (returns waist in m)
def Waist_FirstPass(P):
    w0y_um = (0.05236877)*P + (51.24607532)
    w0z_um = (-1.53490908e-02)*P + (4.76563043e+01)
    return w0y_um*1e-6, w0z_um*1e-6

def Waist_SecondPass(P):
    w0y_um = (0.0782928)*P + (51.15435204)
    w0z_um = (-1.19900284e-02)*P + (4.87809228e+01)
    return w0y_um*1e-6, w0z_um*1e-6



def MakeFigure2D(X, Y, Z, scale, title):
    
    Z_scaled = Z*scale
    
    # Convert axis  meters to um
    extent_um = [X.min()*1e6, X.max()*1e6, Y.min()*1e6, Y.max()*1e6]

    plt.figure(figsize=(5, 3))
    contour = plt.contourf(X*1e6, Y*1e6, Z_scaled, levels=100, cmap='jet', origin='lower')
    # plt.imshow(Z, extent=extent_um, origin='lower', cmap='hot')
    # vals = [0, 3, 6, 9, 12]
    cbar = plt.colorbar(contour, label=title)
    # cbar.ax.tick_params(labelsize=14)
    # cbar.set_label(title, fontsize=16)
    plt.xlabel('x ($\mu$m)', fontsize=16)
    plt.ylabel('y ($\mu$m)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.axis('equal')
    plt.tight_layout()
    
def add_noise(Itot, SNR):
    signal_power = np.mean(Itot**2)
    noise_power = signal_power / SNR
    noise = np.random.normal(0, np.sqrt(noise_power), size=Itot.shape)
    return Itot + noise


#%%
from scipy.interpolate import RegularGridInterpolator

def Slices_Horiz(X, Y, Z, POI):
    y_vals = Y[:, 0]
    x_vals = X[0, :]

    interpolator = RegularGridInterpolator((y_vals, x_vals), Z, method='linear', bounds_error=False, fill_value=0)

    slices = []
    for y_pos in POI:
        pts = np.column_stack((np.full_like(x_vals, y_pos), x_vals))
        slice_vals = interpolator(pts)
        slices.append(slice_vals)

    return x_vals, np.array(slices)


def Slices_Vert(X, Y, Z, POI):

    y_vals = Y[:, 0]
    x_vals = X[0, :]

    interpolator = RegularGridInterpolator((y_vals, x_vals), Z, method='linear', bounds_error=False, fill_value=0)

    slices = []
    for x_pos in POI:
        pts = np.column_stack((y_vals, np.full_like(y_vals, x_pos)))
        slice_vals = interpolator(pts)
        slices.append(slice_vals)

    return y_vals, np.array(slices)

def Slices_Plot(vals, slices, POI, sliceType, scale=1):
    
    slices = slices*scale
    
    plt.figure(figsize=(5, 4))

    for i, y_pos in enumerate(POI):
        plt.plot(vals * 1e6, slices[i], label=f'y = {y_pos*1e6:.1f} $\mu$m')
    
    if sliceType == 'horiz':
        axis = 'x'
        title = 'Horizontal slice'
        
    elif sliceType == 'vert':
        axis = 'y'
        title = 'Vertical slice'
    
    plt.xlabel(f'{axis} ($\mu$m)')
    plt.ylabel('Temperature')
    plt.title(title)
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    
#%%

hbar = Const.hbar
kB = Const.Boltzmann

mass = 9.988431e-27 # kg
gamma = 2*np.pi*5.8724e6 # Hz

freq_0 = 446.789e12 # Hz, transition frequency for Li
lamb = Const.c / freq_0 # m
omega_0 = 2*np.pi*freq_0


lamb_IR = 1064e-9
freq_IR = Const.c / lamb_IR # frequency of IR beams
omega_IR = 2*np.pi*freq_IR

#data for 2s to 3p transition
lamb3p = 323e-9
gamma3p = 1.002e6 #/s
omega_03p = 2*np.pi * Const.c/lamb3p


def TrapDepth_cODT(P1_W, P2_W, angle_deg):
    
    # first pass waist (m)
    w_0Y_1, w_0Z_1 = Waist_FirstPass(P1_W)
    
    # second pass waist (m)
    w_0Y_2, w_0Z_2 = Waist_FirstPass(P2_W)

    # areas
    w_bar_1 = w_0Y_1 * w_0Z_1
    w_bar_2 = w_0Y_2 * w_0Z_2
    
    # crossing angle
    phi = angle_deg/2 * np.pi/180
    
    # Rayleigh ranges
    z_RY_1 = np.pi * w_0Y_1**2 / lamb
    z_RZ_1 = np.pi * w_0Z_1**2 / lamb
    z_RY_2 = np.pi * w_0Y_2**2 / lamb
    z_RZ_2 = np.pi * w_0Z_2**2 / lamb
    
    # inverse square sums
    rho_1 = (1/(z_RY_1**2) + 1/(z_RZ_1**2))
    rho_2 = (1/(z_RY_2**2) + 1/(z_RZ_2**2))
    
    
    # In order to account for the coupling between the x and y coordinates, we mut diagonalize the 2x2 matrixs that is used to for the x^2, y^2 and xy terms in the
    # intensity profile
    a = (P1_W/w_bar_1) * ((rho_1/4) * np.cos(phi)**2 + (1/w_0Y_1**2) * np.sin(phi)**2) + (P2_W/w_bar_2) * ((rho_2 / 4)*np.cos(phi)**2 + (1 / w_0Y_2**2) * np.sin(phi)**2)
    b = (np.cos(phi) * np.sin(phi) * (rho_1 - (4 / w_0Y_1**2) + (4 / w_0Y_2**2) - rho_2) * ((P1_W/w_bar_1) + (P2_W/w_bar_2))) / 2
    c = (P1_W/w_bar_1) * ((rho_1/4)*np.sin(phi)**2 + (1/w_0Y_1**2) * np.cos(phi)**2) + (P2_W/w_bar_2) * ((rho_2/4)*np.sin(phi)**2 + (1/w_0Y_2**2) * np.cos(phi)**2)
    
    # lambda 1 and 2 are the eigen values of the diagonalization matrix
    # lambda 3 is the pre-factor for the z^2 term
    # I placed in this block because of the similarity of use to the other two variables
    lambda_1 = 0.5*(-np.sqrt(a**2 - 2 * a * c + 4 * b**2 + c**2) + a + c)
    lambda_2 = 0.5*(np.sqrt(a**2 - 2 * a * c + 4 * b**2 + c**2) + a + c)
    lambda_3 = (P1_W / (w_bar_1 * (w_0Z_1**2))) + (P2_W / (w_bar_2 * (w_0Z_2**2)))
    
    # the prefactor of the dipole potential
    U_pre = (6 * Const.c**2) / (omega_0**3) * ((gamma / (omega_0 - omega_IR) + (gamma / (omega_0 + omega_IR))))
    
    # angular trap frequencies
    omega_x = np.sqrt((2 * U_pre * lambda_1) / mass)
    omega_y = np.sqrt((2 * U_pre * lambda_2) / mass)
    omega_z = np.sqrt((2 * U_pre * lambda_3) / mass)
    
    # trap frequencies
    freq_x = omega_x / (2 * np.pi) # axial
    freq_y = omega_y / (2 * np.pi) # transverse horizontal
    freq_z = omega_z / (2 * np.pi) # transverse vertical
    
    intensity = ((P1_W / w_bar_1) + (P2_W / w_bar_2))
    
    # trap depth (J)
    U = (-3 * np.pi * (Const.c**2) / (2 * (omega_0)**3)) * ((gamma / (omega_0 - omega_IR) + (gamma / (omega_0 + omega_IR)))) * (2 / np.pi) * intensity
    T = -U/kB
    
    return U, T, [freq_x, freq_y, freq_z], [omega_x, omega_y, omega_z]

def Potential_Harmonic(x, x0, omegaX):
    return 0.5 * mass * omegaX**2 * (x-x0)**2

def Potential_AstigGaussian1D(w0y, w0z, Power, phi, X, x0, lamb=1064e-9):
    
    # Rayleigh ranges
    x_RY = np.pi * w0y**2 / lamb
    x_RZ = np.pi * w0z**2 / lamb

    # beam widths
    wY = w0y * np.sqrt(1 + ((X-x0)*np.cos(phi))**2 / x_RY**2)
    wZ = w0z * np.sqrt(1 + ((X-x0)*np.cos(phi))**2 / x_RZ**2)
    
    # intensity
    I = 2*Power/(np.pi * wY * wZ) * np.exp(-2*(X-x0)**2*np.sin(phi)**2 / wY**2)
    return I

def DipolePotential__1(I):
    
    U0 = -3*np.pi*Const.c**2/(2*omega_0**3) * (gamma/(omega_0-omega_IR) + gamma/(omega_0+omega_IR))
    
    U = U0*I
    
    return U

def thermal_lambda(T):
    return np.sqrt(2*np.pi*hbar**2/(mass*kB*T))

def mu_global(T, N, omegas):
    beta = 1/(kB*T)
    lambTH = thermal_lambda(T)
    
    mu0 = 1/beta * np.log(N*omegas[0]*omegas[1]*omegas[2]*lambTH**3 * (beta*mass/(2*np.pi))**(3/2))
    return mu0


def potential1D_classical(columnDensity_1D, N, sigma, omegaX):
    
    A = N / (sigma*np.sqrt(2*np.pi))
    print(A)
    
    beta = 1/(mass * sigma**2 * omegaX**2)
    
    Poten = - np.log(columnDensity_1D/A) / beta
    
    return Poten

#%% Absorption Image Analysis 
from scipy.optimize import curve_fit

def shiftCD(columnDensity1D):
    shiftVal = np.abs(np.min(columnDensity1D))
    shiftedArr = columnDensity1D + shiftVal
    return shiftedArr


def GetColumnDensity(columnDensity, dx):
    
    CD1Dx = shiftCD(np.nansum(columnDensity, axis=0) * dx / 1e6**2)
    CD1Dy = shiftCD(np.nansum(columnDensity, axis=1) * dx / 1e6**2)
    
    # CD1Dx = np.nansum(columnDensity, axis=0) * dx / 1e6**2
    # CD1Dy = np.nansum(columnDensity, axis=1) * dx / 1e6**2

    x_real = np.arange(columnDensity.shape[1]) * dx
    y_real = np.arange(columnDensity.shape[0]) * dx
    
    return [x_real, y_real], CD1Dx, CD1Dy

def Gauss1D(x, xc, sigX, A, offset):
    G = A * np.exp(-(x-xc)**2 / (2*sigX**2)) + offset
    return G

def FitGaussian_1D(arrayX, arrayY, shapeType):
               
    sigGuess = 30
    offset = 0
    
    if shapeType=='negative':     
        min_Y = np.min(arrayY)
        min_Y_index = np.argmin(arrayY)
        min_X_index = arrayX[min_Y_index]
        guess = [min_X_index, sigGuess, min_Y, offset]
        
    elif shapeType=='positive':
        max_Y = np.max(arrayY)
        max_Y_index = np.argmax(arrayY)
        max_X_index = arrayX[max_Y_index]
        guess = [max_X_index, sigGuess, max_Y, offset]

        
    params,_ = curve_fit(Gauss1D, arrayX, arrayY, p0=guess)
    
    xFit = np.linspace(min(arrayX), max(arrayX), 1000)
    yFit = Gauss1D(xFit, *params)
    
    return params, [xFit,yFit]