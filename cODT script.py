import CrossedODTLibrary as cODT
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


plt.close('all')

kB = 1.38e-23

P1 = 75
P2 = 73
phi1 = 5
phi2 = -phi1

#%%
gridExtent = [2000e-6, 1500e-6]
x0, y0 = gridExtent[0]/2, gridExtent[1]/2
res = 2000

x = np.linspace(0, gridExtent[0], res)
y = np.linspace(0, gridExtent[1], res)

# First pass
w0y_1, w0z_1 = cODT.Waist_FirstPass(P1)
I1, X, Y = cODT.Rotated_Astig_Gaussian2D(w0y_1, w0z_1, P1, phi1, x0, y0, gridExtent, res=res)

# Second pass
w0y_2, w0z_2 = cODT.Waist_FirstPass(P2)
I2, _, _ = cODT.Rotated_Astig_Gaussian2D(w0y_2, w0z_2, P2, phi2, x0, y0, gridExtent, res=res)

# Total intensity
Itot = I1 + I2

# Dipole potential in K
Utot = cODT.DipolePotential(Itot)

cODT.MakeFigure2D(X, Y, -Utot/kB, scale=1e6, title='Trap Depth ($\mu$K)')

#%% Figure with slices

deltaX = 500e-6
deltaY = 25e-6
POI_x = np.array([x0-deltaX, x0, x0+deltaX])
POI_y = np.array([y0-deltaY, y0, y0+deltaY])

cODT.Plot_Figure2DwithSlices(-Utot/kB, x, y, POI_x, POI_y, levels=50)

#%% Trap depth vs power

P_firstpass = np.linspace(0, 75, 20)
P_secondpass = 0.95 * P_firstpass

w0y_1, w0z_1 = cODT.Waist_FirstPass(P_firstpass)
w0y_2, w0z_2 = cODT.Waist_FirstPass(P_secondpass)

U_list = []
for j in range(len(P_firstpass)):
    I1, _, _ = cODT.Rotated_Astig_Gaussian2D(w0y_1[j], w0z_1[j], P_firstpass[j], phi1, x0, y0, gridExtent, res=res)
    I2, _, _ = cODT.Rotated_Astig_Gaussian2D(w0y_2[j], w0z_2[j], P_secondpass[j], phi2, x0, y0, gridExtent, res=res)
    
    I = np.max(I1 + I2)
    U = cODT.DipolePotential(I)
    U_list.append(U)
    print(-U/kB * 1e6)
    
U_list = np.array(U_list)
T_list_uK = -U_list / kB * 1e6
slope = (np.max(T_list_uK) - np.min(T_list_uK)) / (np.max(P_firstpass) - np.min(P_firstpass))


plt.figure(figsize=(5,4))
plt.scatter(P_firstpass, T_list_uK, label='U/kB')
plt.plot(P_firstpass, P_firstpass*slope, label=f'm = {slope:.2f} $\mu$K/W')

plt.xlabel('First pass power (W)')
plt.ylabel('Trap depth ($\mu$K)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()