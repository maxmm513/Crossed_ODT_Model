import CrossedODTLibrary as cODT
import matplotlib.pyplot as plt
import numpy as np


gridExtent = [850e-6, 400e-6]
x0, y0 = gridExtent[0]/2, gridExtent[1]/2

P1 = 0.5
w0y_1, w0z_1 = cODT.Waist_FirstPass(P1)
phi1 = 5
I1, X, Y = cODT.Rotated_Astig_Gaussian2D(w0y_1, w0z_1, P1, phi1, x0, y0, gridExtent)

P2 = 0.45
w0y_2, w0z_2 = cODT.Waist_FirstPass(P2)
phi2 = -phi1
I2, _, _ = cODT.Rotated_Astig_Gaussian2D(w0y_2, w0z_2, P2, phi2, x0, y0, gridExtent)

# Total intensity
Itot = I1 + I2


# Dipole potential in K
Utot = cODT.DipolePotential(Itot)

cODT.MakeFigure2D(X, Y, -Utot/(1.38e-23), scale=1e6, title='Trap Depth ($\mu$K)')

#%%

POIs = np.array([-15e-6, 0, 15e-6])
xVals, xSlices = cODT.Slices_Horiz(X, Y, -Utot/1.38e-23 * 1e6, POIs)
yVals, ySlices = cODT.Slices_Vert(X, Y, -Utot/1.38e-23 * 1e6, POIs)

cODT.Slices_Plot(xVals, xSlices, POIs, 'horiz',scale=1e6)
cODT.Slices_Plot(yVals, ySlices, POIs, 'vert')


#%%

P_range = np.linspace(0, 130, 1000)

w0y_1, w0z_1 = cODT.Waist_FirstPass(P_range)
w0y_2, w0z_2 = cODT.Waist_SecondPass(P_range)

fig, ax = plt.subplots(1,2)
ax[0].set_title('First pass')
ax[0].plot(P_range, w0y_1*1e6, label='w0y')
ax[0].plot(P_range, w0z_1*1e6, label='w0z')

ax[1].set_title('Second pass')
ax[1].plot(P_range, w0y_2*1e6, label='w0y')
ax[1].plot(P_range, w0z_2*1e6, label='w0z')