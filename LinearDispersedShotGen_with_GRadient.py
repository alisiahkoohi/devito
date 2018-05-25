import numpy as np
import h5py
from argparse import ArgumentParser
import os
from devito.logger import info
from devito import TimeFunction, clear_cache, configuration, Function
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Model, RickerSource, Receiver, TimeAxis
from math import floor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import floor
from scipy import ndimage

# shape = (1601, 401)
origin = (0., 0.)
spacing = (7.5, 7.5)
tn=1000.
nbpml=160
# Define your vp in km/sec (x, z)
tssr = 4
xssr = 1

# strModel = os.path.join('/data/overthrust3d/', 'overthrust3d.hdf5')
# datasetModel= "model"
# fileModel = h5py.File(strModel, 'r')
# vp = 1e-3 * fileModel[datasetModel][:,:]
# vp = vp[:, :]
# vp = vp[0, :, :]
# vp = vp[100:500, :]
#
# shape=vp.shape

vp = np.fromfile(os.path.join('/nethome/asiahkoohi3/Desktop/Ali/opesci-data/data/Simple2D/', 'vp_marmousi_bi'),
            dtype='float32', sep="")
vp = np.reshape(vp, (1601, 401))
vp = vp[601:801, :201]
shape=vp.shape

filter_sigma = (1, 1)
vp0 = ndimage.gaussian_filter(vp, sigma=filter_sigma, order=0)


# plt.show()
model = Model(origin, spacing, shape, 2, vp, nbpml=nbpml)
# Derive timestepping from model spacing
dt = model.critical_dt
t0 = 0.0
nt = int(1 + (tn-t0) / dt)  # Number of timesteps
time = np.linspace(t0, tn, nt)  # Discretized time axis
# nt = 861
tstep = nt - 1

model0 = Model(origin, spacing, shape, 2, vp0, nbpml=nbpml)

dm = model.m.data - model0.m.data
plt.figure(); plt.imshow(np.transpose(dm), vmin=-.1, vmax=.1, cmap="seismic")

num_rec = 401
rec_samp = np.linspace(0., model.domain_size[0], num=num_rec);
rec_samp = rec_samp[1]-rec_samp[0]



kk = 0
mm = 0

xsrc = 1


clear_cache()
time_range = TimeAxis(start=t0, stop=tn, step=dt)
src = RickerSource(name='src', grid=model.grid, f0=0.025,  time_range=time_range, space_order=1, npoint=1)
src.coordinates.data[0, :] = np.array([xsrc*rec_samp, 2*spacing[1]]).astype(np.float32)

# Define receiver geometry (spread across x, just below surface)
# To use the wrapper ignore it
rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=num_rec)
rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=num_rec)
rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

# Create solver object to provide relevant operators
solver = AcousticWaveSolver(model, source=src, receiver=rec, kernel='OT2',
        space_order=20, freesurface=False)
solver_dispersed = AcousticWaveSolver(model, source=src, receiver=rec, kernel='OT2',
        space_order=19, freesurface=False)

u0 = TimeFunction(name="u", grid=model.grid, time_order=20, space_order=2, save=nt)
# uB.data.fill(0.)
deltad, u0, _, _ = solver.born(dm, src=src,  m=model0.m, time=nt-1, save=True)

##############################

# Create solver object to provide relevant operators


grad = Function(name="grad", grid=model.grid)
grad_dispersed =  Function(name="grad", grid=model.grid)


solver.gradient(rec=deltad, u=u0, m=model0.m, grad=grad, time=nt-1)

solver_dispersed.gradient(rec=deltad, u=u0, m=model0.m, grad=grad_dispersed, time=nt-1)




f, axarr = plt.subplots(1, 3)
org = axarr[0].imshow(np.transpose(grad.data[nbpml:-nbpml, nbpml:-nbpml], (1, 0)), vmin=-3, vmax=3, cmap="Greys")
axarr[0].set_title('20 point stencil')
axarr[0].set_xlabel('x axis')
axarr[0].set_ylabel('depth')

axarr[1].imshow(np.transpose(grad_dispersed.data[nbpml:-nbpml, nbpml:-nbpml], (1, 0)), vmin=-.3, vmax=.3, cmap="Greys")
axarr[1].set_title('2 point stencil')
axarr[1].set_xlabel('x axis')
axarr[1].set_ylabel('depth')

axarr[2]. imshow(np.transpose(grad.data[nbpml:-nbpml, nbpml:-nbpml]-grad_dispersed.data[nbpml:-nbpml, nbpml:-nbpml], (1, 0)),
            vmin=-3, vmax=3, cmap="Greys")
axarr[2].set_title('difference')
axarr[2].set_xlabel('x axis')
axarr[2].set_ylabel('depth')

f.subplots_adjust(hspace=.5)
f.colorbar(org, ax=axarr.ravel().tolist())


a = np.array(deltad.data[:,:])
a = a[0:np.shape(deltad.data)[0]:tssr, 0:np.shape(deltad.data)[1]:xssr]
a = np.absolute(np.fft.fftshift(np.fft.fft2(a)))
plt.figure(); plt.imshow(a[floor(np.shape(a)[0]/2):np.shape(a)[0], :], vmin=0, vmax=1e3, aspect=1)

plt.figure()
plt.imshow(deltad.data[0:np.shape(deltad.data)[0]:tssr, 0:np.shape(deltad.data)[1]:xssr], vmin=-.1, vmax=.1, cmap="seismic", aspect=1)
plt.title("linearized data")

plt.show()
