#!/usr/bin/env python
"""
Example script to test beam interpolation.

Call using the following to include XLA JIT compilation:

(CPU version)
$ TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" ./beam_test.py

(GPU version)
$ TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./beam_test.py
"""
import numpy as np
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
from pyuvdata import UVBeam
import time, sys
import TensorVis as tv

np.random.seed(10)

FLOAT_TYPE = tf.float64
DEBUG = True

# Default simulation settings
Nlsts = 4
Nfreqs = 8
Nptsrc = 500

# Debugging and check for GPU
if DEBUG:
    device_name = tf.test.gpu_device_name()
    print("GPU device:", device_name)
    tf.config.list_physical_devices()


# Load beam data from file
uvb = UVBeam()
uvb.read_beamfits("/home/phil/hera/hera_pspec/hera_pspec/data/HERA_NF_efield.beamfits")
uvb.interpolation_function = 'healpix_simple' #'az_za_simple'
uvb.freq_interp_kind = 'linear'
freq_range = (np.min(uvb.freq_array), np.max(uvb.freq_array))


# Frequency and time arrays
freqs = tf.convert_to_tensor(np.linspace(100., 120., Nfreqs)*1e6, 
                             dtype=FLOAT_TYPE)
lsts = tf.convert_to_tensor([2.4030742+0.001*i for i in range(Nlsts)], 
                            dtype=FLOAT_TYPE)

# Generate randomly-placed point sources
ra = tf.convert_to_tensor(np.random.uniform(0., np.pi, Nptsrc), 
                          dtype=FLOAT_TYPE)
dec = tf.convert_to_tensor(np.random.uniform(-0.5*np.pi, 0.5*np.pi, Nptsrc), 
                           dtype=FLOAT_TYPE)
flux = tf.convert_to_tensor(10.**np.random.uniform(-8., -6., Nptsrc), 
                            dtype=FLOAT_TYPE)
spectral_idx = tf.convert_to_tensor(-2.7*np.ones(Nptsrc), 
                                    dtype=FLOAT_TYPE)

# Calculate alt and az coordinates of point sources
az, za = tv.coords.eq_to_az_za(ra, dec, lsts[0])

# Construct interpolation grid for beam
t0 = time.time()
grid_re, grid_im = tv.beams.construct_beam_grid(uvb, 
                                                Nza=100, Naz=101, freq=None, 
                                                axis=0, feed=0, spw=0, 
                                                dtype=tf.float64)
tf.print("Constructing beam grid took %3.3f sec" % (time.time() - t0))
tf.print(grid_re.shape)

# Interpolate at point source positions
t0 = time.time()
bb = tv.beams.interpolate_beam(grid_re, grid_im, za, az, freqs, 
                               freq_range=freq_range, dtype=tf.float64)

tf.print("Interpolating beam took %3.3f sec" % (time.time() - t0))
tf.print("Interpolated beam:", bb.shape)
tf.print("Point sources:", az.shape)
tf.print("Freqs:", freqs.shape)

#import pylab as plt
##plt.matshow(bb.numpy().real, aspect='auto')
#plt.plot(bb.numpy().real.T)
#plt.show()

