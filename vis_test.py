#!/usr/bin/env python
"""
Example script to run a visibility simulation.

Call using the following to include XLA JIT compilation:

(CPU version)
$ TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" ./vis_test.py

(GPU version)
$ TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./vis_test.py
"""
import numpy as np
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)

import time, sys
import TensorVis as tv
from pyuvdata import UVBeam

np.random.seed(10)

FLOAT_TYPE = tf.float64
DEBUG = True
#NBLOCKS = 5 # Number of blocks to use when calculating point source contrib.

# Beam and catalogue files
beam_path = "/home/phil/hera/hera_pspec/hera_pspec/data/HERA_NF_efield.beamfits"
catalogue_path = "examples/cat_gleam_cut.npy"

# Array spec
#hex_spec = (8,20) # 344 antennas
hex_spec = (3,4)

# Default simulation settings
Nlsts = 4
Nfreqs = 8
Nptsrc = 50
if len(sys.argv) > 1:
    try:
        Nlsts = int(sys.argv[1])
        Nfreqs = int(sys.argv[2])
        Nptsrc = int(sys.argv[3])
    except:
        print("Requires the following arguments: Nlsts Nfreqs Nptsrc [Nblocks]")
        sys.exit(1)
    if len(sys.argv) == 5:
        NBLOCKS = int(sys.argv[4])


# Debugging and check for GPU
if DEBUG:
    device_name = tf.test.gpu_device_name()
    print("GPU device:", device_name)
    tf.config.list_physical_devices()


# Logging
writer = tf.summary.create_file_writer("./logs")
with writer.as_default():

    # Generate hex array and convert to antpos tensor
    antpos_dict = tv.utils.build_hex_array(hex_spec=hex_spec, d=14.6)
    antpos_arr = np.column_stack([val for val in antpos_dict.values()]).T
    antpos = tf.convert_to_tensor(antpos_arr, dtype=FLOAT_TYPE)
    Nants = len(antpos_arr)
    print("Nants:", Nants)

    # Frequency and time arrays
    freqs = tf.convert_to_tensor(np.linspace(100., 120., Nfreqs), 
                                 dtype=FLOAT_TYPE) * 1e6 # Hz
    lsts = tf.convert_to_tensor([2.4030742+0.001*i for i in range(Nlsts)], 
                                dtype=FLOAT_TYPE) # radians

    """
    # Generate randomly-placed point sources
    ra = tf.convert_to_tensor(np.random.uniform(0., np.pi, Nptsrc), 
                              dtype=FLOAT_TYPE)
    dec = tf.convert_to_tensor(np.random.uniform(-0.5*np.pi, 0.5*np.pi, Nptsrc), 
                               dtype=FLOAT_TYPE)
    flux = tf.convert_to_tensor(10.**np.random.uniform(-8., -6., Nptsrc), 
                                dtype=FLOAT_TYPE)
    spectral_idx = tf.convert_to_tensor(-2.7*np.ones(Nptsrc), 
                                        dtype=FLOAT_TYPE)
    """
    # Load GLEAM catalogue, but trim to a given number of entries
    ra, dec, flux, spectral_idx = np.load(catalogue_path)[:Nptsrc,:].T
    print("Point sources after cut:", ra.shape)
    ra = tf.convert_to_tensor(ra*np.pi/180., dtype=FLOAT_TYPE)
    dec = tf.convert_to_tensor(dec*np.pi/180., dtype=FLOAT_TYPE)
    flux = tf.convert_to_tensor(flux, dtype=FLOAT_TYPE)
    spectral_idx = tf.convert_to_tensor(spectral_idx, dtype=FLOAT_TYPE)
    
    # Load beam data from file
    uvb = UVBeam()
    uvb.read_beamfits(beam_path)
    uvb.interpolation_function = 'healpix_simple' #'az_za_simple'
    uvb.freq_interp_kind = 'linear'
    freq_range = (np.min(uvb.freq_array), np.max(uvb.freq_array))

    # Construct interpolation grid for beam
    grid_re, grid_im = tv.beams.construct_beam_grid(uvb, 
                                                    Nza=400, Naz=401, 
                                                    freq=np.unique(uvb.freq_array), 
                                                    axis=0, feed=0, spw=0, 
                                                    dtype=tf.float64)
    #beams = tf.expand_dims(tf.complex(grid_re, grid_im), 
    #                       axis=0) # FIXME: Should calculate a beam per antenna
    A = tf.complex(grid_re, grid_im) # (1, Naz, Nza, Nfreq, Npol[?])
    tile_shape = tf.convert_to_tensor([Nants, 1, 1, 1, 1], dtype=tf.int32)
    beams = tf.tile(A, tile_shape)
    
    # Run visibility simulation
    t0 = time.time()
    vis = tv.vis_specidx(antpos, 
                         lsts, freqs, 
                         ra, dec, flux, spectral_idx, 
                         beams=beams, freq_range=freq_range)
    t1 = time.time()
    
    # Save output to file
    np.save("test_vis_data", vis.numpy())

    # Output statistics
    tf.print("Run took %3.3f sec" % (t1 - t0))
    tf.print("Output shape:", vis.shape, "(Nlsts, Nants, Nants, Nfreqs)")
    tf.print("Visibility mean:", np.mean(vis.numpy()))
    tf.print("Visibility real:", np.min(vis.numpy().real), np.max(vis.numpy().real))
    tf.print("Visibility imag:", np.min(vis.numpy().imag), np.max(vis.numpy().imag))

