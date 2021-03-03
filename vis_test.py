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
tf.debugging.set_log_device_placement(True)

import time, sys
import TensorVis as tv

np.random.seed(10)

FLOAT_TYPE = tf.float64
DEBUG = True
NBLOCKS = 5 # Number of blocks to use when calculating point source contrib.

# Default simulation settings
Nlsts = 4
Nfreqs = 8
Nptsrc = 500
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


# Generate hex array and convert to antpos tensor
antpos_dict = tv.utils.build_hex_array(hex_spec=(8,20), d=14.6)
antpos_arr = np.column_stack([val for val in antpos_dict.values()]).T
antpos = tf.convert_to_tensor(antpos_arr, dtype=FLOAT_TYPE)
Nants = len(antpos_arr)
print("Nants:", Nants)

# Frequency and time arrays
freqs = tf.convert_to_tensor(np.linspace(100., 120., Nfreqs), dtype=FLOAT_TYPE)
lsts = tf.convert_to_tensor([2.4030742+0.001*i for i in range(Nlsts)], dtype=FLOAT_TYPE)

# Generate randomly-placed point sources
ra = tf.convert_to_tensor(np.random.uniform(0., np.pi, Nptsrc), 
                          dtype=FLOAT_TYPE)
dec = tf.convert_to_tensor(np.random.uniform(-0.5*np.pi, 0.5*np.pi, Nptsrc), 
                           dtype=FLOAT_TYPE)
flux = tf.convert_to_tensor(10.**np.random.uniform(-8., -6., Nptsrc), 
                            dtype=FLOAT_TYPE)
spectral_idx = tf.convert_to_tensor(-2.7*np.ones(Nptsrc), 
                                    dtype=FLOAT_TYPE)

# Construct beam interpolation function
beams = tf.ones((Nants, 1), dtype=FLOAT_TYPE)

# Run visibility simulation
t0 = time.time()
vis = tv.vis(antpos, 
             lsts, freqs, 
             ra, dec, flux, spectral_idx, 
             beams, 
             nblocks=NBLOCKS)
t1 = time.time()

# Output statistics
print("Run took %3.3f sec" % (t1 - t0))
print("Output shape:", vis.shape)
print("Visibility mean:", np.mean(vis.numpy()))


