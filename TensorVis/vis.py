
import numpy as np
import tensorflow as tf

from .coords import az_za_to_delay

@tf.function
def vis(antpos, freqs, az, za, flux, spectral_idx, beam1, beam2):
    """
    Calculate visibility for a set of point sources
    freqs in MHz
    """
    d = tf.constant(14.6) # baseline length in m
    C = 299792458. # speed of light in m/s
    freq_ref = tf.constant(100.) # MHz
    phase_fac = tf.constant(-2. * np.pi * 1e6 / C)
    
    # lambda = c / f
    # 2 pi d / lambda = 2 pi d f / c
    
    # Calculate beam values at zenith angle/azimuth of each source
    #A1 = beam1(za, az) # freqs should also be an argument
    #A2 = beam2(za, az) # freqs should also be an argument
    A = tf.ones((antpos.shape[0], freqs.shape[0], az.shape[0]), 
                dtype=freq_ref.dtype) # FIXME: Dummy
    
    # Amplitude part of product used to form visibilities for this antenna
    # (sqrt of intensity)
    v = tf.sqrt(flux) * (tf.expand_dims(freqs, 1) / freq_ref)**(0.5*spectral_idx)
    v = tf.multiply(A, v) # multiply by antenna pattern
    
    # Calculate baseline delays and phase factor 
    tau = az_za_to_delay(az, za, antpos)
    ang_freq = phase_fac * tf.expand_dims(freqs, 1)
    phase = tf.tensordot(tf.expand_dims(tau, 0), ang_freq, axes=[0,1])
    
    # Multiply amplitude by phase factor and sum over all ptsrc contributions
    v = tf.einsum( 'ijk,ikj->ij', 
                   tf.complex(v, 0.),             # Amplitude factor
                   tf.exp(tf.complex(0.,phase)) ) # exp(i.phase)
    
    # Perform outer product to get visibilities
    vij = tf.einsum('ik,jk->ijk', v, tf.math.conj(v))
    return vij
    
    # Return upper triangular part
    #return tf.matrix_band_part(vij, 0, -1)
    #return tf.reduce_sum(v, 1)
