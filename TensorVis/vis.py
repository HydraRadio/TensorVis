
import numpy as np
import tensorflow as tf
from .coords import az_za_to_delay

# Define constants
C = 299792458. # speed of light in m/s
freq_ref = tf.constant(100., dtype=tf.float64) # MHz
phase_fac = tf.constant(-2. * np.pi * 1e6 / C, dtype=tf.float64)
# 2 pi d / lambda = 2 pi d f / c


@tf.function
def vis_ptsrc_block(antpos, freqs, az, za, flux, spectral_idx, beam1, beam2):
    """
    Calculate per-antenna visibility response (i.e. the sqrt of a full visibility) 
    for a given set of point sources.
    """
    # Calculate beam values at zenith angle/azimuth of each source
    #A1 = beam1(za, az) # freqs should also be an argument
    #A2 = beam2(za, az) # freqs should also be an argument
    A = tf.ones((antpos.shape[0], freqs.shape[0], az.shape[0]), dtype=freq_ref.dtype) # FIXME: Dummy
    
    # Amplitude part of product used to form visibilities for this antenna
    # (sqrt of intensity, single antenna pattern)
    v = tf.sqrt(flux) * (tf.expand_dims(freqs, 1) / freq_ref)**(0.5*spectral_idx)
    v = tf.multiply(A, v) # multiply by antenna pattern
    # v: (Nfreqs, Nptsrc)
    # A: (Nants, Nfreqs, Nptsrc) - since Nptsrc = Naz
    # v <- A.v: (Nants, Nfreqs, Nptsrc)
    
    # Calculate baseline delays and phase factor 
    tau = az_za_to_delay(az, za, antpos)
    ang_freq = phase_fac * tf.expand_dims(freqs, 1)
    phase = tf.tensordot(tf.expand_dims(tau, 0), ang_freq, axes=[0,1])
    # tau: (Nants, Nptsrc)
    # ang_freq: (Nfreqs, 1)
    # phase: (Nants, Nptsrc, Nfreqs)
    
    # Multiply amplitude by phase factor and sum over all ptsrc contributions
    # v . exp(i.phase): (Nants, Nfreqs, Nptsrc) x (Nants, Nptsrc, Nfreqs)
    vis = tf.einsum('ijk,ikj->ij', 
                     tf.complex(v, tf.cast(0., tf.float64)), # Amplitude factor
                     tf.exp(tf.complex(tf.cast(0., tf.float64), phase)) ) # exp(i.phase)
    # vis: (Nants, Nfreqs)
    return vis


@tf.function
def vis_snapshot(antpos, freqs, az, za, flux, spectral_idx, beams, nblocks=1):
    """
    Calculate visibilities for a set of point sources at a given snapshot in 
    time. All auto- and cross-spectra are computed for the full set of antennas. 
    
    Note that `antpos` is used as the reference datatype. All other inputs 
    should be of the same type (i.e. `tf.float32` or `tf.float64`).
    
    Parameters
    ----------
    antpos : Tensor
        Tensor containing x,y,z positions of all antennas.
        Shape (Nants, 3).
    
    freqs : Tensor
        Tensor containing frequency values in MHz. Shape: (Nfreqs,).
    
    az, za : Tensor
        Tensors containing azimuth and zenith angle of point sources, in 
        radians. Shape: (Nptsrc,).
    
    flux, spectral_idx : Tensor
        Tensors containing flux at 100 MHz and spectral index of point sources. 
        The flux is in arbitrary units, usually Jy. Shape: (Nptsrc,)
    
    beams : Tensor
        Data for the antenna patterns, to be used by an interpolation routine. 
        Shape: (Nants, Ngrid_az, Ngrid_za, Ngridfreq).
    
    nblock : int, optional
        Number of blocks to split the point source data into. This is useful if 
        the catalogue results in intermediate calculations that give out-of-
        memory errors, but slows down the calculation as a loop over blocks of 
        point sources must be used instead. Default: 1.
    """
    Nants = antpos.shape[0]
    Nfreqs = freqs.shape[0]
    
    # Split intput ptsrc catalogue into chunks
    if az.shape[0] % nblocks != 0:
        raise ValueError("nblocks must be an integer factor of az.shape")
    az_blocks = tf.reshape(az, (nblocks, -1))
    za_blocks = tf.reshape(za, (nblocks, -1))
    flux_blocks = tf.reshape(flux, (nblocks, -1))
    spectral_idx_blocks = tf.reshape(spectral_idx, (nblocks, -1))
    
    # Loop to calculate each block in turn and update
    vis_for_block = lambda inp: vis_ptsrc_block(antpos, freqs, 
                                                inp[0], inp[1], inp[2], inp[3], 
                                                beam1, beam2)
    
    vis_blocks = tf.map_fn(vis_for_block, 
                           (az_blocks, za_blocks, flux_blocks, spectral_idx_blocks), 
                           fn_output_signature=tf.complex128) 
    # vis_blocks: (Nblocks, Nants, Nfreqs)
    
    # Sum blocks together
    v = tf.reduce_sum(vis_blocks, axis=0)
    # v: (Nants, Nfreqs)
    
    # Perform outer product to get visibilities
    vij = tf.einsum('ik,jk->ijk', v, tf.math.conj(v))
    return vij


@tf.function
def vis(antpos, lsts, freqs, ra, dec, flux, spectral_idx, beams, nblock=1):
    """
    Calculate visibilities for a set of point sources at a range of LSTS. 
    All auto- and cross-spectra are computed for the full set of antennas. 
    
    Note that `antpos` is used as the reference datatype. All other inputs 
    should be of the same type (i.e. `tf.float32` or `tf.float64`).
    
    Parameters
    ----------
    antpos : Tensor
        Tensor containing x,y,z positions of all antennas.
        Shape (Nants, 3).
    
    lsts : Tensor
        LST values to calculate visibilities at, in radians. Shape: (Nlsts,)
    
    freqs : Tensor
        Tensor containing frequency values in MHz. Shape: (Nfreqs,).
    
    ra, dec : Tensor
        Tensors containing Right Ascension and Declination of point sources, in 
        radians. ** For reasonable accuracy, these should be evaluated at the 
        present epoch **. Shape: (Nptsrc,).
    
    flux, spectral_idx : Tensor
        Tensors containing flux at 100 MHz and spectral index of point sources. 
        The flux is in arbitrary units, usually Jy. Shape: (Nptsrc,)
    
    beams : Tensor
        Data for the antenna patterns, to be used by an interpolation routine. 
        Shape: (Nants, Ngrid_az, Ngrid_za, Ngridfreq).
    
    nblock : int, optional
        Number of blocks to split the point source data into. This is useful if 
        the catalogue results in intermediate calculations that give out-of-
        memory errors, but slows down the calculation as a loop over blocks of 
        point sources must be used instead. Default: 1.
    """
    # Function to calculate source coords and visibilities for a single LST
    def vis_for_lst(lst):
        az, za = eq_to_az_za(ra, dec, lst)
        return tf.stack(vis_snapshot(antpos, freqs, 
                                     az, za, flux, spectral_idx, 
                                     beams, nblock=nblock))
    
    # Loop over LSTs
    vij_lst = tf.map_fn(vis_for_lst, lsts, fn_output_signature=tf.complex128)
    return vij_lst


# This is deprecated
@tf.function
def vis_bruteforce(antpos, freqs, az, za, flux, spectral_idx, beam1, beam2):
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
