
import numpy as np
import tensorflow as tf
from . import coords

FLOAT_TYPE = tf.float64
COMPLEX_TYPE = tf.complex128

# Define constants
C = 299792458. # speed of light in m/s
freq_ref = tf.constant(100., dtype=FLOAT_TYPE) # MHz
phase_fac = tf.constant(-2. * np.pi * 1e6 / C, dtype=FLOAT_TYPE)
# 2 pi d / lambda = 2 pi d f / c


@tf.function
def vis_ptsrc_block(antpos, freqs, az, za, flux, spectral_idx, beams):
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
    tau = coords.az_za_to_delay(az, za, antpos)
    ang_freq = phase_fac * tf.expand_dims(freqs, 1)
    phase = tf.tensordot(tf.expand_dims(tau, 0), ang_freq, axes=[0,1])
    # tau: (Nants, Nptsrc)
    # ang_freq: (Nfreqs, 1)
    # phase: (Nants, Nptsrc, Nfreqs)
    
    # Multiply amplitude by phase factor and sum over all ptsrc contributions
    # v . exp(i.phase): (Nants, Nfreqs, Nptsrc) x (Nants, Nptsrc, Nfreqs)
    vis = tf.einsum('ijk,ikj->ij', 
                     tf.complex(v, tf.cast(0., FLOAT_TYPE)), # Amplitude factor
                     tf.exp(tf.complex(tf.cast(0., FLOAT_TYPE), phase)) ) # exp(i.phase)
    # vis: (Nants, Nfreqs)
    return vis


@tf.function
def vis_snapshot(antpos, freqs, az, za, flux, spectral_idx, beams, nblocks=1):
    """
    Calculate visibilities for a set of point sources at a given snapshot in 
    time. All auto- and cross-spectra are computed for the full set of antennas. 
    
    Note that `antpos` is used as the reference datatype. All other inputs 
    should be of the same type (i.e. `tf.float32` or `FLOAT_TYPE`).
    
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
    
    nblocks : int, optional
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
                                                beams)
    
    vis_blocks = tf.map_fn(vis_for_block, 
                           (az_blocks, za_blocks, flux_blocks, spectral_idx_blocks), 
                           dtype=COMPLEX_TYPE) 
                           #fn_output_signature=COMPLEX_TYPE)  # TensorFlow>=v2.3
    # vis_blocks: (Nblocks, Nants, Nfreqs)
    
    # Sum blocks together
    v = tf.reduce_sum(vis_blocks, axis=0)
    # v: (Nants, Nfreqs)
    
    # Perform outer product to get visibilities
    vij = tf.einsum('ik,jk->ijk', v, tf.math.conj(v))
    return vij


@tf.function
def vis(antpos, lsts, freqs, ra, dec, flux, spectral_idx, beams, nblocks=1):
    """
    Calculate visibilities for a set of point sources at a range of LSTS. 
    All auto- and cross-spectra are computed for the full set of antennas. 
    
    Note that `antpos` is used as the reference datatype. All other inputs 
    should be of the same type (i.e. `tf.float32` or `FLOAT_TYPE`).
    
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
    
    nblocks : int, optional
        Number of blocks to split the point source data into. This is useful if 
        the catalogue results in intermediate calculations that give out-of-
        memory errors, but slows down the calculation as a loop over blocks of 
        point sources must be used instead. Default: 1.
    """
    # Function to calculate source coords and visibilities for a single LST
    def vis_for_lst(lst):
        az, za = coords.eq_to_az_za(ra, dec, lst)
        return tf.stack(vis_snapshot(antpos, freqs, 
                                     az, za, flux, spectral_idx, 
                                     beams, nblocks=nblocks))
    
    # Loop over LSTs
    vij_lst = tf.map_fn(vis_for_lst, lsts, dtype=COMPLEX_TYPE)
    #vij_lst = tf.map_fn(vis_for_lst, lsts, fn_output_signature=COMPLEX_TYPE) # TensorFlow>=v2.3
    return vij_lst


