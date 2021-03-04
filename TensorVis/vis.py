
import numpy as np
import tensorflow as tf
from . import coords
from . import beams as tvbeams

FLOAT_TYPE = tf.float64
COMPLEX_TYPE = tf.complex128

# Define constants
C = 299792458. # speed of light in m/s
freq_ref = tf.constant(100.e6, dtype=FLOAT_TYPE) # Hz
phase_fac = tf.constant(-2. * np.pi / C, dtype=FLOAT_TYPE) # multiply nu in Hz
ZERO = tf.constant(0., dtype=FLOAT_TYPE)
# 2 pi d / lambda = 2 pi d f / c


@tf.function
def vis_ptsrc_block(antpos, freqs, az, za, flux, spectral_idx, beams, freq_range):
    """
    Calculate per-antenna visibility response (i.e. the sqrt of a full visibility) 
    for a given set of point sources.
    
    Parameters
    ----------
    antpos : Tensor
        Tensor containing x,y,z positions of all antennas.
        Shape (Nants, 3).
    
    freqs : Tensor
        Tensor containing frequency values in Hz. Shape: (Nfreqs,).
    
    az, za : Tensor
        Tensors containing azimuth and zenith angle of point sources, in 
        radians. Shape: (Nptsrc,).
    
    flux, spectral_idx : Tensor
        Tensors containing flux at 100 MHz and spectral index of point sources. 
        The flux is in arbitrary units, usually Jy. Shape: (Nptsrc,)
    
    beams : Tensor
        Data for the antenna patterns, to be used by an interpolation routine. 
        Shape: (Nants, Ngrid_az, Ngrid_za, Ngridfreq).
    
    freq_range : tuple or list
        Tuple or list of len(2), with the maximum and minimum of the frequency 
        range that will be covered by the interpolation grid. 
    
    Returns
    -------
    vis : Tensor
        Visibilities for all pairs of antennas, returned as a complex-valued 
        tensor of shape (Nants, Nants, Nfreqs).
    """
    # Calculate beam values at zenith angle/azimuth of each source
    Nants = tf.convert_to_tensor([antpos.shape[0], 1], dtype=tf.int32)
    # FIXME: This just duplicates the same beam many times
    A = tvbeams.interpolate_beam(tf.math.real(beams[0]), 
                                 tf.math.imag(beams[0]), 
                                 za, az, freqs, 
                                 freq_range=freq_range, 
                                 dtype=freq_ref.dtype)
    A = tf.transpose(A) # A: (Nfreqs, Nptsrc)
    A = tf.reshape(tf.tile(A, Nants), (-1, A.shape[0], A.shape[1]))
    # A: (Nants, Nfreqs, Nptsrc)
    # FIXME: If only one beam pattern is provided, memory can be saved here 
    # by using Nants -> 1 in the A operator
    
    # Amplitude part of product used to form visibilities for this antenna
    # (sqrt of intensity, single antenna pattern)
    v = tf.sqrt(flux) * (tf.expand_dims(freqs, 1) / freq_ref)**(0.5*spectral_idx)
    v = tf.multiply(A, tf.complex(v, ZERO)) # multiply by antenna pattern
    # v: (Nfreqs, Nptsrc)
    # A: (Nants, Nfreqs, Nptsrc) - since Nptsrc = Naz
    # v <- A.v: (Nants, Nfreqs, Nptsrc)
    
    # Calculate baseline delays and phase factor 
    tau = coords.az_za_to_delay(az, za, antpos)
    ang_freq = phase_fac * tf.expand_dims(freqs, 1) # freqs in Hz
    phase = tf.tensordot(tf.expand_dims(tau, 0), ang_freq, axes=[0,1]) # tau in s
    # tau: (Nants, Nptsrc)
    # ang_freq: (Nfreqs, 1)
    # phase: (Nants, Nptsrc, Nfreqs)
    
    # Multiply amplitude by phase factor and sum over all ptsrc contributions
    # v . exp(i.phase): (Nants, Nfreqs, Nptsrc) x (Nants, Nptsrc, Nfreqs)
    vis = tf.einsum('ijk,ikj->ij', v, tf.exp(tf.complex(ZERO, phase)) )
    # vis: (Nants, Nfreqs)
    return vis


@tf.function
def vis_snapshot(antpos, freqs, az, za, flux, spectral_idx, beams, 
                 freq_range, nblocks=1):
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
        Tensor containing frequency values in Hz. Shape: (Nfreqs,).
    
    az, za : Tensor
        Tensors containing azimuth and zenith angle of point sources, in 
        radians. Shape: (Nptsrc,).
    
    flux, spectral_idx : Tensor
        Tensors containing flux at 100 MHz and spectral index of point sources. 
        The flux is in arbitrary units, usually Jy. Shape: (Nptsrc,)
    
    beams : Tensor
        Data for the antenna patterns, to be used by an interpolation routine. 
        Shape: (Nants, Ngrid_az, Ngrid_za, Ngridfreq).
    
    freq_range : tuple or list
        Tuple or list of len(2), with the maximum and minimum of the frequency 
        range that will be covered by the interpolation grid. 
    
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
                                                beams=beams, 
                                                freq_range=freq_range)
    
    vis_blocks = tf.map_fn(vis_for_block, 
                           (az_blocks, za_blocks, flux_blocks, spectral_idx_blocks), 
                           dtype=COMPLEX_TYPE) 
                           #fn_output_signature=COMPLEX_TYPE)  # TensorFlow>=v2.3
    # vis_blocks: (Nblocks, Nants, Nfreqs)
    
    # Sum blocks together
    v = tf.reduce_sum(vis_blocks, axis=0) # v: (Nants, Nfreqs)
    
    # Perform outer product to get visibilities
    vij = tf.einsum('ik,jk->ijk', v, tf.math.conj(v))
    return vij


@tf.function
def vis(antpos, lsts, freqs, ra, dec, flux, spectral_idx, beams, freq_range, 
        nblocks=1):
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
        Tensor containing frequency values in Hz. Shape: (Nfreqs,).
    
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
    
    freq_range : tuple or list
        Tuple or list of len(2), with the maximum and minimum of the frequency 
        range that will be covered by the interpolation grid. 
    
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
                                     beams=beams, freq_range=freq_range, 
                                     nblocks=nblocks))
    
    # Loop over LSTs
    vij_lst = tf.map_fn(vis_for_lst, lsts, dtype=COMPLEX_TYPE)
    #vij_lst = tf.map_fn(vis_for_lst, lsts, fn_output_signature=COMPLEX_TYPE) # TensorFlow>=v2.3
    return vij_lst


