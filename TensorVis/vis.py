
import numpy as np
import tensorflow as tf
from . import coords
from . import beams as tvbeams

FLOAT_TYPE = tf.float64
COMPLEX_TYPE = tf.complex128

# Define constants
C = 299792458. # speed of light in m/s
freq_ref = tf.constant(100.e6, dtype=FLOAT_TYPE) # Hz
phase_fac = tf.constant(2.*np.pi, dtype=FLOAT_TYPE)
ZERO = tf.constant(0., dtype=FLOAT_TYPE)
PI = tf.constant(np.pi, dtype=FLOAT_TYPE)


@tf.function
def vis_for_specidx_source(antpos, freqs, az, za, flux, spectral_idx, beams, freq_range):
    """
    Calculate visibilities for a set of sources with given spectral indices. 
    
    Use this function if the source SED models are all power laws. This will 
    calculate the SED for each source within this function, which should reduce 
    CPU -> GPU memory transfer times. For more complicated SEDs, use the 
    ``vis_for_sed_source`` function, which has a full (Nfreqs, Nsrc) Tensor as 
    its input.
    
    Parameters
    ----------
    antpos : Tensor
        Tensor containing x,y,z positions of all antennas.
        Shape (Nants, 3).
    
    freqs : Tensor
        Tensor containing frequency values in Hz. Shape: (Nfreqs,).
    
    az, za : Tensor
        Tensors containing azimuth and zenith angle of point sources, in 
        radians. Shape: (Nsrc,).
    
    flux, spectral_idx : Tensor
        Tensors containing flux at 100 MHz and spectral index of point sources. 
        The flux is in arbitrary units, usually Jy. Shape: (Nsrc,)
    
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
    # No. of frequencies and sources
    Nfreqs = freqs.shape[0]
    Nsrc = az.shape[0]
    
    # Interpolate beam values at zenith angle/azimuth of each source for each antenna
    interp_fn = lambda bm: tf.transpose(
                                tvbeams.interpolate_beam( tf.math.real(bm), 
                                                          tf.math.imag(bm), 
                                                          za, az, freqs, 
                                                          freq_range=freq_range, 
                                                          dtype=freq_ref.dtype))
    A = tf.map_fn(interp_fn, (beams, ), 
                  fn_output_signature=tf.TensorSpec((Nfreqs, Nsrc), dtype=COMPLEX_TYPE)) 
    # A: (Nants, Nfreqs, Nsrc)
    
    # Zero-out sources below the horizon
    flux_masked = tf.where(za < PI/2., flux, 0.)
    
    # Amplitude part of product used to form visibilities for this antenna
    # (sqrt of intensity, single antenna pattern)
    v = tf.sqrt(flux_masked) * (tf.expand_dims(freqs, 1) / freq_ref)**(0.5*spectral_idx)
    v = tf.multiply(A, tf.complex(v, ZERO)) # multiply by antenna pattern
    # v: (Nfreqs, Nsrc)
    # A: (Nants, Nfreqs, Nsrc) - since Nsrc = Naz
    # v <- A.v: (Nants, Nfreqs, Nsrc)
    
    # Calculate baseline delays and phase factor
    tau = coords.az_za_to_delay(az, za, antpos)
    ang_freq = phase_fac * tf.expand_dims(freqs, 1) # freqs in Hz
    phase = tf.tensordot(tf.expand_dims(tau, 0), ang_freq, axes=[0,1]) # tau in s
    # tau: (Nants, Nsrc)
    # ang_freq: (Nfreqs, 1)
    # phase: (Nants, Nsrc, Nfreqs)
    
    # Multiply amplitude by phase factor
    # v . exp(i.phase): (Nants, Nfreqs, Nsrc) x (Nants, Nsrc, Nfreqs)
    vis = tf.einsum('ijk,ikj->ijk', v, tf.exp(tf.complex(ZERO, phase)) )
    # vis: (Nants, Nfreqs, Nsrc)
    
    # Perform outer product to get visibilities and sum
    vij = tf.einsum('kij,lij->kli', tf.math.conj(vis), vis)
    return vij


@tf.function
def vis_for_sed_source(antpos, freqs, az, za, flux, beams, freq_range):
    """
    Calculate visibilities for a set of sources with pre-computed SEDs (flux at 
    each frequency channel).
    
    See ``vis_for_specidx_source`` for a faster implementation specialsed to 
    sources with power law SEDs.
    
    Parameters
    ----------
    antpos : Tensor
        Tensor containing x,y,z positions of all antennas.
        Shape (Nants, 3).
    
    freqs : Tensor
        Tensor containing frequency values in Hz. Shape: (Nfreqs,).
    
    az, za : Tensor
        Tensors containing azimuth and zenith angle of point sources, in 
        radians. Shape: (Nsrc,).
    
    flux : Tensor
        Tensor containing flux of each source for each frequency channel. 
        The flux is in arbitrary units, usually Jy. Shape: (Nfreqs, Nsrc).
    
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
    # No. of frequencies and sources
    Nfreqs = freqs.shape[0]
    Nsrc = az.shape[0]
    
    # Interpolate beam values at zenith angle/azimuth of each source for each antenna
    interp_fn = lambda bm: tf.transpose(
                                tvbeams.interpolate_beam( tf.math.real(bm), 
                                                          tf.math.imag(bm), 
                                                          za, az, freqs, 
                                                          freq_range=freq_range, 
                                                          dtype=freq_ref.dtype))
    A = tf.map_fn(interp_fn, (beams, ), 
                  fn_output_signature=tf.TensorSpec((Nfreqs, Nsrc), dtype=COMPLEX_TYPE)) 
    # A: (Nants, Nfreqs, Nsrc)
    
    # Zero-out sources below the horizon (za > pi/2)
    flux_masked = tf.where(za < PI/2., flux, 0.)
    
    # Amplitude part of product used to form visibilities for this antenna
    # (sqrt of intensity, single antenna pattern)
    v = tf.sqrt(flux_masked)
    v = tf.multiply(A, tf.complex(v, ZERO)) # multiply by antenna pattern
    # v: (Nfreqs, Nsrc)
    # A: (Nants, Nfreqs, Nsrc), since Nsrc = Naz
    # v <- A.v: (Nants, Nfreqs, Nsrc)
    
    # Calculate baseline delays and phase factor
    tau = coords.az_za_to_delay(az, za, antpos)
    ang_freq = phase_fac * tf.expand_dims(freqs, 1) # freqs in Hz
    phase = tf.tensordot(tf.expand_dims(tau, 0), ang_freq, axes=[0,1]) # tau in s
    # tau: (Nants, Nsrc)
    # ang_freq: (Nfreqs, 1)
    # phase: (Nants, Nsrc, Nfreqs)
    
    # Multiply amplitude by phase factor
    # v . exp(i.phase): (Nants, Nfreqs, Nsrc) x (Nants, Nsrc, Nfreqs)
    vis = tf.einsum('ijk,ikj->ijk', v, tf.exp(tf.complex(ZERO, phase)) )
    # vis: (Nants, Nfreqs, Nsrc)
    
    # Perform outer product to get visibilities and sum
    vij = tf.einsum('kij,lij->kli', tf.math.conj(vis), vis)
    return vij


@tf.function
def vis_specidx(antpos, lsts, freqs, ra, dec, flux, spectral_idx, beams, 
                freq_range, parallel_iterations=None, swap_memory=False):
    """
    Calculate visibilities for a set of sources with power-law SEDs.
    
    All auto- and cross-spectra are computed for the full set of antennas, as a 
    function of frequency and time.
    
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
        radians. Shape: (Nsrc,).
    
    flux, spectral_idx : Tensor
        Tensors containing flux at 100 MHz and spectral index of point sources. 
        The flux is in arbitrary units, usually Jy. Shape: (Nsrc,)
    
    beams : Tensor
        Data for the antenna patterns, to be used by an interpolation routine. 
        Shape: (Nants, Ngrid_az, Ngrid_za, Ngrid_freq).
    
    freq_range : tuple or list
        Tuple or list of len(2), with the maximum and minimum of the frequency 
        range that will be covered by the beam interpolation grid. 
    
    parallel_iterations : int, optional
        The number of iterations of `tf.map_fn` allowed to run in parallel. The 
        map is over LSTs. Parallel execution may not be used in TF eager 
        execution mode. Default: None.
    
    swap_memory : bool, optional
        Enables GPU-CPU memory swapping in the call to `tf.map_fn`. 
        Default: False
    
    Returns
    -------
    vis : Tensor
        Tensor of complex visibility values. Shape: (Nlsts, Nants, Nants, Nfreqs).
    """
    # Function to calculate source coords and visibilities for a single LST
    @tf.function
    def vis_for_lst_sed(lst):
        topo_cosines = coords.equatorial_to_topocentric(ra, dec, lst)
        az, za = coords.topocentric_to_az_za(topo_cosines[1], topo_cosines[0])
        return tf.stack(vis_for_specidx_source(
                                     antpos=antpos, freqs=freqs, 
                                     az=az, za=za, 
                                     flux=flux, spectral_idx=spectral_idx,
                                     beams=beams, freq_range=freq_range)
                                     )
    
    # Output template for `vis_for_lst`
    Nants = antpos.shape[0]
    Nfreqs = freqs.shape[0]
    vis_output_spec = tf.TensorSpec((Nants, Nants, Nfreqs), dtype=COMPLEX_TYPE)
    
    # Loop over LSTs
    vij_lst = tf.map_fn(vis_for_lst_sed, lsts, 
                        fn_output_signature=vis_output_spec, 
                        parallel_iterations=parallel_iterations, 
                        swap_memory=swap_memory)
    return vij_lst


@tf.function
def vis_sed(antpos, lsts, freqs, ra, dec, flux, beams, freq_range, 
                parallel_iterations=None, swap_memory=False):
    """
    Calculate visibilities for a set of sources with general SEDs.
    
    All auto- and cross-spectra are computed for the full set of antennas, as a 
    function of frequency and time.
    
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
        radians. Shape: (Nsrc,).
    
    flux : Tensor
        Tensor containing flux of each source for each frequency channel. 
        The flux is in arbitrary units, usually Jy. Shape: (Nfreqs, Nsrc).
    
    beams : Tensor
        Data for the antenna patterns, to be used by an interpolation routine. 
        Shape: (Nants, Ngrid_az, Ngrid_za, Ngrid_freq).
    
    freq_range : tuple or list
        Tuple or list of len(2), with the maximum and minimum of the frequency 
        range that will be covered by the beam interpolation grid. 
    
    parallel_iterations : int, optional
        The number of iterations of `tf.map_fn` allowed to run in parallel. The 
        map is over LSTs. Parallel execution may not be used in TF eager 
        execution mode. Default: None.
    
    swap_memory : bool, optional
        Enables GPU-CPU memory swapping in the call to `tf.map_fn`. 
        Default: False
    
    Returns
    -------
    vis : Tensor
        Tensor of complex visibility values. Shape: (Nlsts, Nants, Nants, Nfreqs).
    """
    # Function to calculate source coords and visibilities for a single LST
    @tf.function
    def vis_for_lst(lst):
        topo_cosines = coords.equatorial_to_topocentric(ra, dec, lst)
        az, za = coords.topocentric_to_az_za(topo_cosines[1], topo_cosines[0])
        return tf.stack(vis_for_sed_source(
                                     antpos=antpos, freqs=freqs, 
                                     az=az, za=za, flux=flux, 
                                     beams=beams, freq_range=freq_range)
                                     )
    
    # Output template for `vis_for_lst`
    Nants = antpos.shape[0]
    Nfreqs = freqs.shape[0]
    vis_output_spec = tf.TensorSpec((Nants, Nants, Nfreqs), dtype=COMPLEX_TYPE)
    
    # Loop over LSTs
    vij_lst = tf.map_fn(vis_for_lst, lsts, 
                        fn_output_signature=vis_output_spec, 
                        parallel_iterations=parallel_iterations, 
                        swap_memory=swap_memory)
    return vij_lst


