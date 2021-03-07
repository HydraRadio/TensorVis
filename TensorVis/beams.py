
import numpy as np
import tensorflow as tf
from .tensorflow_interp import interpolate3d
from .utils import unit_interval

def construct_beam_grid(uvb, Nza, Naz, freq=None, axis=0, feed=0, spw=0, 
                        dtype=tf.float32, **uvbeam_interp_opts):
    """
    Construct a 3D grid of beam values from a UVBeam object, with axes zenith 
    angle, azimuth, and frequency. The resulting grid of values can then be 
    used by a 3D interpolator.
    
    The output is a pair of TensorFlow tensors for the real and imaginary parts 
    respectively.
    
    Parameters
    ----------
    uvb : UVBeam object
        A UVBeam object that has been correctly initialised so that the 
        `interp()` method can be used.
    
    Nza, Naz : int
        Number of grid points in the zenith angle and azimuth directions. Note 
        that the range of `za` is [0, pi/2] and the range of `az` is [0, 2*pi].
        
    freq : array_like
        1D arrays of grid points in frequency (in Hz). The beam function is 
        evaluated on a regular 3D grid constructed from these 1D arrays.
        
        If `freq` is `None`, it will be taken from the UVBeam object, 
        `uvb.freq_array`.
        
    axis, feed : int, optional
        Indices of the direction vector (axis) on the sky, and the feed of the 
        receiver. How these are defined depends on the input UVBeam object, but 
        the axes will normally be 0 or 1 (theta and phi unit vectors), and the 
        feeds will normally be 0 or 1 (n or e feeds).
        Defaults: 0.
    
    spw : int, optional
        Index of the spectral window to use in the UVBeam object. Default: 0.
    
    **uvbeam_interp_opts : kwargs, optional
        Options to pass to the UVBeam `uvb.interp` method.
        
    Returns
    -------
    grid_re, grid_im : Tensor, complex
        3D arrays of beam values, with shape (za.size, az.size, uvb.freq_array.size).
        Returned as real and imaginary parts.
    """
    # Define coordinate grids
    az = np.linspace(0., 2.*np.pi, Naz)
    za = np.linspace(0., np.pi/2., Nza)
    
    # Interpolate beam on 3D grid
    # Returns array of shape: Naxes_vec, Nspws, Nfeeds, Nfreqs, Naz * Nza
    az2d, za2d = np.meshgrid(az, za)
    beam, basis_vec = uvb.interp(az2d.flatten(), za2d.flatten(), 
                                 freq_array=np.atleast_1d(freq), 
                                 **uvbeam_interp_opts)
    
    # Reshape grid
    Nfreq = beam.shape[3]
    beam = beam[axis,spw,feed,:,:].reshape(Nfreq, za.size, az.size)
    beam = np.swapaxes(beam, 0, -1) # az, za, freq
    
    # Output tensors
    grid_re = tf.convert_to_tensor(beam[np.newaxis,:,:,:,np.newaxis].real, 
                                   dtype=dtype)
    grid_im = tf.convert_to_tensor(beam[np.newaxis,:,:,:,np.newaxis].imag, 
                                   dtype=dtype)
    return grid_re, grid_im, 
    

@tf.function
def coords_for_interp(za, az, freqs, freq_range, grid_shape, dtype=tf.float64):
    """
    Construct a Tensor with coordinates within the unit cube for each zenith 
    angle and azimuth, at each frequency. These are the coordinates that the 
    beam interpolation will be evaluated at.
    
    Parameters
    ----------
    za, az : array_like or Tensor
        1D numpy arrays or Tensors containing zenith angle and azimuth 
        coordinates of point sources, in radians.
    
    freqs : array_like or Tensor
        1D numpy array of frequency values (in Hz) that the beam interpolation 
        will be evaluated at.
    
    freq_range : tuple or list
        Tuple or list of len(2), with the maximum and minimum of the frequency 
        range that will be covered by the interpolation grid.
    
    grid_shape : tuple
        Shape of interpolation data grid, (Ngrid_za, Ngrid_az, Ngrid_freq).
    
    dtype : tf.dtype
        Data type to use for coordinate Tensors. Default: tf.float64.
    
    Returns
    -------
    sample_pts : Tensor
        Tensor with the correct shape for evaluating a trilinear spline with 
        `interpolate3d()`.
    """
    # Unit interval datatype (can be reduced precision)
    unit_type = tf.float64
    
    # Interpolation grid size
    Ngrid_za, Ngrid_az, Ngrid_freq = grid_shape
    
    # Convert frequency array to right shape/type
    freqs = tf.squeeze(tf.convert_to_tensor(freqs, dtype=dtype))
    Nfreqs = freqs.shape[0]
    Nptsrc = tf.convert_to_tensor([za.shape[0],], tf.int32) # int32/64 for tf.tile
    
    # Expand arrays to get full set of sample points
    az_rpt = tf.repeat(tf.convert_to_tensor(az, dtype=dtype), Nfreqs)
    za_rpt = tf.repeat(tf.convert_to_tensor(za, dtype=dtype), Nfreqs)
    freq_rpt = tf.tile(freqs, Nptsrc)
    
    # Convert arrays to unit interval and cast to different dtype
    az_unit = tf.cast(unit_interval(az_rpt, 0., 2.*np.pi, scale_factor=Ngrid_az), 
                      dtype=unit_type)
    za_unit = tf.cast(unit_interval(za_rpt, 0., np.pi/2., scale_factor=Ngrid_za), 
                      dtype=unit_type)
    freq_unit = tf.cast(unit_interval(freq_rpt, freq_range[0], freq_range[1], 
                                      Ngrid_freq), 
                        dtype=unit_type)
    
    # Combine coords into a structure of the expected shape
    sample_pts = tf.expand_dims(tf.transpose( 
                                    tf.stack([az_unit, za_unit, freq_unit]) ),
                                axis=0)
    return sample_pts


@tf.function
def interpolate_beam(grid_re, grid_im, za, az, freqs, freq_range, dtype=tf.float64):
    """
    Interpolate beam on a 3D grid in zenith angle, azimuth, and frequency, 
    using TensorFlow routines.
    
    Parameters
    ----------
    grid_re, grid_im : array_like, complex
        3D array of beam values, with shape (za.size, az.size, uvb.freq_array.size).
        Assumed to be a complex numpy ndarray.
    
    za, az : array_like or Tensor
        1D numpy arrays or Tensors containing zenith angle and azimuth 
        coordinates of point sources, in radians.
    
    freqs : array_like or Tensor
        1D numpy array of frequency values (in Hz) that the beam interpolation 
        will be evaluated at.
    
    freq_range : tuple or list
        Tuple or list of len(2), with the maximum and minimum of the frequency 
        range that will be covered by the interpolation grid.  
    
    dtype : tf.dtype
        Data type to use for coordinate Tensors. Default: tf.float64.
    
    Returns
    -------
    beam_vals : Tensor, complex
        Complex-valued tensor with beam value at the position of each source, 
        for each requested frequency.
    """
    # Put sample points in appropriately-scaled unit cube
    grid_shape = (grid_re.shape[2], grid_re.shape[1], grid_re.shape[3]) # za, az, freq
    sample_pts = coords_for_interp(za, az, freqs, freq_range=freq_range, 
                                   grid_shape=grid_shape, dtype=dtype)
    
    # Interpolate on grid
    beam_re = interpolate3d(grid_re, sample_pts)
    beam_im = interpolate3d(grid_im, sample_pts)
    beam = tf.complex(beam_re, beam_im)
    
    # Reshape and return
    # FIXME: Need to check that this is being shaped correctly
    return tf.reshape(beam, (za.shape[0], -1))
    
