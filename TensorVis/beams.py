
import numpy as np
import tensorflow as tf
from .tensorflow_interp import interpolate3d


def construct_beam_grid(uvb, za, az, freq=None, axis=0, feed=0, spw=0, 
                        dtype=tf.float32, **uvbeam_interp_opts):
    """
    Construct a 3D grid of beam values from a UVBeam object, 
    with axes zenith angle, azimuth, and frequency. The resulting 
    grid of values can then be used by a 3D interpolator.
    
    The output is a pair of TensorFlow tensors for the real and 
    imaginary parts respectively.
    
    Parameters
    ----------
    uvb : UVBeam object
        A UVBeam object that has been correctly initialised 
        so that the `interp()` method can be used.
    
    za, az, freq : array_like
        1D arrays of grid points in zenith angle and azimuth
        (in radians) and frequency (in Hz). The beam function 
        is evaluated on a regular 3D grid constructed from 
        these 1D arrays.
        
        If `freq` is `None`, it will be taken from the UVBeam 
        object, `uvb.freq_array`.
        
        Note that the range of za is [0, pi/2] and the range of 
        az is [0, 2*pi].
    
    axis, feed : int, optional
        Indices of the direction vector (axis) on the sky, and 
        the feed of the receiver. How these are defined depends 
        on the input UVBeam object, but the axes will normally 
        be 0 or 1 (theta and phi unit vectors), and the feeds 
        will normally be 0 or 1 (n or e feeds).
        
        Defaults: 0.
    
    spw : int, optional
        Index of the spectral window to use in the UVBeam object. 
        Default: 0.
    
    **uvbeam_interp_opts : kwargs, optional
        Options to pass to the UVBeam `uvb.interp` method.
        
    Returns
    -------
    grid_re, grid_im : array_like, complex
        3D arrays of beam values, with shape (za.size, az.size, uvb.freq_array.size).
        Returned as real and imaginary parts.
    """
    # Interpolate beam on 3D grid
    # Returns array of shape: Naxes_vec, Nspws, Nfeeds, Nfreqs, Naz * Nza
    beam, basis_vec = uvb.interp(az, za, freq_array=freq, az_za_grid=True, 
                                 **uvbeam_interp_opts)
    
    # Reshape grid
    Nfreq = beam.shape[3]
    beam = beam[axis,spw,feed,:,:].reshape(Nfreq, za.size, az.size)
    beam = np.swapaxes(beam, 0, -1) # az, za, freq
    
    # Output tensors
    grid_re = tf.convert_to_tensor(beam[np.newaxis,:,:,:,np.newaxis].real, dtype=dtype)
    grid_im = tf.convert_to_tensor(beam[np.newaxis,:,:,:,np.newaxis].imag, dtype=dtype)
    return grid_re, grid_im


def interpolate_beam_grid(grid_re, grid_im, za, az, freqs, dtype=tf.float32):
    """
    Interpolate beam on a 3D grid in zenith angle, azimuth, and frequency, 
    using TensorFlow routines.
    
    Parameters
    ----------
    grid_re, grid_im : array_like, complex
        3D array of beam values, with shape (za.size, az.size, uvb.freq_array.size).
        Assumed to be a complex numpy ndarray.
    
    xx
    """
    # Stack into tensor of shape (Nbatches, Nsample_points, Ndimensions)
    # where Ndimensions = 3
    ang_coords = tf.stack([az, za])
    coords = tf.meshgrid(ang_coords, freqs)
    #sample_pts = tf.expand_dims( , freq]), axis=0)

    sample_pts = tf.convert_to_tensor(coords[np.newaxis,:,:])
    sample_pts.shape
    y = interpolate3d(grid, sample_pts)
    y2 = y.numpy()[0].reshape(za.size, az.size, 2)
    print(y2.shape)
    
