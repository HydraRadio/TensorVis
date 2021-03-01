
import numpy as np


def build_hex_array(hex_spec=(3,4), ants_per_row=None, d=14.6):
    """
    Build an antenna position dict for a hexagonally close-packed array.
    
    Parameters
    ----------
    hex_spec : tuple, optional
        If `ants_per_row = None`, this is used to specify a hex array as 
        `hex_spec = (nmin, nmax)`, where `nmin` is the number of antennas in 
        the bottom and top rows, and `nmax` is the number in the middle row. 
        The number per row increases by 1 until the middle row is reached.
        
        Default: (3,4) [a hex with 3,4,3 antennas per row]
    
    ants_per_row : array_like, optional
        Number of antennas per row. Default: None.
    
    d : float, optional
        Minimum baseline length between antennas in the hex array, in meters. 
        Default: 14.6.
    
    Returns
    -------
    ants : dict
        Dictionary with antenna IDs as the keys, and tuples with antenna 
        (x, y, z) position values (with respect to the array center) as the 
        values. Units: meters.
    """
    ants = {}
    
    # If ants_per_row isn't given, build it from hex_spec
    if ants_per_row is None:
        r = np.arange(hex_spec[0], hex_spec[1]+1).tolist()
        ants_per_row = r[:-1] + r[::-1]
    
    # Assign antennas
    k = -1
    y = 0.
    dy = d * np.sqrt(3) / 2. # delta y = d sin(60 deg)
    for j, r in enumerate(ants_per_row):
        
        # Calculate y coord and x offset
        y = -0.5 * dy * (len(ants_per_row)-1) + dy * j
        x = np.linspace(-d*(r-1)/2., d*(r-1)/2., r)
        for i in range(r):
            k += 1
            ants[k] = (x[i], y, 0.)
            
    return ants
    
