"""Pure geometry routines extracted from the authors' voxel_utils.py and utils.py.

The source modules also initialize remote services. These functions preserve the
authors' voxel-coordinate, cross-section and foreground-statistic conventions.
"""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, convolve, measurements
from skimage import measure


def spherical_2_cart(r,theta,phi):
    '''
    Convert spherical to cartesian coordinates.

    Args:
        r: Radius
        theta: Polar angle
        phi: Azimuthal angle

    Returns:
        tuple: (x, y, z) cartesian coordinates
    '''
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    return x, y, z


def cart_2_spherical(x,y,z):
    '''
    Convert cartesian to spherical coordinates.

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        tuple: (r, theta, phi) spherical coordinates
    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos( z / r )
    if x==0:
        # if y=0, phi is undefined but we'll call it zero
        phi = 0 if y==0 else np.sign(y) * np.pi/2
    elif y==0:
        # x is non-zero but y is 0
        phi = 0 if x>0 else np.pi
    else:
        # x and y are non-zero, so figure out which quadrant we're in
        if (x>0) and (y>0): phi = np.arctan(y/x)
        elif (x<0) and (y>0): phi = np.arctan(np.abs(x)/y) + np.pi/2
        elif (x<0) and (y<0): phi = np.arctan( np.abs(y) / np.abs(x) ) + np.pi
        elif (x>0) and (y<0):phi = np.arctan( x / np.abs(y) ) + 3*np.pi/2
    return r, theta, phi


def calc_orthonormal_basis(vec):
    '''
    Calculate orthonormal basis vectors from input vector.

    Args:
        vec: 2D vector (theta, phi components)

    Returns:
        tuple: Three orthonormal basis vectors
    '''
    assert len(vec)==2

    theta = np.zeros((3,),dtype='float'); phi = np.zeros((3,),dtype='float')
    theta[0] = vec[0]; phi[0] = vec[1]
    phi[1] = phi[0] - np.pi/4 # ensures cos( phi_1 - phi_2 ) is not zero

    if theta[0] == 0:
        # the above equation will have a divide by zero error
        # this is because the input vector is entirely along z-axis
        x,y,z = spherical_2_cart(1, vec[0], vec[1])
        _, theta[1], phi[1] = cart_2_spherical(1,0,0)
    else:
        theta[1] = np.arctan( -np.cos(theta[0]) / (np.sin(theta[0])*np.cos(phi[0]-phi[1])) ) # equation from mathematica

    # find third vector by cross product of first two vectors in cartesian coordinates
    x = np.zeros((3,),dtype=float); y = np.zeros((3,),dtype=float); z = np.zeros((3,),dtype=float)
    for i in [0,1]:
        x[i], y[i], z[i] = spherical_2_cart(1, theta[i], phi[i])
    x[2], y[2], z[2] = np.cross([x[0],y[0],z[0]], [x[1],y[1],z[1]])

    # normalize vectors (although, they should aready by normalized
    norm_factor = np.sqrt( x**2 + y**2 + z**2 )
    x /= norm_factor; y /= norm_factor; z /= norm_factor

    # make sure all vector pairs are orthogonal
    for pair in [[0,1],[0,2],[1,2]]:
        assert np.abs(x[pair[0]]*x[pair[1]] + y[pair[0]]*y[pair[1]] + z[pair[0]]*z[pair[1]]) < 10**-3
    return x,y,z


def get_center_object(array, center = None):
    """
    Extracts the central connected component from a binary volume.

    This function identifies the largest connected component in the volume
    and returns a binary mask containing only that component. This is useful
    for isolating individual objects or structures in the data.

    Args:
        array: Binary volume to process
        center: Optional center coordinates to use as reference

    Returns:
        Binary mask containing only the central connected component
    """
    assert np.any(array==1)
    # consider using measure.regionprops for this part
    plane_labels = measure.label(array.astype(int), background=0)
    unique_labels = np.unique(plane_labels)
    unique_labels = unique_labels[ unique_labels>0 ] # get rid of label for background

    if len(array.shape) == 2:
        # this is a cross section
        row, col = np.ceil( np.array(plane_labels.shape) / 2).astype(int) if center is None else center
        if plane_labels[ row, col ] == 0:
            # find cloest row,col element with labels greater than 0
            foreground_idxs = np.array( np.where(plane_labels > 0) ) # 2xn matrix
            label_idx = np.argmin( np.sum( ( foreground_idxs - np.array([[row],[col]]) )**2, axis=0 ) )
            row, col = foreground_idxs[:,label_idx]
        center_label = plane_labels[row,col]
    else:
        # this is a 3D Volume
        row, col, page = np.ceil( np.array(plane_labels.shape) / 2).astype(int) if center is None else center
        if plane_labels[ row, col, page ] == 0:
            # find cloest row,col element with labels greater than 0
            foreground_idxs = np.array( np.where(plane_labels > 0) ) # 3xn matrix
            label_idx = np.argmin( np.sum( ( foreground_idxs - np.array([[row],[col],[page]]) )**2, axis=0 ) )
            row, col, page = foreground_idxs[:,label_idx]
        center_label = plane_labels[row,col,page]
    assert center_label > 0, "center label should not be the background"
    return center_label == plane_labels


def calc_plane_coords(theta, phi, box_zyx, height=None, width=None):
    """
    Calculates coordinates for a plane in 3D space.

    This function generates the coordinates for a plane defined by spherical
    angles theta and phi, within a given bounding box. The plane can be
    specified with custom height and width, or will default to the maximum
    dimensions that fit within the bounding box.

    Args:
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
        box_zyx: Bounding box coordinates in zyx order [min_zyx, max_zyx]
        height: Optional height of the plane
        width: Optional width of the plane

    Returns:
        Array of plane coordinates
    """
    center = np.flip(np.array(box_zyx[1] + box_zyx[0])/2).astype(int)
    x, y, z = calc_orthonormal_basis( [theta,phi] )
    if height is None:
        max_dr = np.max( box_zyx[1] - box_zyx[0] )
        height, width = int(max_dr), int(max_dr)

    height_vec = np.array( [x[1],y[1],z[1]] ).reshape((1,1,3)) * height
    width_vec = np.array( [x[2],y[2],z[2]] ).reshape((1,1,3)) * width
    base_point = np.ceil(center - (height_vec + width_vec)*(1/2.0)).astype(int)
    hor_grid, ver_grid = np.meshgrid( np.linspace(-0.5,0.5,num=width),np.linspace(-0.5,0.5,num=height) )
    assert hor_grid.shape[0]==height; assert hor_grid.shape[1]==width
    hor_grid = np.repeat( hor_grid[:,:,np.newaxis], 3, axis=2 )
    ver_grid = np.repeat( ver_grid[:,:,np.newaxis], 3, axis=2 )

    plane_coords = center.reshape((1,1,3)) + (hor_grid * width_vec) + (ver_grid * height_vec)
    return plane_coords.astype(int)


def find_cross_section(theta, phi, subvol, height=None, width=None, skel_CS=None):
    """
    Finds the cross-section of a 3D volume along a specified plane.

    This function calculates the intersection of a plane (defined by theta and phi)
    with a 3D volume. It can optionally use a skeleton cross-section as a reference.

    Args:
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
        subvol: 3D volume data
        height: Optional height of the cross-section
        width: Optional width of the cross-section
        skel_CS: Optional skeleton cross-section for reference

    Returns:
        Array containing the cross-section values
    """
    box_zyx = np.array([ [0,0,0], subvol.shape ]).astype(int) # I don't need to real coordinates just to get the CS values
    plane_coords = calc_plane_coords(theta, phi, box_zyx, height = height, width = width)
    flat_plane_vals = np.zeros( (plane_coords.shape[0] * plane_coords.shape[1], ), dtype=type(subvol[0,0,0]) )
    min_coords = np.flip(box_zyx[0]) # x,y,z coordinates
    plane_idxs = (plane_coords - min_coords.reshape((1,1,3))).astype(int)
    flat_plane_idxs = plane_idxs.reshape( ( plane_idxs.shape[0]*plane_idxs.shape[1], 3 ) )
    flat_valid_idxs = ~np.any( [np.any(flat_plane_idxs<0,axis=1), np.any(flat_plane_idxs>=np.flip(box_zyx[1] - box_zyx[0])[np.newaxis,:],axis=1)], axis=0 )
    flat_plane_vals[ flat_valid_idxs ] = subvol[ flat_plane_idxs[flat_valid_idxs,2], flat_plane_idxs[flat_valid_idxs,1], flat_plane_idxs[flat_valid_idxs,0] ]
    plane_vals = flat_plane_vals.reshape( (plane_coords.shape[0], plane_coords.shape[1]) )

    if np.all( np.isin( plane_vals.astype(int), [0,1] ) ) and np.all( np.isin( [0,1], plane_vals.astype(int) ) ):
        # this is a boolean array, so only keep the forground group nearest to the center
        if skel_CS is None:
            plane_vals = get_center_object( binary_fill_holes(plane_vals) )
        else:
            plane_labels = measure.label(plane_vals.astype(int), background=0)
            unique_labels = np.unique( plane_labels[skel_CS==1] )

            if np.all(unique_labels == 0):
                return np.zeros(plane_vals.shape)
            unique_labels = unique_labels[ unique_labels>0 ] # get rid of label for background
            plane_vals = binary_fill_holes(np.isin(plane_labels, unique_labels))
    elif skel_CS is not None:
        plane_vals = np.where( binary_fill_holes(skel_CS==1), plane_vals, 0 )

    return plane_vals


def get_border_mask(bool_array, border_type):
    """
    Creates a mask of the border/edge of a boolean array.

    This function can identify either the inner border (True pixels touching False)
    or outer border (False pixels touching True) of a binary mask.

    Args:
        bool_array: Boolean array to find border of
        border_type: Type of border to extract ('inner' or 'outer')

    Returns:
        Boolean array marking the border region
    """
    if border_type == 'off':
        border_mask = np.all([ binary_dilation(bool_array == 1) == 1, bool_array == 0], axis=0)
    elif border_type == 'on':
        border_mask = np.all([ binary_erosion(bool_array == 1) == 0, bool_array == 1], axis=0)
    return border_mask


def get_foreground_stats(vals, box_zyx, fill_holes=True, theta=None, phi=None, return_COM=True):
    """
    Calculates statistics about the foreground objects in a boolean array.

    Computes various metrics including area/volume, surface area/perimeter,
    and optionally center of mass for foreground objects in 2D or 3D.

    Args:
        vals: Boolean array to analyze
        box_zyx: Bounding box coordinates
        fill_holes: Whether to fill holes in objects
        theta: Optional azimuthal angle for cross-sections
        phi: Optional polar angle for cross-sections
        return_COM: Whether to return center of mass

    Returns:
        Dictionary of calculated statistics
    """
    if not np.any(vals):
        if not return_COM:
            return 0, 0
        else:
            return 0, 0, None, None

    offset = 1
    kernel_size = 2*offset + 1
    if len(vals.shape) == 3:
        xv, yv, zv = np.meshgrid(np.arange(kernel_size)-offset,
                                 np.arange(kernel_size)-offset,
                                 np.arange(kernel_size)-offset)
        conv_kernel = np.ones( (kernel_size, kernel_size, kernel_size) )
        conv_kernel[np.sqrt( xv**2 + yv**2 + zv**2 ) > offset] = 0
        conv_kernel[offset,offset,offset] = 0
    elif len(vals.shape) == 2:
        xv, yv = np.meshgrid(np.arange(kernel_size)-offset,
                             np.arange(kernel_size)-offset)
        conv_kernel = np.ones( (kernel_size, kernel_size) )
        conv_kernel[np.sqrt( xv**2 + yv**2 ) > offset] = 0
        conv_kernel[offset,offset] = 0
    num_bkg_edges = convolve((vals==0).astype(float),conv_kernel) # number of background pixels within offset radius
    border_mask = get_border_mask(vals, 'on')
    SA = np.sum( np.sqrt( num_bkg_edges[border_mask==1] ) )
    enclosed_space = np.sum(vals==1)

    if not return_COM:
        return enclosed_space, SA

    COM_idx = get_closest_true_idx(vals, np.array(measurements.center_of_mass(vals)).astype(int))
    if theta is None:
        # vals should be 3D
        assert len(vals.shape) == 3 and vals[COM_idx[0],COM_idx[1],COM_idx[2]] == 1
        COM_coords = np.flip(box_zyx[0] + COM_idx)
    else:
        # vals should be 2D
        assert len(vals.shape) == 2 and vals[COM_idx[0],COM_idx[1]] == 1
        plane_coords = calc_plane_coords(theta, phi, box_zyx)
        COM_coords = plane_coords[COM_idx[0], COM_idx[1],:]
    assert np.all(np.flip(COM_coords) >= box_zyx[0]) and np.all(np.flip(COM_coords) < box_zyx[1])

    return enclosed_space, SA, COM_coords, COM_idx
