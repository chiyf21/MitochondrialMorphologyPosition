"""Authors' section/overhang routines with a public CloudVolume data adapter."""
from functools import lru_cache

from cloudvolume import CloudVolume
import numpy as np
from scipy.ndimage import measurements
from scipy.optimize import minimize

from paper_voxel_geometry import (
    find_cross_section, get_border_mask, get_center_object, spherical_2_cart,
)

ROOT = "precomputed://https://storage.googleapis.com/neuroglancer-janelia-flyem-hemibrain/v1.2/"


@lru_cache(maxsize=2)
def volume(layer):
    return CloudVolume(ROOT + layer, mip=0, progress=False, cache=False,
                       fill_missing=(layer == "mito-objects-grouped"), parallel=1)


def get_subvol_any_size(box_zyx, datatype):
    layer = "mito-objects-grouped" if datatype == "mito-objects" else datatype
    source = volume(layer)
    lo, hi = np.asarray(box_zyx, dtype=int)[:, ::-1]
    start = np.maximum(lo, source.bounds.minpt)
    stop = np.minimum(hi, source.bounds.maxpt)
    if np.any(start >= stop):
        raise ValueError("Requested box lies outside the source volume")
    data = np.asarray(source[tuple(slice(int(a), int(b)) for a, b in zip(start, stop))])
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    padding = tuple(zip(start - lo, hi - stop))[::-1]
    return np.pad(data.transpose(2, 1, 0), padding) if any(a or b for a, b in padding) else data.transpose(2, 1, 0)


def adjust_center(center_init, dr_small, bodyId, data_instance = 'segmentation'):
    """
    Adjusts the initial center point to be centered within the neuron body or mitochondria.

    This function iteratively searches around the initial center point to find a valid
    center that lies within the specified structure (neuron body or mitochondria).

    Args:
        center_init: Initial center coordinates [x, y, z]
        dr_small: Radius for searching around the center
        bodyId: ID of the neuron body to analyze
        data_instance: Type of data to analyze ('segmentation' or 'mito-objects')

    Returns:
        Adjusted center coordinates or None if no valid center found
    """
    def sphere_mask(radius):
        """
        Creates a boolean spherical mask of given radius.

        Args:
            radius: Radius of the sphere in voxels

        Returns:
            3D boolean array representing spherical mask
        """
        n = radius * 2
        x, y, z = np.meshgrid( np.linspace(-1,1,n), np.linspace(-1,1,n), np.linspace(-1,1,n) )
        return x**2 + y**2 + z**2 <= 1

    subvol = [False]
    num_increment = 1#0
    while not np.any(subvol):
        num_increment = int( num_increment * 2 )
        if dr_small*num_increment > 200: #5/8*1000:
            # center_init is too far from any tissue, so return None
            return None
        circ_bool = sphere_mask(dr_small*num_increment)
        min_zyx = np.flip(center_init).astype(int) - dr_small*num_increment
        max_zyx = np.flip(center_init).astype(int) + dr_small*num_increment
        box_zyx = np.array([ min_zyx, max_zyx])
        subvol = (get_subvol_any_size(box_zyx, data_instance) == bodyId) * circ_bool
        if data_instance == 'mito-objects':
            # multiply subvol by segmentation bool to get voxels that contain the mito and neuron
            subvol = (get_subvol_any_size(box_zyx, 'segmentation') == bodyId) * subvol
        if np.any(subvol):
            subvol = get_center_object(subvol)
    if dr_small == 1:
        # just return coordinates of true value closest to scenter
        body_idxs = np.array( np.where(subvol) )
        voxel_idx = np.argmin(np.sum( (body_idxs - num_increment)**2 ,axis=0))
        center = np.flip( body_idxs[:,voxel_idx] + min_zyx) # x,y,z coordinates
        return center.astype(int)

    if num_increment > 1:
        circ_bool = sphere_mask(dr_small)

    COM_idx = np.array(measurements.center_of_mass(subvol)).astype(int)
    past_subvol = np.copy(subvol); past_min_zyx = np.copy(min_zyx)
    dist_thresh = 2; count = 0
    past_COM_idx = COM_idx + 2*dist_thresh
    #old_min_zyx = min_zyx
    while np.sqrt(np.sum((COM_idx-past_COM_idx)**2)) >= dist_thresh:
        # center of mass has not converged yet
        count += 1
        past_COM_idx = np.copy(COM_idx)
        min_zyx = COM_idx + min_zyx - dr_small
        box_zyx = np.array([ min_zyx, min_zyx + 2*dr_small])
        subvol = (get_subvol_any_size(box_zyx, data_instance) == bodyId) * circ_bool
        if data_instance == 'mito-objects':
            # multiply subvol by segmentation bool to get voxels that contain the mito and neuron
            subvol = (get_subvol_any_size(box_zyx, 'segmentation') == bodyId) * subvol

        if np.any(subvol):
            subvol = get_center_object(subvol)
            past_subvol = np.copy(subvol)
            past_min_zyx = np.copy(min_zyx)
            COM_idx = np.array(measurements.center_of_mass(subvol)).astype(int)
        else:
            min_zyx = np.copy(past_min_zyx)
            subvol = np.copy(past_subvol)
        assert count < 10, 'infinite while loop detected'

    COM_idx = get_closest_true_idx(subvol, COM_idx)

    if subvol[COM_idx[0],COM_idx[1],COM_idx[2]]:
        # use COM as the center
        center = np.flip(COM_idx + min_zyx)
    else:
        # find coordinate closest to COM belonging to bodyId
        body_idxs = np.array( np.where(subvol) )
        voxel_idx = np.argmin(np.sum( (body_idxs - dr_small)**2 ,axis=0))
        center = np.flip( body_idxs[:,voxel_idx] + min_zyx) # x,y,z coordinates

    return center.astype(int)


def get_closest_true_idx(vals, idx):
    """
    Finds the closest True value in a boolean array to a given index.

    This function is used to find the nearest valid point in a binary mask
    to a reference point, which is useful for ensuring operations are performed
    on valid data points.

    Args:
        vals: Boolean array
        idx: Reference index to find closest True value to

    Returns:
        Index of closest True value
    """
    vals = vals==1
    if len(vals.shape) == 3:
        if vals[idx[0],idx[1],idx[2]]:
            # use idx as the center
            return idx
    elif len(vals.shape) == 2:
        if vals[idx[0],idx[1]]:
            # use idx as the center
            return idx
    # find index closest to idx that is true
    all_idxs = np.array( np.where(vals) )
    closest_true_idx = np.argmin(np.sum( (all_idxs - idx[:,np.newaxis])**2 ,axis=0))
    return all_idxs[:,closest_true_idx]


def find_best_cross_section(subvol):
    """
    Finds the optimal cross-section orientation by minimizing the cross-sectional area.

    This function uses optimization to find the plane orientation that produces
    the most compact cross-section of the volume, with a penalty for cropped sections.

    Args:
        subvol: 3D volume to analyze

    Returns:
        Minimizer object containing the optimal theta and phi angles
    """
    def CA_obj_fun(plane_norm, subvol):
        theta, phi = plane_norm
        CS = find_cross_section( theta, phi, subvol )
        CA = np.sum( CS > 0 )
        if is_cropped(CS, box_shape = subvol.shape, theta=theta, phi=phi):
            # multiply loss funtion to penalize for cropping
            CA *= 1000
        return CA

    # get best initial guess of theta, phi
    init_thetas = np.linspace(0, np.pi, 7)
    init_phis = np.linspace(0, np.pi, 7)
    min_area = np.inf
    for theta, phi in zip(init_thetas, init_phis):
        this_area = CA_obj_fun((theta,phi), subvol)
        if this_area < min_area:
            # found a plane normal with smaller cross-sectional area than area_0
            init_guess = (theta, phi)
            min_area = this_area + 0
    param_bounds = ( (init_guess[0]-np.pi/2,init_guess[0]+np.pi/2), (init_guess[1]-np.pi/2,init_guess[1]+np.pi/2) )
    res = minimize(CA_obj_fun, init_guess, args=(subvol),method='Nelder-Mead',bounds=param_bounds)
    assert res.fun <= min_area
    return res


def is_cropped(vals, box_shape=None, theta=None, phi=None):
    """
    Determines if a volume or cross-section is cropped at the edges.

    This function checks whether the data extends to the boundaries of the
    volume or cross-section, indicating that it might be cropped.

    Args:
        vals: Array to check for cropping
        box_shape: Optional shape of bounding box
        theta: Optional azimuthal angle for cross-sections
        phi: Optional polar angle for cross-sections

    Returns:
        Boolean indicating if the array is cropped
    """
    if theta is None:
        # vals should be 3D
        assert len(vals.shape) == 3
        cropped = False
        for side in [0,-1]:
            cropped = cropped or np.any(vals[side,:,:] == 1)
            cropped = cropped or np.any(vals[:,side,:] == 1)
            cropped = cropped or np.any(vals[:,:,side] == 1)
    else:
        # vals should be 2D
        assert len(vals.shape) == 2
        is_included = find_cross_section( theta, phi, np.broadcast_to(np.uint8(1), box_shape))
        cropped = np.any( np.all([get_border_mask(is_included, 'on') == 1, vals == 1], axis=0) )
    return cropped


def get_CrossSection_info(bodyId, coord, dr, return_mito=False, return_grayscale=False,
                         return_plane_coords=False, max_dr=700):
    """
    Retrieves comprehensive information about a cross-section through a neuron.

    This function extracts cross-section data and various metrics at a given
    coordinate along a neuron, optionally including mitochondria and grayscale data.

    Args:
        bodyId: ID of the neuron to analyze
        coord: Coordinates of the cross-section center
        dr: Radius around center to analyze
        return_mito: Whether to return mitochondria data
        return_grayscale: Whether to return raw image data
        return_plane_coords: Whether to return plane coordinates
        max_dr: Maximum allowed radius

    Returns:
        Dictionary containing requested cross-section information
    """
    is_crop = True
    adjusted_coord = adjust_center(coord, 5, bodyId) # coordinates of local center of mass
    #print(adjusted_coord)
    if adjusted_coord is None:
        output = []
        for _ in range(1 + int(return_mito) + int(return_grayscale) + int(return_plane_coords)):
            output.append(None)
        return output
    else:
        while is_crop:
            box_zyx = [ np.flip(adjusted_coord).astype(int) - dr, np.flip(adjusted_coord).astype(int) + dr ]
            skel_subvol = get_subvol_any_size(box_zyx, 'segmentation') == bodyId
            res = find_best_cross_section( skel_subvol )
            theta, phi = res.x
            skel_CS = find_cross_section( theta, phi, skel_subvol)
            is_crop = is_cropped(skel_CS, box_shape = skel_subvol.shape, theta=theta, phi=phi)
            if is_crop and dr == max_dr:
                # even after going to the max_dr, the skeleton is still cropped so just return None
                output = []
                for _ in range(1 + int(return_mito) + int(return_grayscale) + int(return_plane_coords)):
                    output.append(None)
                return output
            if is_crop:
                # going back through the loop, so use a bigger dr
                if dr < 400: dr += 100
                else: dr += 150
                dr = int(dr)
                if dr > max_dr:
                    dr = int(max_dr)
        del skel_subvol
        output = [skel_CS]

        if return_mito:
            mito_subvol = get_subvol_any_size(box_zyx, 'mito-objects') == bodyId
            mito_CS = find_cross_section( theta, phi, mito_subvol, skel_CS = skel_CS)
            del mito_subvol
            output.append(mito_CS)
        if return_grayscale:
            grayscale_subvol = get_subvol_any_size(box_zyx, 'grayscale')
            grayscale_CS = find_cross_section( theta, phi, grayscale_subvol, skel_CS = skel_CS)
            #grayscale_CS = find_cross_section( theta, phi, grayscale_subvol, skel_CS = binary_dilation(skel_CS, iterations=2) )
            del grayscale_subvol
            output.append(grayscale_CS)
        if return_plane_coords:
            output.append( calc_plane_coords(theta, phi, box_zyx) )
    return output


def get_down_idxs(s_np, node, bool_stop, count_start = False):
    '''
    Find all indices downstream of a node until reaching root or branch.

    Args:
        s_np: Neuron skeleton numpy array
        node: Node to start search from
        bool_stop: Boolean array indicating where to stop
        count_start: Whether to include start node in stop condition

    Returns:
        array: Indices of downstream nodes
    '''
    idxs = np.where( node == s_np[:,0] )[0]
    if not count_start:
        bool_stop[idxs[0]] = False
    while s_np[idxs[-1],5] != -1 and not bool_stop[idxs[-1]]:
        idxs = np.append(idxs, np.where( s_np[idxs[-1],5] == s_np[:,0] )[0][0] )
    return idxs


def get_up_idx_dist(s_np, coord):
    '''
    Find upstream index and distance for a coordinate.

    Args:
        s_np: Neuron skeleton numpy array
        coord: Coordinate to find upstream node for

    Returns:
        tuple: (upstream index, distance to upstream node)
    '''
    found_up = False
    for s_idx in np.argsort( np.sum( (s_np[:,[1,2,3]] - coord.reshape((1,3)))**2, axis=1 ) - s_np[:,4] )[:20]:
        xyz = np.array( spherical_2_cart(1, s_np[s_idx,7], s_np[s_idx,8]) )
        up_dist = np.sum(xyz * (coord - s_np[s_idx,[1,2,3]]) )
        if up_dist >= 0:
            if up_dist < s_np[s_idx,6]:
                return s_idx, up_dist
            if not found_up:
                found_up = True
                best_s_idx = int(s_idx); best_up_dist = up_dist + 0
    if not found_up:
        # this object is a leaf, so it doesn't have an upstream skeleton node
        # make up_dist negative
        s_idx = np.argmin( np.sum( (s_np[:,[1,2,3]] - coord.reshape((1,3)))**2, axis=1 ) - s_np[:,4] )
        xyz = np.array( spherical_2_cart(1, s_np[s_idx,7], s_np[s_idx,8]) )
        up_dist = np.sum(xyz * (coord - s_np[s_idx,[1,2,3]]) )
        return s_idx, -np.abs(up_dist)

    return best_s_idx, best_up_dist


def overlay_mito_on_skeleton(s_np, mito_coords, length_along_skeleton, mito_radius):
    '''
    Overlay mitochondria coordinates on dataframe and find overlapping objects.

    Args:
        df: Dataframe to overlay on (synapse sites or skeleton)
        df_graph: Directed graph of objects in df
        s_np: Neuron skeleton numpy array
        mito_coords: Coordinates of mitochondria to overlay
        orig_i_mito: Index of mitochondrion in original dataframe
        length_along_skeleton: Length of mitochondrion along skeleton
        mito_radius: Radius of mitochondrion

    Returns:
        list: Edge nodes where mitochondrion overlaps
    '''
    def get_overhang(on_idx, mito_coords, s_np, off_idx):
        if off_idx is None:
            # I am computing the up overhang on a leaf node, so the off_idx is not defined
            xyz = - np.array(spherical_2_cart(1, s_np[on_idx,7], s_np[on_idx,8] ))
            overhang = np.max( np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(s_np[on_idx,[1,2,3]] * xyz) )
        else:
            xyz = s_np[off_idx,[1,2,3]] - s_np[on_idx,[1,2,3]]
            xyz = xyz / np.sqrt(np.sum(xyz**2))
            overhang = np.max( np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(s_np[on_idx,[1,2,3]] * xyz) )
        return overhang

    def is_overlapping(skel_idx, this_coords):
        # are mito_coords overlapping on this_coords given the orientation specified by skel_idx
        xyz = np.array(spherical_2_cart(1, s_np[skel_idx,7], s_np[skel_idx,8]))
        proj_mito_dists = np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(this_coords * xyz)
        if np.max(proj_mito_dists) >= 0 and np.min(proj_mito_dists) <= 0:
            if np.min( np.sqrt(np.sum( (this_coords[np.newaxis,:] - mito_coords)**2, axis=1)) ) < s_np[skel_idx,4]*2:
                return True
        return False

    edge_nodes = []

    df_coords = s_np[:, 1:4]
    this_is_on = np.zeros(len(s_np), dtype=bool)
    obj_nodes = s_np[:, 0]
    down_obj_nodes = s_np[:, 5]
    dists = np.linalg.norm(df_coords - mito_coords.mean(axis=0), axis=1)
    for i_object in np.where( dists < 2 * length_along_skeleton )[0]:
        skel_idx = i_object
        this_is_on[i_object] = is_overlapping(skel_idx, df_coords[i_object])

    if np.any(this_is_on):
        if np.sum(this_is_on) > 1:
            num_nodes = np.zeros( (np.sum(this_is_on), np.sum(this_is_on)) )
            for ii, node in enumerate(obj_nodes[this_is_on]):
                for jj, target_node in enumerate(obj_nodes[this_is_on]):
                    if node != target_node:
                        obj_idx = np.where(obj_nodes == node)[0][0]
                        num_nodes[ii,jj] += 1
                        while down_obj_nodes[obj_idx] > 0 and down_obj_nodes[obj_idx] != target_node:
                            obj_idx = np.where(down_obj_nodes[obj_idx] == obj_nodes)[0][0]
                            num_nodes[ii,jj] += 1
                        if down_obj_nodes[obj_idx] == -1:
                            num_nodes[ii,jj] = 0
            # keep longest length that doesn't touch the root
            if np.max(num_nodes) == 0:
                # the touching nodes are on different segments of the arbor
                # take the closest synapse to the mitochondria as the true synapse on the mitochondrion
                this_dists = [ np.min( np.sqrt(np.sum( (this_coord[np.newaxis,:] - mito_coords)**2, axis=1)) ) for this_coord in df_coords[this_is_on] ]
                closest_i_object = np.where(this_is_on)[0][np.argmin(this_dists)]
                most_up_node, most_down_node = obj_nodes[closest_i_object], obj_nodes[closest_i_object]
            else:
                row, col = np.where( num_nodes == np.max(num_nodes) )
                most_up_node, most_down_node = obj_nodes[np.where(this_is_on)[0][row[0]]], obj_nodes[np.where(this_is_on)[0][col[0]]]
        else:
            most_up_node, most_down_node = obj_nodes[np.where(this_is_on)[0][0]], obj_nodes[np.where(this_is_on)[0][0]]

        path_idxs = get_down_idxs(s_np, most_up_node, s_np[:,0] == most_down_node, count_start = True)
        assert s_np[path_idxs[-1],0] == most_down_node and s_np[path_idxs[0],0] == most_up_node
        if s_np[path_idxs[0],0] not in s_np[:,5]: # most_up_node is a leaf
            edge_nodes.append([None, s_np[path_idxs[0],0], get_overhang(path_idxs[0], mito_coords, s_np, None), 'up'])
        for i_path, idx in enumerate(path_idxs):
            for off_idx in np.where( s_np[:,5] == s_np[idx,0] )[0]:
                if off_idx not in path_idxs:
                    # keep going up the off_idx until mito does not overlap with it
                    if i_path == 0:
                        # this is the most upstream node, so use projected distances
                        while is_overlapping(off_idx, s_np[off_idx,[1,2,3]]) and (s_np[off_idx,0] in s_np[:,5]):
                            off_idx = np.where( s_np[off_idx,0] == s_np[:,5] )[0][0]
                        on_idx = np.where( s_np[off_idx,5] == s_np[:,0] )[0][0]
                        edge_nodes.append([s_np[off_idx,0], s_np[on_idx,0], get_overhang(on_idx, mito_coords, s_np, off_idx), 'up'])
                    else:
                        is_on = np.sqrt(np.sum((s_np[off_idx,[1,2,3]] - s_np[idx,[1,2,3]])**2)) < mito_radius
                        while is_on and (s_np[off_idx,0] in s_np[:,5]):
                            off_idx = np.where( s_np[off_idx,0] == s_np[:,5] )[0][0]
                            is_on = np.sqrt(np.sum((s_np[off_idx,[1,2,3]] - s_np[idx,[1,2,3]])**2)) < mito_radius
                        on_idx = np.where( s_np[off_idx,5] == s_np[:,0] )[0][0]
                        overhang = mito_radius - np.sqrt(np.sum((s_np[on_idx,[1,2,3]] - s_np[idx,[1,2,3]])**2))
                        edge_nodes.append([s_np[off_idx,0], s_np[on_idx,0], overhang, 'up'])
        off_idx = np.where( s_np[path_idxs[-1],5] == s_np[:,0] )[0][0]
        edge_nodes.append([s_np[off_idx,0], s_np[path_idxs[-1],0], get_overhang(path_idxs[-1], mito_coords, s_np, off_idx), 'down'])
    else:
        up_idx = get_up_idx_dist(s_np, np.mean(mito_coords,axis=0))[0]
        down_idx = np.where( s_np[up_idx,5] == s_np[:,0] )[0][0]

        xyz = np.array(spherical_2_cart(1, s_np[up_idx,7], s_np[up_idx,8] ))
        overhang = np.min( np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(s_np[up_idx,[1,2,3]] * xyz) )
        edge_nodes.append([s_np[down_idx,0], s_np[up_idx,0], -overhang, 'up'])

        xyz = -np.array(spherical_2_cart(1, s_np[down_idx,7], s_np[down_idx,8] ))
        overhang = np.min( np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(s_np[down_idx,[1,2,3]] * xyz) )
        edge_nodes.append([s_np[up_idx,0], s_np[down_idx,0], -overhang, 'down'])

    return edge_nodes
