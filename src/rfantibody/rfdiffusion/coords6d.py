"""
6D Coordinate System for Protein Structure Representation

This module implements a 6D coordinate system for representing protein structures,
which encodes both distance and angular information between residues. This representation
is useful for structure prediction and comparison tasks.

The 6D system consists of:
- Distance (Cb-Cb)
- Omega dihedral (Ca-Cb-Cb-Ca)
- Theta dihedral (N-Ca-Cb-Cb)
- Phi angle (Ca-Cb-Cb)
"""

import numpy as np
import scipy
import scipy.spatial


def get_dihedrals(a, b, c, d):
    """
    Calculate dihedral angles defined by 4 sets of 3D points.

    A dihedral angle is the angle between two planes defined by four points.
    It measures the rotation around the central bond (b-c).

    Args:
        a (np.ndarray): First set of 3D coordinates, shape (N, 3)
        b (np.ndarray): Second set of 3D coordinates, shape (N, 3)
        c (np.ndarray): Third set of 3D coordinates, shape (N, 3)
        d (np.ndarray): Fourth set of 3D coordinates, shape (N, 3)

    Returns:
        np.ndarray: Dihedral angles in radians, shape (N,)
    """

    # Calculate vectors representing the bonds
    b0 = -1.0*(b - a)  # Vector from b to a
    b1 = c - b          # Vector from b to c (central bond)
    b2 = d - c          # Vector from c to d

    # Normalize the central bond vector
    b1 /= np.linalg.norm(b1, axis=-1)[:,None]

    # Project b0 and b2 onto the plane perpendicular to b1
    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    # Calculate the dihedral angle using the dot product and cross product
    x = np.sum(v*w, axis=-1)  # Cosine component
    y = np.sum(np.cross(b1, v)*w, axis=-1)  # Sine component

    return np.arctan2(y, x)


def get_angles(a, b, c):
    """
    Calculate planar angles defined by 3 sets of 3D points.

    A planar angle is the angle formed at point b by the rays ba and bc.

    Args:
        a (np.ndarray): First set of 3D coordinates, shape (N, 3)
        b (np.ndarray): Second set of 3D coordinates (vertex), shape (N, 3)
        c (np.ndarray): Third set of 3D coordinates, shape (N, 3)

    Returns:
        np.ndarray: Angles in radians, shape (N,)
    """

    # Compute normalized vectors from b to a and from b to c
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]

    # Compute the angle using the dot product
    x = np.sum(v*w, axis=1)

    # Clip values to handle numerical errors that might push x outside [-1, 1]
    return np.arccos(np.clip(x, -1.0, 1.0))


def get_coords6d(xyz, dmax):
    """
    Extract 6D coordinates from N, Ca, C atom coordinates.

    This function computes a 6D representation of protein structure that includes:
    - Distance between Cb atoms
    - Three angular features (omega, theta, phi) describing relative orientations

    The 6D representation is computed only for residue pairs within a specified distance
    threshold (dmax) for computational efficiency.

    Args:
        xyz (np.ndarray): Coordinates of backbone atoms, shape (3, N, 3)
                          where index 0=N, 1=Ca, 2=C
        dmax (float): Maximum Cb-Cb distance for computing 6D features

    Returns:
        tuple: A tuple containing:
            - dist6d (np.ndarray): Distance matrix between Cb atoms, shape (N, N)
            - omega6d (np.ndarray): Ca-Cb-Cb-Ca dihedral angles, shape (N, N)
            - theta6d (np.ndarray): N-Ca-Cb-Cb dihedral angles, shape (N, N)
            - phi6d (np.ndarray): Ca-Cb-Cb planar angles, shape (N, N)
            - mask (np.ndarray): Binary mask indicating computed pairs, shape (N, N)
    """

    nres = xyz.shape[1]

    # Extract backbone atoms: N, Ca, and C
    N  = xyz[0]
    Ca = xyz[1]
    C  = xyz[2]

    # Reconstruct Cb coordinates from N, Ca, C using ideal geometry
    # For glycine (which has no Cb), this creates a virtual Cb position
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    # These coefficients are derived from ideal backbone geometry
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # Use KD-tree for efficient neighbor search
    # Find all Cb-Cb pairs within distance threshold dmax
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # Extract pairs of contacting residues (excluding self-pairs)
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]  # First residue indices in pairs
    idx1 = idx[1]  # Second residue indices in pairs

    # Initialize distance matrix with large values (999.9 Angstroms)
    # Then fill in actual distances for contacting pairs
    dist6d = np.full((nres, nres), 999.9, dtype=np.float32)
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # Compute Ca-Cb-Cb-Ca dihedral angles (omega)
    # This describes the twist between two residues
    omega6d = np.zeros((nres, nres), dtype=np.float32)
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # Compute N-Ca-Cb-Cb dihedral angles (theta)
    # This describes the azimuthal orientation in spherical coordinates
    theta6d = np.zeros((nres, nres), dtype=np.float32)
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # Compute Ca-Cb-Cb planar angles (phi)
    # This describes the polar angle in spherical coordinates
    phi6d = np.zeros((nres, nres), dtype=np.float32)
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    # Create mask indicating which pairs were computed (1.0) vs. not computed (0.0)
    mask = np.zeros((nres, nres), dtype=np.float32)
    mask[idx0, idx1] = 1.0

    return dist6d, omega6d, theta6d, phi6d, mask
