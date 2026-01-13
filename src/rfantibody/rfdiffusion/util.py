"""
Utility functions for protein structure manipulation and geometric calculations.

This module provides a comprehensive suite of utilities for working with protein structures
in RFdiffusion, including:
    - Geometric transformations (rigid body motions, frames, rotations)
    - Protein coordinate manipulation (Cbeta generation, centering, alignment)
    - Torsion angle calculations (backbone and side-chain dihedrals)
    - Structure I/O (PDB reading/writing)
    - Contact and interaction analysis (hotspots, disulfides)
    - Diffusion-specific utilities (timestep calculations)

The functions use PyTorch tensors for efficient GPU computation and are designed
to work with both batch and single-structure inputs.
"""
import torch
import numpy as np

import scipy.sparse
import itertools

from rfantibody.rfdiffusion.chemical import *
from rfantibody.rfdiffusion.scoring import *

class Dotdict(dict):
    """
    Dictionary with dot notation access to attributes.

    Allows accessing dictionary keys as attributes, e.g., d.key instead of d['key'].
    This provides cleaner syntax for configuration objects and parameter dictionaries.

    Example:
        >>> d = Dotdict({'a': 1, 'b': 2})
        >>> d.a
        1
        >>> d.c = 3
        >>> d['c']
        3
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Alphabet for Chothia numbering and chain ID conversion
alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def generate_Cbeta(N,Ca,C):
    """
    Generate Cbeta coordinates from backbone atoms N, CA, and C.

    For glycine (which lacks a real Cbeta), this function generates a "pseudo-Cbeta"
    at the ideal position. Uses Rosetta's ideal geometry parameters.

    The Cbeta is constructed using a linear combination of vectors derived from the
    backbone geometry:
        a = (CA-N) × (C-CA)  [normal to the peptide plane]
        b = CA - N
        c = C - CA
        CB = -0.579*a + 0.569*b - 0.544*c + CA

    Args:
        N: N atom coordinates, shape (..., 3)
        Ca: CA atom coordinates, shape (..., 3)
        C: C atom coordinates, shape (..., 3)

    Returns:
        Cbeta coordinates with same shape as input (..., 3)

    Note:
        Coefficients match Rosetta's sidechain generator parameters for consistency.
    """
    b = Ca - N  # Vector from N to CA
    c = C - Ca  # Vector from CA to C
    a = torch.cross(b, c, dim=-1)  # Normal to the peptide plane

    # Ideal Cbeta position using Rosetta parameters
    # (Earlier version used slightly different coefficients: -0.583, 0.568, -0.541)
    Cb = -0.57910144*a + 0.5689693*b - 0.5441217*c + Ca

    return Cb

def center_and_realign_missing(xyz, mask_t):
    """
    Center structure at origin and realign missing residues to nearest valid residues.

    This function performs two operations:
    1. Centers the structure by moving the center of mass of CA atoms to the origin
    2. Moves missing/invalid residues to the position of their nearest valid residue

    This is useful for handling partially resolved structures or masked regions during
    diffusion, ensuring that all residues have reasonable coordinates.

    Args:
        xyz: Atomic coordinates, shape (L, 27, 3) where L is sequence length,
             27 is max atoms per residue, 3 is xyz coordinates
        mask_t: Atom mask indicating valid atoms, shape (L, 27)
                True where atom exists, False otherwise

    Returns:
        Centered and realigned coordinates, same shape as input (L, 27, 3)

    Note:
        A residue is considered valid if all backbone atoms (N, CA, C) are present.
    """
    L = xyz.shape[0]
    assert L > 0
    assert mask_t.shape[0] == L

    # Determine valid residues: those with all backbone atoms (N, CA, C) present
    mask = mask_t[:,:3].all(dim=-1)  # Shape: (L,), True for valid residues

    # Step 1: Center structure at origin using CA center of mass
    center_CA = (mask[...,None]*xyz[:,1]).sum(dim=0) / (mask[...,None].sum(dim=0) + 1e-5)  # (3,)
    xyz = torch.where(mask.view(L,1,1), xyz - center_CA.view(1, 1, 3), xyz)

    # Step 2: Move missing residues to the position of their nearest valid residue
    exist_in_xyz = torch.where(mask)[0]  # Indices of valid residues, shape (L_valid,)
    # Compute distance in sequence space to all valid residues
    seqmap = (torch.arange(L, device=xyz.device)[:,None] - exist_in_xyz[None,:]).abs()  # (L, L_valid)
    seqmap = torch.argmin(seqmap, dim=-1)  # Index of nearest valid residue, shape (L,)
    idx = torch.gather(exist_in_xyz, 0, seqmap)  # Map to actual residue index
    # Get CA position of nearest valid residue for each residue
    offset_CA = torch.gather(xyz[:,1], 0, idx.reshape(L,1).expand(-1,3))
    # Apply offset only to invalid residues
    xyz = torch.where(mask.view(L,1,1), xyz, xyz + offset_CA.reshape(L,1,3))

    return xyz

def th_ang_v(ab,bc,eps:float=1e-8):
    """
    Compute bond angle from two vectors.

    Calculates the angle between vectors ab and bc, returning both cosine and sine
    components. This representation is useful for differentiable geometric calculations.

    Args:
        ab: First vector, shape (..., 3)
        bc: Second vector, shape (..., 3)
        eps: Small epsilon for numerical stability (default: 1e-8)

    Returns:
        Angle representation as (cos, sin) pairs, shape (..., 2)
        First element is cos(angle), second is sin(angle)

    Note:
        Uses normalized vectors and clamps cosine to [-1, 1] for numerical stability.
    """
    def th_norm(x,eps:float=1e-8):
        """Compute L2 norm with numerical stability."""
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()

    def th_N(x,alpha:float=0):
        """Normalize vector."""
        return x/th_norm(x).add(alpha)

    # Normalize both vectors
    ab, bc = th_N(ab), th_N(bc)

    # Compute angle components
    cos_angle = torch.clamp((ab*bc).sum(-1), -1, 1)  # Dot product gives cosine
    sin_angle = torch.sqrt(1-cos_angle.square() + eps)  # Pythagorean identity

    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih_v(ab,bc,cd):
    """
    Compute dihedral angle from three consecutive bond vectors.

    Calculates the dihedral (torsion) angle defined by four points connected by
    vectors ab, bc, and cd. Returns (cos, sin) representation for differentiability.

    The dihedral angle is computed using the normal vectors to the planes formed by
    (ab, bc) and (bc, cd):
        n1 = (ab × bc) / |ab × bc|
        n2 = (bc × cd) / |bc × cd|
        cos(angle) = n1 · n2
        sin(angle) = (n1 × bc) · n2

    Args:
        ab: First bond vector (A to B), shape (..., 3)
        bc: Second bond vector (B to C), shape (..., 3)
        cd: Third bond vector (C to D), shape (..., 3)

    Returns:
        Dihedral angle as (cos, sin) pair, shape (..., 2)

    Note:
        All vectors are normalized before computation for numerical stability.
    """
    def th_cross(a,b):
        """Cross product with broadcasting."""
        a,b = torch.broadcast_tensors(a,b)
        return torch.cross(a,b, dim=-1)

    def th_norm(x,eps:float=1e-8):
        """L2 norm with numerical stability."""
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()

    def th_N(x,alpha:float=0):
        """Normalize vector."""
        return x/th_norm(x).add(alpha)

    # Normalize all bond vectors
    ab, bc, cd = th_N(ab), th_N(bc), th_N(cd)

    # Compute normal vectors to the two planes
    n1 = th_N(th_cross(ab,bc))  # Normal to plane 1 (ab, bc)
    n2 = th_N(th_cross(bc,cd))  # Normal to plane 2 (bc, cd)

    # Compute dihedral angle components
    sin_angle = (th_cross(n1,bc)*n2).sum(-1)  # Signed sine component
    cos_angle = (n1*n2).sum(-1)                # Cosine component

    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih(a,b,c,d):
    """
    Compute dihedral angle from four atom positions.

    Convenience wrapper for th_dih_v that computes bond vectors from atom positions.

    Args:
        a, b, c, d: Atom positions defining the dihedral angle, each shape (..., 3)

    Returns:
        Dihedral angle as (cos, sin) pair, shape (..., 2)

    Note:
        The dihedral is the angle around the b-c bond, viewed from b to c.
    """
    return th_dih_v(a-b, b-c, c-d)

def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    """
    Construct local coordinate frames from backbone N, CA, C atoms.

    Builds a rigid transformation (rotation matrix + translation) for each residue
    based on its backbone geometry. The frame is centered at CA with:
        - e1 pointing along CA->C direction
        - e2 in the peptide plane, perpendicular to e1
        - e3 perpendicular to the peptide plane (e1 × e2)

    This is the fundamental operation for converting between local and global coordinates
    in protein structure generation.

    Args:
        N: N atom coordinates, shape (B, L, 3) where B is batch size, L is length
        Ca: CA atom coordinates, shape (B, L, 3)
        C: C atom coordinates, shape (B, L, 3)
        non_ideal: If True, correct for non-ideal N-CA-C bond angles by applying
                   a rotation to match the ideal angle (~111°). This splits the
                   geometric error between CA-N and CA-C bonds, giving more accurate
                   sidechain placement. (default: False)
        eps: Small epsilon for numerical stability (default: 1e-8)

    Returns:
        R: Rotation matrices, shape (B, L, 3, 3)
           These are proper rotation matrices with det(R) = 1 and inv(R) = R.T
        Ca: Translation vectors (CA positions), shape (B, L, 3)

    Note:
        The non_ideal correction is particularly useful when building all-atom
        structures from idealized backbone geometry, as it prevents accumulation
        of geometric errors in the side chains.
    """
    B,L = N.shape[:2]

    # Build orthonormal frame from backbone atoms
    v1 = C-Ca   # Vector along CA-C bond
    v2 = N-Ca   # Vector along CA-N bond

    # e1: unit vector along CA-C
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)

    # e2: unit vector in peptide plane, perpendicular to e1
    # Project v2 onto e1 and subtract to get perpendicular component
    u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)

    # e3: perpendicular to peptide plane
    e3 = torch.cross(e1, e2, dim=-1)

    # Assemble rotation matrix from basis vectors
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1)  # Shape: (B, L, 3, 3)

    # Apply correction for non-ideal bond angles if requested
    if non_ideal:
        v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
        cosref = torch.sum(e1*v2, dim=-1)  # Cosine of current N-CA-C bond angle
        costgt = cos_ideal_NCAC.item()      # Cosine of ideal angle (~111°)

        # Compute rotation angle to correct the bond angle
        # Using half-angle formula: cos(2θ) = 2cos²(θ) - 1
        cos2del = torch.clamp(
            cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps),
            min=-1.0, max=1.0
        )
        cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
        sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)

        # Build rotation matrix to correct the angle
        Rp = torch.eye(3, device=N.device).repeat(B,L,1,1)
        Rp[:,:,0,0] = cosdel
        Rp[:,:,0,1] = -sindel
        Rp[:,:,1,0] = sindel
        Rp[:,:,1,1] = cosdel

        # Apply correction rotation
        R = torch.einsum('blij,bljk->blik', R, Rp)

    return R, Ca

def get_tor_mask(seq, torsion_indices, mask_in=None):
    """
    Generate mask for valid torsion angles in protein structures.

    Creates a boolean mask indicating which torsion angles are defined/valid for
    each residue. Handles:
        - 10 torsion types: omega, phi, psi, chi1-4, CB-bend, CB-twist, CG-bend
        - Amino acid specific constraints (e.g., GLY has no CB)
        - Missing atom detection from input mask

    Args:
        seq: Amino acid sequence as indices, shape (B, L)
        torsion_indices: Precomputed indices for torsion atoms, shape (22, 4, 4)
        mask_in: Optional atom mask for missing atoms, shape (B, L, 27)

    Returns:
        Boolean mask for valid torsions, shape (B, L, 10)
        Indices: [0:omega, 1:phi, 2:psi, 3-6:chi1-4, 7:CB-bend, 8:CB-twist, 9:CG-bend]
    """
    B,L = seq.shape[:2]
    tors_mask = torch.ones((B,L,10), dtype=torch.bool, device=seq.device)

    # Chi angles (3-6): valid only if all 4 atoms defining the torsion exist
    tors_mask[...,3:7] = torsion_indices[seq,:,-1] > 0

    # Boundary conditions
    tors_mask[:,0,1] = False   # No phi for first residue
    tors_mask[:,-1,0] = False  # No omega for last residue

    # Additional angle masks based on amino acid type
    tors_mask[:,:,7] = seq!=aa2num['GLY']  # CB-bend: GLY has no CB
    tors_mask[:,:,8] = seq!=aa2num['GLY']  # CB-twist: GLY has no CB
    # CG-bend: need side chain beyond CB (not GLY, ALA, UNK, MAS)
    tors_mask[:,:,9] = torch.logical_and(seq!=aa2num['GLY'], seq!=aa2num['ALA'])
    tors_mask[:,:,9] = torch.logical_and(tors_mask[:,:,9], seq!=aa2num['UNK'])
    tors_mask[:,:,9] = torch.logical_and(tors_mask[:,:,9], seq!=aa2num['MAS'])

    if mask_in != None:
        # Further mask based on missing atoms
        # Chi angles: require all 4 defining atoms present
        ti0 = torch.gather(mask_in,2,torsion_indices[seq,:,0])
        ti1 = torch.gather(mask_in,2,torsion_indices[seq,:,1])
        ti2 = torch.gather(mask_in,2,torsion_indices[seq,:,2])
        ti3 = torch.gather(mask_in,2,torsion_indices[seq,:,3])
        is_valid = torch.stack((ti0, ti1, ti2, ti3), dim=-2).all(dim=-1)
        tors_mask[...,3:7] = torch.logical_and(tors_mask[...,3:7], is_valid)

        tors_mask[:,:,7] = torch.logical_and(tors_mask[:,:,7], mask_in[:,:,4])  # CB exists?
        tors_mask[:,:,8] = torch.logical_and(tors_mask[:,:,8], mask_in[:,:,4])  # CB exists?
        tors_mask[:,:,9] = torch.logical_and(tors_mask[:,:,9], mask_in[:,:,5])  # XG exists?

    return tors_mask

def get_torsions(xyz_in, seq, torsion_indices, torsion_can_flip, ref_angles, mask_in=None):
    """
    Compute all torsion angles from atomic coordinates.

    Calculates 10 types of torsion/angle descriptors for each residue:
        - Backbone: omega (ω), phi (φ), psi (ψ)
        - Side-chain: chi1-4 (χ1-χ4)
        - Geometry: CB-bend, CB-twist, CG-bend

    Before computing torsions, backbone atoms (N, CA, C) are idealized to standard
    geometry to avoid numerical issues. Also generates "alternate" torsions for
    symmetric side chains.

    Args:
        xyz_in: Atomic coordinates, shape (B, L, 27, 3)
        seq: Amino acid sequence indices, shape (B, L)
        torsion_indices: Atom indices for torsion calculation, shape (22, 4, 4)
        torsion_can_flip: Boolean mask for flippable torsions, shape (22, 10)
        ref_angles: Reference angles for geometric descriptors, shape (22, 3, 2)
        mask_in: Optional atom mask, shape (B, L, 27)

    Returns:
        torsions: Torsion angles as (cos, sin) pairs, shape (B, L, 10, 2)
        torsions_alt: Alternate torsions for symmetric side chains, shape (B, L, 10, 2)
        tors_mask: Mask for valid torsions, shape (B, L, 10)
        tors_planar: Mask for planar torsions (should be 0° or 180°), shape (B, L, 10)
    """
    B,L = xyz_in.shape[:2]

    tors_mask = get_tor_mask(seq, torsion_indices, mask_in)
    
    # torsions to restrain to 0 or 180degree
    tors_planar = torch.zeros((B, L, 10), dtype=torch.bool, device=xyz_in.device)
    tors_planar[:,:,5] = seq == aa2num['TYR'] # TYR chi 3 should be planar

    # idealize given xyz coordinates before computing torsion angles
    xyz = xyz_in.clone()
    Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:])
    Nideal = torch.tensor([-0.5272, 1.3593, 0.000], device=xyz_in.device)
    Cideal = torch.tensor([1.5233, 0.000, 0.000], device=xyz_in.device)
    xyz[...,0,:] = torch.einsum('brij,j->bri', Rs, Nideal) + Ts
    xyz[...,2,:] = torch.einsum('brij,j->bri', Rs, Cideal) + Ts

    torsions = torch.zeros( (B,L,10,2), device=xyz.device )
    # avoid undefined angles for H generation
    torsions[:,0,1,0] = 1.0
    torsions[:,-1,0,0] = 1.0

    # omega
    torsions[:,:-1,0,:] = th_dih(xyz[:,:-1,1,:],xyz[:,:-1,2,:],xyz[:,1:,0,:],xyz[:,1:,1,:])
    # phi
    torsions[:,1:,1,:] = th_dih(xyz[:,:-1,2,:],xyz[:,1:,0,:],xyz[:,1:,1,:],xyz[:,1:,2,:])
    # psi
    torsions[:,:,2,:] = -1 * th_dih(xyz[:,:,0,:],xyz[:,:,1,:],xyz[:,:,2,:],xyz[:,:,3,:])

    # chis
    ti0 = torch.gather(xyz,2,torsion_indices[seq,:,0,None].repeat(1,1,1,3))
    ti1 = torch.gather(xyz,2,torsion_indices[seq,:,1,None].repeat(1,1,1,3))
    ti2 = torch.gather(xyz,2,torsion_indices[seq,:,2,None].repeat(1,1,1,3))
    ti3 = torch.gather(xyz,2,torsion_indices[seq,:,3,None].repeat(1,1,1,3))
    torsions[:,:,3:7,:] = th_dih(ti0,ti1,ti2,ti3)
    
    # CB bend
    NC = 0.5*( xyz[:,:,0,:3] + xyz[:,:,2,:3] )
    CA = xyz[:,:,1,:3]
    CB = xyz[:,:,4,:3]
    t = th_ang_v(CB-CA,NC-CA)
    t0 = ref_angles[seq][...,0,:]
    torsions[:,:,7,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )
    
    # CB twist
    NCCA = NC-CA
    NCp = xyz[:,:,2,:3] - xyz[:,:,0,:3]
    NCpp = NCp - torch.sum(NCp*NCCA, dim=-1, keepdim=True)/ torch.sum(NCCA*NCCA, dim=-1, keepdim=True) * NCCA
    t = th_ang_v(CB-CA,NCpp)
    t0 = ref_angles[seq][...,1,:]
    torsions[:,:,8,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )

    # CG bend
    CG = xyz[:,:,5,:3]
    t = th_ang_v(CG-CB,CA-CB)
    t0 = ref_angles[seq][...,2,:]
    torsions[:,:,9,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )
    
    mask0 = torch.isnan(torsions[...,0]).nonzero()
    mask1 = torch.isnan(torsions[...,1]).nonzero()
    torsions[mask0[:,0],mask0[:,1],mask0[:,2],0] = 1.0
    torsions[mask1[:,0],mask1[:,1],mask1[:,2],1] = 0.0

    # alt chis
    torsions_alt = torsions.clone()
    torsions_alt[torsion_can_flip[seq,:]] *= -1

    return torsions, torsions_alt, tors_mask, tors_planar

# Process ideal frames for kinematic structure building
def make_frame(X, Y):
    """
    Construct orthonormal coordinate frame from two vectors.

    Creates a right-handed orthonormal basis using Gram-Schmidt orthogonalization:
        1. Normalize X to get first basis vector
        2. Remove X component from Y and normalize to get second basis vector
        3. Compute cross product to get third basis vector

    Args:
        X: First vector, shape (3,)
        Y: Second vector (not necessarily orthogonal to X), shape (3,)

    Returns:
        Rotation matrix with columns [X_norm, Y_orth_norm, Z_norm], shape (3, 3)
        Forms right-handed orthonormal coordinate system

    Note:
        Used for building local coordinate frames in kinematic chain calculations.
    """
    # Normalize X to get first basis vector
    Xn = X / torch.linalg.norm(X)

    # Orthogonalize Y with respect to X (Gram-Schmidt)
    Y = Y - torch.dot(Y, Xn) * Xn
    Yn = Y / torch.linalg.norm(Y)

    # Compute third basis vector via cross product
    Z = torch.cross(Xn, Yn)
    Zn = Z / torch.linalg.norm(Z)

    return torch.stack((Xn,Yn,Zn), dim=-1)

def cross_product_matrix(u):
    B, L = u.shape[:2]
    matrix = torch.zeros((B, L, 3, 3), device=u.device)
    matrix[:,:,0,1] = -u[...,2]
    matrix[:,:,0,2] = u[...,1]
    matrix[:,:,1,0] = u[...,2]
    matrix[:,:,1,2] = -u[...,0]
    matrix[:,:,2,0] = -u[...,1]
    matrix[:,:,2,1] = u[...,0]
    return matrix

# PDB file writing utilities
def writepdb(filename, atoms, seq, binderlen=None, idx_pdb=None, bfacts=None, chain_idx=None):
    """
    Write protein structure to PDB file.

    Supports multiple representations:
        - CA-only (atoms shape: L x 3)
        - Backbone (N, CA, C): (L x 3 x 3)
        - Backbone + O: (L x 4 x 3)
        - Full atom: (L x 14 x 3) or (L x 27 x 3)

    Args:
        filename: Output PDB filename
        atoms: Atomic coordinates, shape (L, [3|3|4|14|27], 3)
        seq: Amino acid sequence as indices, shape (L,)
        binderlen: Optional length of binder chain for multi-chain output
        idx_pdb: Optional custom residue numbering, shape (L,)
        bfacts: Optional B-factors (temperature factors), shape (L,)
        chain_idx: Optional chain identifiers for each residue, shape (L,)

    Note:
        If binderlen is provided, residues 0:binderlen are chain A, rest are chain B.
        Handles histidine protonation state automatically.
    """
    f = open(filename,"w")
    ctr = 1
    scpu = seq.cpu().squeeze()
    atomscpu = atoms.cpu().squeeze()
    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
    for i,s in enumerate(scpu):
        if chain_idx is None:
            if binderlen is not None:
                if i < binderlen:
                    chain = 'A'
                else:
                    chain = 'B'
            elif binderlen is None:
                chain = 'A'
        else:
            chain = chain_idx[i]
        if (len(atomscpu.shape)==2):
            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, " CA ", num2aa[s],
                    chain, idx_pdb[i], atomscpu[i,0], atomscpu[i,1], atomscpu[i,2],
                    1.0, Bfacts[i] ) )
            ctr += 1
        elif atomscpu.shape[1]==3:
            for j,atm_j in enumerate([" N  "," CA "," C  "]):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm_j, num2aa[s],
                        chain, idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                        1.0, Bfacts[i] ) )
                ctr += 1
        elif atomscpu.shape[1]==4:
            for j,atm_j in enumerate([" N  "," CA "," C  "," O  "]):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm_j, num2aa[s],
                        chain, idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                        1.0, Bfacts[i] ) ) 
                ctr += 1
            
        else:
            natoms = atomscpu.shape[1]
            if (natoms!=14 and natoms!=27):
                print ('bad size!', atoms.shape)
                assert(False)
            atms = aa2long[s]
            # his prot hack
            if (s==8 and torch.linalg.norm( atomscpu[i,9,:]-atomscpu[i,5,:] ) < 1.7):
                atms = (
                    " N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",
                      None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1",
                    " HD1",  None,  None,  None,  None,  None,  None) # his_d

            for j,atm_j in enumerate(atms):
                if (j<natoms and atm_j is not None): # and not torch.isnan(atomscpu[i,j,:]).any()):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm_j, num2aa[s],
                        chain, idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                        1.0, Bfacts[i] ) )
                    ctr += 1


# resolve tip atom indices
tip_indices = torch.full((22,), 0)
for i in range(22):
    tip_atm = aa2tip[i]
    atm_long = aa2long[i]
    tip_indices[i] = atm_long.index(tip_atm)

# resolve torsion indices
torsion_indices = torch.full((22,4,4),0)
torsion_can_flip = torch.full((22,10),False,dtype=torch.bool)
for i in range(22):
    i_l, i_a = aa2long[i], aa2longalt[i]
    for j in range(4):
        if torsions[i][j] is None:
            continue
        for k in range(4):
            a = torsions[i][j][k]
            torsion_indices[i,j,k] = i_l.index(a)
            if (i_l.index(a) != i_a.index(a)):
                torsion_can_flip[i,3+j] = True ##bb tors never flip
# HIS is a special case
torsion_can_flip[8,4]=False

# build the mapping from atoms in the full rep (Nx27) to the "alternate" rep
allatom_mask = torch.zeros((22,27), dtype=torch.bool)
long2alt = torch.zeros((22,27), dtype=torch.long)
for i in range(22):
    i_l, i_lalt = aa2long[i],  aa2longalt[i]
    for j,a in enumerate(i_l):
        if (a is None):
            long2alt[i,j] = j
        else:
            long2alt[i,j] = i_lalt.index(a)
            allatom_mask[i,j] = True

# bond graph traversal
num_bonds = torch.zeros((22,27,27), dtype=torch.long)
for i in range(22):
    num_bonds_i = np.zeros((27,27))
    for (bnamei,bnamej) in aabonds[i]:
        bi,bj = aa2long[i].index(bnamei),aa2long[i].index(bnamej)
        num_bonds_i[bi,bj] = 1
    num_bonds_i = scipy.sparse.csgraph.shortest_path (num_bonds_i,directed=False)
    num_bonds_i[num_bonds_i>=4] = 4
    num_bonds[i,...] = torch.tensor(num_bonds_i)


# LJ/LK scoring parameters
ljlk_parameters = torch.zeros((22,27,5), dtype=torch.float)
lj_correction_parameters = torch.zeros((22,27,4), dtype=bool) # donor/acceptor/hpol/disulf
for i in range(22):
    for j,a in enumerate(aa2type[i]):
        if (a is not None):
            ljlk_parameters[i,j,:] = torch.tensor( type2ljlk[a] )
            lj_correction_parameters[i,j,0] = (type2hb[a]==HbAtom.DO)+(type2hb[a]==HbAtom.DA)
            lj_correction_parameters[i,j,1] = (type2hb[a]==HbAtom.AC)+(type2hb[a]==HbAtom.DA)
            lj_correction_parameters[i,j,2] = (type2hb[a]==HbAtom.HP)
            lj_correction_parameters[i,j,3] = (a=="SH1" or a=="HS")

# hbond scoring parameters
def donorHs(D,bonds,atoms):
    dHs = []
    for (i,j) in bonds:
        if (i==D):
            idx_j = atoms.index(j)
            if (idx_j>=14):  # if atom j is a hydrogen
                dHs.append(idx_j)
        if (j==D):
            idx_i = atoms.index(i)
            if (idx_i>=14):  # if atom j is a hydrogen
                dHs.append(idx_i)
    assert (len(dHs)>0)
    return dHs

def acceptorBB0(A,hyb,bonds,atoms):
    if (hyb == HbHybType.SP2):
        for (i,j) in bonds:
            if (i==A):
                B = atoms.index(j)
                if (B<14):
                    break
            if (j==A):
                B = atoms.index(i)
                if (B<14):
                    break
        for (i,j) in bonds:
            if (i==atoms[B]):
                B0 = atoms.index(j)
                if (B0<14):
                    break
            if (j==atoms[B]):
                B0 = atoms.index(i)
                if (B0<14):
                    break
    elif (hyb == HbHybType.SP3 or hyb == HbHybType.RING):
        for (i,j) in bonds:
            if (i==A):
                B = atoms.index(j)
                if (B<14):
                    break
            if (j==A):
                B = atoms.index(i)
                if (B<14):
                    break
        for (i,j) in bonds:
            if (i==A and j!=atoms[B]):
                B0 = atoms.index(j)
                break
            if (j==A and i!=atoms[B]):
                B0 = atoms.index(i)
                break

    return B,B0


hbtypes = torch.full((22,27,3),-1, dtype=torch.long) # (donortype, acceptortype, acchybtype)
hbbaseatoms = torch.full((22,27,2),-1, dtype=torch.long) # (B,B0) for acc; (D,-1) for don
hbpolys = torch.zeros((HbDonType.NTYPES,HbAccType.NTYPES,3,15)) # weight,xmin,xmax,ymin,ymax,c9,...,c0

for i in range(22):
    for j,a in enumerate(aa2type[i]):
        if (a in type2dontype):
            j_hs = donorHs(aa2long[i][j],aabonds[i],aa2long[i])
            for j_h in j_hs:
                hbtypes[i,j_h,0] = type2dontype[a]
                hbbaseatoms[i,j_h,0] = j
        if (a in type2acctype):
            j_b, j_b0 = acceptorBB0(aa2long[i][j],type2hybtype[a],aabonds[i],aa2long[i])
            hbtypes[i,j,1] = type2acctype[a]
            hbtypes[i,j,2] = type2hybtype[a]
            hbbaseatoms[i,j,0] = j_b
            hbbaseatoms[i,j,1] = j_b0

for i in range(HbDonType.NTYPES):
    for j in range(HbAccType.NTYPES):
        weight = dontype2wt[i]*acctype2wt[j]

        pdist,pbah,pahd = hbtypepair2poly[(i,j)]
        xrange,yrange,coeffs = hbpolytype2coeffs[pdist]
        hbpolys[i,j,0,0] = weight
        hbpolys[i,j,0,1:3] = torch.tensor(xrange)
        hbpolys[i,j,0,3:5] = torch.tensor(yrange)
        hbpolys[i,j,0,5:] = torch.tensor(coeffs)
        xrange,yrange,coeffs = hbpolytype2coeffs[pahd]
        hbpolys[i,j,1,0] = weight
        hbpolys[i,j,1,1:3] = torch.tensor(xrange)
        hbpolys[i,j,1,3:5] = torch.tensor(yrange)
        hbpolys[i,j,1,5:] = torch.tensor(coeffs)
        xrange,yrange,coeffs = hbpolytype2coeffs[pbah]
        hbpolys[i,j,2,0] = weight
        hbpolys[i,j,2,1:3] = torch.tensor(xrange)
        hbpolys[i,j,2,3:5] = torch.tensor(yrange)
        hbpolys[i,j,2,5:] = torch.tensor(coeffs)

# kinematic parameters
base_indices = torch.full((22,27),0, dtype=torch.long)
xyzs_in_base_frame = torch.ones((22,27,4))
RTs_by_torsion = torch.eye(4).repeat(22,7,1,1)
reference_angles = torch.ones((22,3,2))

for i in range(22):
    i_l = aa2long[i]
    for name, base, coords in ideal_coords[i]:
        idx = i_l.index(name)
        base_indices[i,idx] = base
        xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

    # omega frame
    RTs_by_torsion[i,0,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,0,:3,3] = torch.zeros(3)

    # phi frame
    RTs_by_torsion[i,1,:3,:3] = make_frame(
        xyzs_in_base_frame[i,0,:3] - xyzs_in_base_frame[i,1,:3],
        torch.tensor([1.,0.,0.])
    )
    RTs_by_torsion[i,1,:3,3] = xyzs_in_base_frame[i,0,:3]

    # psi frame
    RTs_by_torsion[i,2,:3,:3] = make_frame(
        xyzs_in_base_frame[i,2,:3] - xyzs_in_base_frame[i,1,:3],
        xyzs_in_base_frame[i,1,:3] - xyzs_in_base_frame[i,0,:3]
    )
    RTs_by_torsion[i,2,:3,3] = xyzs_in_base_frame[i,2,:3]

    # chi1 frame
    if torsions[i][0] is not None:
        a0,a1,a2 = torsion_indices[i,0,0:3]
        RTs_by_torsion[i,3,:3,:3] = make_frame(
            xyzs_in_base_frame[i,a2,:3]-xyzs_in_base_frame[i,a1,:3],
            xyzs_in_base_frame[i,a0,:3]-xyzs_in_base_frame[i,a1,:3],
        )
        RTs_by_torsion[i,3,:3,3] = xyzs_in_base_frame[i,a2,:3]

    # chi2~4 frame
    for j in range(1,4):
        if torsions[i][j] is not None:
            a2 = torsion_indices[i,j,2]
            if ((i==18 and j==2) or (i==8 and j==2)):  # TYR CZ-OH & HIS CE1-HE1 a special case
                a0,a1 = torsion_indices[i,j,0:2]
                RTs_by_torsion[i,3+j,:3,:3] = make_frame(
                    xyzs_in_base_frame[i,a2,:3]-xyzs_in_base_frame[i,a1,:3],
                    xyzs_in_base_frame[i,a0,:3]-xyzs_in_base_frame[i,a1,:3] )
            else:
                RTs_by_torsion[i,3+j,:3,:3] = make_frame(
                    xyzs_in_base_frame[i,a2,:3],
                    torch.tensor([-1.,0.,0.]), )
            RTs_by_torsion[i,3+j,:3,3] = xyzs_in_base_frame[i,a2,:3]
            

    # CB/CG angles
    NCr = 0.5*(xyzs_in_base_frame[i,0,:3]+xyzs_in_base_frame[i,2,:3])
    CAr = xyzs_in_base_frame[i,1,:3]
    CBr = xyzs_in_base_frame[i,4,:3]
    CGr = xyzs_in_base_frame[i,5,:3]
    reference_angles[i,0,:]=th_ang_v(CBr-CAr,NCr-CAr)
    NCp = xyzs_in_base_frame[i,2,:3]-xyzs_in_base_frame[i,0,:3]
    NCpp = NCp - torch.dot(NCp,NCr)/ torch.dot(NCr,NCr) * NCr
    reference_angles[i,1,:]=th_ang_v(CBr-CAr,NCpp)
    reference_angles[i,2,:]=th_ang_v(CGr,torch.tensor([-1.,0.,0.]))

N_BACKBONE_ATOMS = 3
N_HEAVY = 14 
def writepdb_multi(filename, atoms_stack, bfacts, seq_stack, backbone_only=False, chain_ids=None, use_hydrogens=True, return_pdblines=False):
    """
    Function for writing multiple structural states of the same sequence into a single 
    pdb file. 
    """

    pdblines = []

    if seq_stack.ndim != 2:
        T = atoms_stack.shape[0]
        seq_stack = torch.tile(seq_stack, (T,1))
    seq_stack = seq_stack.cpu()
    for atoms, scpu in zip(atoms_stack, seq_stack):
        ctr = 1
        atomscpu = atoms.cpu()
        Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
        for i,s in enumerate(scpu):
            atms = aa2long[s]
            for j,atm_j in enumerate(atms):

                if backbone_only and j >= N_BACKBONE_ATOMS:
                    break
                if not use_hydrogens and j >= N_HEAVY:
                    break 
                if (atm_j is None) or (torch.all(torch.isnan(atomscpu[i,j]))):
                    continue
                chain_id = 'A'
                if chain_ids is not None:
                    chain_id = chain_ids[i]
                pdblines.append("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, atm_j, num2aa[s],
                    chain_id, i+1, atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                    1.0, Bfacts[i] ) )
                ctr += 1

        pdblines.append('ENDMDL\n')

    if return_pdblines:
        return pdblines

    with open(filename,"w") as f:
        for line in pdblines:
            f.write(line)

def get_mu_xt_x0(xt, px0, t, schedule, alphabar_schedule, eps=1e-6):
    """
    Given xt, predicted x0 and the timestep t, give mu of x(t-1)
    Assumes t is 0 indexed
    """
    #sigma is predefined from beta. Often referred to as beta tilde t
    t_idx = t-1

    sigma = ((1-alphabar_schedule[t_idx-1])/(1-alphabar_schedule[t_idx]))*schedule[t_idx]

    xt_ca = xt[:,1,:]
    px0_ca = px0[:,1,:]

    a = ((torch.sqrt(alphabar_schedule[t_idx-1] + eps)*schedule[t_idx])/(1-alphabar_schedule[t_idx]))*px0_ca
    b = ((torch.sqrt(1-schedule[t_idx] + eps)*(1-alphabar_schedule[t_idx-1]))/(1-alphabar_schedule[t_idx]))*xt_ca
    mu = a + b

    return mu, sigma

def get_disulphides(xyz, seq):
    """
    Function to get disulphides in a structure.
    Defined as Cbeta-Cbeta distance <5.5A, between two cysteines.
    Input:
        xyz coordinates [L, 27, 3]
        seq [L]
    Output:
        2D cystogram [L, L] e (0,1)
    
    5.5A cutoff is from here:
    https://febs.onlinelibrary.wiley.com/doi/full/10.1016/j.febslet.2013.11.024
    """
    L = xyz.shape[2]
    FakeCB=generate_Cbeta(xyz[0,0,:,0],xyz[0,0,:,1],xyz[0,0,:,2])
    gly = torch.where(seq[0]==7, 1, 0).bool()
    xyz[0,0,gly,4] = FakeCB[gly]
    dist=torch.cdist(xyz[0,0,:,4],xyz[0,0,:,4])
    x=torch.zeros_like(dist)
    y=torch.zeros_like(dist)
    cys_mask=torch.where(seq[0]==4, 1, 0).bool()
    x[:,cys_mask] = 1
    y[cys_mask,:] = 1
    cys=x*y
    upper_map = torch.where(dist < 6.5, 1, 0)
    lower_map = torch.where(dist > 3.5, 1, 0)
    contact_map=upper_map*lower_map
    cysteine_map=contact_map * cys
    return cysteine_map.float(), torch.max(cysteine_map, dim=-1)[0].bool()

def get_cdr_hotspots(
    xyz : torch.tensor,
    atom_mask : torch.tensor,
    target_mask : torch.tensor,
    loop_mask=None,
    cutoff=8,
    topk=1
    ):
    """
    Function to get hotspots on the target chain that are
    within some distance of the target (i.e. are contacting)
    inputs:
        - xyz: Concatenated target & binder coordinates [L, 14/27, 3]
        - atom_mask: Atom mask. True where atom is present, False
            otherwise [L, 14/27]
        - target_mask: True where target, False where binder [L]
        - loop_mask: True where CDR loop, False where not [L]
        - cutoff: The Cb-Cb distance cutoff used in hotspot determination (float)
        - topk: The number of loop neighbors to average over to determine the
                cutoff distance of a residue (default: 1; take the closest residue)
    output:
        - hotspot_mask: True where hotspot (on target), False where not [L]
    """
    # Sanity checks
    assert xyz.shape[:2]==atom_mask.shape[:2]
    assert xyz.shape[0] == target_mask.shape[0],\
        'Target mask is the wrong size for xyz'
    if loop_mask is not None:
        assert xyz.shape[0] == loop_mask.shape[0],\
            'Loop mask is the wrong size for xyz'
        assert loop_mask.dtype == torch.bool,\
            'Loop mask should be boolean'
        assert (target_mask * loop_mask).sum() == 0,\
            'Target mask and loop mask overlab'
    assert target_mask.dtype == torch.bool,\
        'Target mask should be boolean'
    assert topk > 0, 'topk must be a positive integer'

    # 1) Generate Cbeta. NB idx 4 == Cb
    Cb = generate_Cbeta(N=xyz[:,0],Ca=xyz[:,1],C=xyz[:,2])
    xyz[~atom_mask[:,4],4] = Cb[~atom_mask[:,4]]

    # 2) Get Cbeta-Cbeta dist
    if loop_mask is None or loop_mask.sum() == 0: # edge case where no loops
        dist=torch.cdist(xyz[target_mask,4],xyz[~target_mask,4]) # [target_L, loop_L]
    else:
        dist=torch.cdist(xyz[target_mask,4], xyz[loop_mask,4])

    # If topk is 1 then we do default neighbor determination
    if topk == 1:
        dist=torch.min(dist, dim=1).values
    else:
        # If topk is greater than 1, we will average over the K closest
        # loop residues for each target residue to determine the distance
        # used in the hotspot determination

        # The motivation here is that residues which are contacted by multiple loop
        # residues will be more central to the interface and a richer source of
        # information about where the dock should form
        closestK=torch.topk(dist, k=topk, largest=False, dim=1).values

        # Average over the K closest residues to determine the dist
        dist=torch.mean(closestK, dim=1)

    assert dist.shape[0] == (target_mask).sum()

    L = xyz.shape[0]
    # 3) Get hotspots
    hotspots=torch.zeros(L).bool()
    # torch.where doesn't work on cpu
    convert=lambda a: torch.from_numpy(np.array(a)).bool()
    hotspots[target_mask] = convert([True if i < cutoff else False for i in dist])

    return hotspots

def write_chothia(pdb, filename):
    """
    Function to write chothia pdbs
    from pdb object
    Gets heavy atoms, and masks out missing
    Writes pdb file
    """
    f = open(filename,"w")

    xyz=pdb['xyz'][:,:14] # don't write hydrogens
    mask=pdb['mask']
    seq=pdb['seq']
    idx=pdb['pdb_idx']
    ctr=1
    for i, s in enumerate(seq):
        # get atoms for this residue
        atms = aa2long[s]
        for j,atm_j in enumerate(atms):
            if atm_j is not None and not np.isnan(xyz[i,j,0]):
                f.write ("%-6s%5s %4s %3s %s%5s   %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm_j, num2aa[s],
                        idx[i][0], idx[i][1], xyz[i,j,0], xyz[i,j,1], xyz[i,j,2],
                        1.0, 1.0 ) )
                ctr += 1

def convert_chothia_idx(idx) -> torch.tensor:
    """Function to convert idx from chothia parser into integer
    idx suitable for RFdiffusion positional encoding"""
    # First, group by whether they are integers or strings
    idx = [int(i) if not str(i)[-1].isalpha() else str(i)[-1] for i in idx]
    idx = [list(g) for _, g in itertools.groupby(idx, type)]
    
    # Second, split integers lists covering multiple chains
    out_idx = []
    for i in idx:
        if not isinstance(i[0], int):
            out_idx.append(i)
        else:
            temp = [i[0]]
            for t, j in enumerate(i):
                if t > 0:
                    if t and j < i[t-1]:
                        out_idx.append(temp)
                        temp = [j] 
                    else:
                        temp.append(j)
            if temp:
                out_idx.append(temp)
    assert len([i for j in out_idx for i in j])==len([i for j in idx for i in j]) 
    
    # Third, make an integer 0-indexed tensor of this index
    int_idx, counter = [], 0
    for ls in out_idx:
        if type(ls[0]) == int:
            for i,val in enumerate(ls[:-1]):
                int_idx.append(counter)
                counter += ls[i+1]-val
            int_idx.append(counter)
            counter+=1
        else:
            for i in ls: 
                int_idx.append(counter+alphabet.index(i.upper()))
            counter += alphabet.index(ls[-1].upper())+1
    assert len(int_idx) == sum(map(len, out_idx))
    return torch.tensor(int_idx)
