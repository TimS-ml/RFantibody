"""
Auxiliary Prediction Networks for Protein Structure

This module contains neural network architectures for predicting various structural
properties from learned representations, including:
- Distance and orientation (6D coordinates)
- Amino acid sequence from masked positions
- Local distance difference test (LDDT) scores
- Experimentally resolved residue predictions

These networks serve as auxiliary tasks during training to improve representation
learning and provide interpretable structural outputs.
"""

import torch
import torch.nn as nn


class DistanceNetwork(nn.Module):
    """
    Predict distance and orientation parameters between residue pairs.

    This network predicts the 6D coordinate representation:
    - Distance (dist): Cb-Cb distance, discretized into 37 bins
    - Omega: Ca-Cb-Cb-Ca dihedral angle, discretized into 37 bins (symmetric)
    - Theta: N-Ca-Cb-Cb dihedral angle, discretized into 37 bins (asymmetric)
    - Phi: Ca-Cb-Cb planar angle, discretized into 19 bins (asymmetric)

    Args:
        n_feat (int): Number of input features from pair representation
        p_drop (float): Dropout probability (currently unused but kept for interface)
    """
    def __init__(self, n_feat, p_drop=0.1):
        super(DistanceNetwork, self).__init__()
        # Project to symmetric predictions: distance (37 bins) + omega (37 bins)
        self.proj_symm = nn.Linear(n_feat, 37*2)
        # Project to asymmetric predictions: theta (37 bins) + phi (19 bins)
        self.proj_asymm = nn.Linear(n_feat, 37+19)
    
        self.reset_parameter()
    
    def reset_parameter(self):
        """
        Initialize projection layers with zeros.

        This initialization ensures that the network starts with no bias toward
        any particular structural prediction, allowing it to learn from data.
        """
        nn.init.zeros_(self.proj_symm.weight)
        nn.init.zeros_(self.proj_asymm.weight)
        nn.init.zeros_(self.proj_symm.bias)
        nn.init.zeros_(self.proj_asymm.bias)

    def forward(self, x):
        """
        Predict distance and orientation logits from pair features.

        Args:
            x (torch.Tensor): Pair features, shape (B, L, L, C)
                             B: batch size, L: sequence length, C: feature dimension

        Returns:
            tuple: Four tensors containing:
                - logits_dist: Distance predictions, shape (B, 37, L, L)
                - logits_omega: Omega dihedral predictions, shape (B, 37, L, L)
                - logits_theta: Theta dihedral predictions, shape (B, 37, L, L)
                - logits_phi: Phi angle predictions, shape (B, 19, L, L)
        """
        # Predict theta and phi (asymmetric - direction matters)
        logits_asymm = self.proj_asymm(x)
        logits_theta = logits_asymm[:,:,:,:37].permute(0,3,1,2)
        logits_phi = logits_asymm[:,:,:,37:].permute(0,3,1,2)

        # Predict distance and omega (symmetric - same for (i,j) and (j,i))
        logits_symm = self.proj_symm(x)
        # Symmetrize by averaging predictions in both directions
        logits_symm = logits_symm + logits_symm.permute(0,2,1,3)
        logits_dist = logits_symm[:,:,:,:37].permute(0,3,1,2)
        logits_omega = logits_symm[:,:,:,37:].permute(0,3,1,2)

        return logits_dist, logits_omega, logits_theta, logits_phi


class MaskedTokenNetwork(nn.Module):
    """
    Predict amino acid identities for masked sequence positions.

    This network performs a masked language modeling task, predicting which
    amino acid should appear at positions that were masked during training.
    This encourages learning meaningful sequence representations.

    Args:
        n_feat (int): Number of input features from sequence representation
        p_drop (float): Dropout probability (currently unused)
    """
    def __init__(self, n_feat, p_drop=0.1):
        super(MaskedTokenNetwork, self).__init__()
        # Project to 21 classes: 20 amino acids + 1 unknown/gap
        self.proj = nn.Linear(n_feat, 21)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        """Initialize projection layer with zeros."""
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        Predict amino acid type from sequence features.

        Args:
            x (torch.Tensor): Sequence features, shape (B, N, L, C)
                             B: batch size, N: number of sequences (MSA depth)
                             L: sequence length, C: feature dimension

        Returns:
            torch.Tensor: Logits for 21 amino acid classes, shape (B, 21, N*L)
        """
        B, N, L = x.shape[:3]
        logits = self.proj(x).permute(0,3,1,2).reshape(B, -1, N*L)

        return logits


class LDDTNetwork(nn.Module):
    """
    Predict Local Distance Difference Test (lDDT) scores.

    lDDT is a measure of local structure quality that compares distances
    in a predicted structure to distances in a reference structure.
    Higher scores indicate better local geometry. Scores are discretized
    into bins for classification.

    Args:
        n_feat (int): Number of input features from per-residue representation
        n_bin_lddt (int): Number of bins for discretizing lDDT scores (default: 50)
    """
    def __init__(self, n_feat, n_bin_lddt=50):
        super(LDDTNetwork, self).__init__()
        self.proj = nn.Linear(n_feat, n_bin_lddt)

        self.reset_parameter()

    def reset_parameter(self):
        """Initialize projection layer with zeros."""
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        Predict lDDT score distribution for each residue.

        Args:
            x (torch.Tensor): Per-residue features, shape (B, L, C)

        Returns:
            torch.Tensor: lDDT logits, shape (B, n_bin_lddt, L)
        """
        logits = self.proj(x)  # (B, L, 50)

        return logits.permute(0,2,1)


class ExpResolvedNetwork(nn.Module):
    """
    Predict which residues are experimentally resolved.

    In experimental structures (e.g., X-ray crystallography), some regions
    may be missing or poorly resolved. This network predicts a binary label
    for each residue indicating whether it is likely to be well-resolved
    in the experimental structure.

    Args:
        d_msa (int): Dimension of MSA (sequence) features
        d_state (int): Dimension of structure state features
        p_drop (float): Dropout probability (currently unused)
    """
    def __init__(self, d_msa, d_state, p_drop=0.1):
        super(ExpResolvedNetwork, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_state = nn.LayerNorm(d_state)
        # Project combined features to binary prediction
        self.proj = nn.Linear(d_msa+d_state, 1)

        self.reset_parameter()

    def reset_parameter(self):
        """Initialize projection layer with zeros."""
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, seq, state):
        """
        Predict experimental resolution status from sequence and structure features.

        Args:
            seq (torch.Tensor): MSA features, shape (B, L, d_msa)
            state (torch.Tensor): Structure state features, shape (B, L, d_state)

        Returns:
            torch.Tensor: Resolution logits, shape (B, L)
                         Higher values indicate higher confidence in resolution
        """
        B, L = seq.shape[:2]

        # Normalize both feature types
        seq = self.norm_msa(seq)
        state = self.norm_state(state)

        # Concatenate and project to single logit per residue
        feat = torch.cat((seq, state), dim=-1)
        logits = self.proj(feat)
        return logits.reshape(B, L)



