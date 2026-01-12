# RFantibody Core Functions Documentation

## Overview

This document describes the core functions and their logic in the RFantibody pipeline. RFantibody combines three state-of-the-art methods for structure-based antibody design:

1. **RFdiffusion**: Backbone structure generation
2. **ProteinMPNN**: CDR sequence design
3. **RoseTTAFold2**: Structure validation

---

## Table of Contents

- [1. RFdiffusion Core Functions](#1-rfdiffusion-core-functions)
  - [1.1 AbSampler](#11-absampler)
  - [1.2 Diffusion Process](#12-diffusion-process)
- [2. RoseTTAFold2 Core Functions](#2-rosettafold2-core-functions)
  - [2.1 AbPredictor](#21-abpredictor)
  - [2.2 Network Architecture](#22-network-architecture)
- [3. ProteinMPNN Core Functions](#3-proteinmpnn-core-functions)
  - [3.1 Sequence Design](#31-sequence-design)
- [4. Shared Utilities](#4-shared-utilities)

---

## 1. RFdiffusion Core Functions

### 1.1 AbSampler

**Location**: `src/rfantibody/rfdiffusion/inference/model_runners.py`

#### Purpose
The `AbSampler` class is the antibody-specific implementation of RFdiffusion that handles CDR loop design. It extends the base `Sampler` class with antibody-specific logic.

#### Key Methods

##### `sample_init()`

**Purpose**: Initialize the antibody-target complex for diffusion sampling.

**Logic Flow**:
```python
1. Parse Input Structure
   └─> Load PDB in HLT format (Heavy/Light/Target chains)
   └─> Create AbPose object with chain annotations

2. Adjust CDR Loop Lengths (if specified)
   └─> Insert/delete residues in specified loops (H1, H2, H3, L1, L2, L3)
   └─> Maintain framework geometry

3. Create Design Masks
   └─> loop_mask: Boolean mask of CDR positions to design
   └─> target_mask: Boolean mask of target chain
   └─> diffusion_mask: Regions NOT to diffuse (framework + target)

4. Parse Hotspots
   └─> Identify target epitope residues to guide design

5. Forward Diffusion
   └─> Apply noise to CDR loop coordinates (T timesteps)
   └─> Keep framework and target fixed

6. Mask Sequences
   └─> Set CDR loop sequences to unknown token (21)
   └─> Preserve framework sequences

Returns: (xT, seq_T)
   └─> xT: Fully noised coordinates at timestep T
   └─> seq_T: One-hot sequence with CDR loops masked
```

**Key Data Structures**:
- `AbPose`: Antibody-target complex representation with CDR annotations
- `ab_item`: Dictionary containing:
  - `loop_mask`: (L,) Boolean - True for CDR positions
  - `target_mask`: (L,) Boolean - True for target positions
  - `hotspots`: (L,) Boolean - Target hotspot indicators
  - `interchain_mask`: (L, L) Boolean - Inter-chain contacts

##### `_preprocess()`

**Purpose**: Convert current state into model input features.

**Logic Flow**:
```python
1. Generate Time-Dependent Features
   └─> t1d features (per-residue):
       ├─> Sequence one-hot (22 dims)
       ├─> Timestep encoding: (1 - t/T)
       ├─> Hotspot indicator
       └─> Optional: SS prediction, chi timestep

   └─> t2d features (pairwise):
       ├─> Distance-based RBF
       ├─> Orientation (sin/cos of angles)
       ├─> Self-conditioning structure
       └─> Block adjacency

2. Generate Time-Invariant Features
   └─> MSA features (minimal - single sequence)
   └─> Residue indices (with chain break jumps)
   └─> Template coordinates

3. Combine and Return
   └─> All features batched and moved to device
```

**Input**: `(seq_t, xyz_t, t)`
**Output**: `(msa_masked, msa_full, seq, xyz_prev, idx_pdb, t1d, t2d, xyz_t, alpha_t)`

##### `sample_step()`

**Purpose**: Execute one reverse diffusion step from t → t-1.

**Logic Flow**:
```python
1. Preprocess Inputs
   └─> Convert (seq_t, x_t) to model features

2. Apply Self-Conditioning
   └─> Sequence self-cond: Use previous MSA prediction
   └─> Structure self-cond: Use previous coordinate prediction
   └─> Correct self-cond for antibody-specific bugs

3. Model Forward Pass (with recycling)
   for rec in range(num_recycles):
       └─> RoseTTAFoldModule forward
           ├─> MSA/Pair/State embeddings
           ├─> Track module iterations (36 blocks)
           ├─> Structure prediction (px0)
           └─> Auxiliary predictions (plddt, logits)

       └─> Update xyz_t with px0 for next recycle

4. Sequence Sampling
   └─> Apply softmax to logits with temperature
   └─> Multinomial sampling (autoregressive)
   └─> Keep framework sequences fixed

5. Denoising Step
   └─> Denoise coordinates: x_t → x_{t-1}
   └─> Update sequence: seq_t → seq_{t-1}
   └─> Align motif regions

Returns: (px0, x_{t-1}, seq_{t-1}, tors_{t-1}, plddt)
```

**Key Concepts**:
- **Self-conditioning**: Model sees its previous prediction to improve consistency
- **Recycling**: Multiple passes through network at same timestep (like AlphaFold)
- **Hotspot targeting**: Guides CDR loops toward specified epitope residues

---

### 1.2 Diffusion Process

**Location**: `src/rfantibody/rfdiffusion/diffusion.py`

#### Diffuser Class

**Purpose**: Orchestrate forward diffusion of protein structures.

**Components**:
1. **EuclideanDiffuser**: Translation (C-alpha coordinates)
2. **IGSO3/SLERP**: Rotation (backbone orientation frames)
3. **INTERP**: Torsion angles (chi angles)

##### `diffuse_pose()`

**Purpose**: Apply complete forward diffusion to a protein structure.

**Logic Flow**:
```python
1. Preparation
   └─> Center structure at origin
   └─> Scale coordinates by crd_scale (0.25)

2. Diffuse Translations
   └─> Apply Gaussian noise to C-alpha positions
   └─> Use beta schedule (linear or cosine)
   └─> Equation: x_t = sqrt(1-β_t) * x_{t-1} + sqrt(β_t) * ε

3. Diffuse Rotations
   └─> Sample random rotations from IGSO(3)
   └─> Apply to backbone frames (N-CA-C)
   └─> Use spherical linear interpolation (SLERP)

4. Diffuse Torsions (if enabled)
   └─> Sample random chi angles
   └─> Interpolate from true → random

5. Combine All Components
   └─> Translated + rotated backbone
   └─> Optional: Add diffused sidechains
   └─> Keep motif sidechains fixed

Returns: (fa_stack, aa_masks, xyz_true)
   └─> fa_stack: Coordinates at all timesteps
   └─> aa_masks: Amino acid decoding schedule
   └─> xyz_true: Original coordinates
```

#### IGSO3 Class

**Purpose**: Implement SO(3) diffusion for 3D rotations.

**Key Methods**:

##### `diffuse_frames()`

**Logic Flow**:
```python
1. Convert XYZ to Rotation Matrices
   └─> rigid_from_3_points(N, CA, C)
   └─> R_true: Ground truth orientation frames

2. Sample Random Rotations
   └─> Sample rotation vectors from IGSO(3) distribution
   └─> Magnitude determined by variance schedule
   └─> Direction uniformly random on unit sphere

3. Compute Score
   └─> Precomputed score norm from cache
   └─> Score = gradient of log p(R_t)

4. Apply Rotations
   └─> R_perturbed = R_sampled @ R_true
   └─> Transform coordinates by rotation

Returns: (perturbed_crds, R_perturbed)
```

##### `reverse_sample()`

**Purpose**: Denoise rotations during reverse diffusion.

**Logic Flow**:
```python
1. Compute Rotation to Ground Truth
   └─> r_0t = r_t^T @ r_0
   └─> Convert to rotation vector

2. Approximate Score
   └─> score ≈ rotvec * score_norm(t, omega) / omega
   └─> Scaled by predicted distance to ground truth

3. Sample Perturbation
   └─> drift_term = g(t)^2 * step_size * score
   └─> noise_term = g(t) * sqrt(step_size) * z
   └─> perturb = drift_term + noise_level * noise_term

4. Apply Perturbation
   └─> R_{t-1} = exp(perturb) @ R_t
   └─> Convert rotation vector to matrix

Returns: R_{t-1}
```

---

## 2. RoseTTAFold2 Core Functions

### 2.1 AbPredictor

**Location**: `src/rfantibody/rf2/modules/model_runner.py`

#### Purpose
Validate RFdiffusion designs by predicting their structures and computing confidence metrics.

#### Key Methods

##### `__call__()`

**Purpose**: Run structure prediction with recycling.

**Logic Flow**:
```python
1. Preprocess Inputs
   └─> pose_to_inference_RFinput(pose)
       ├─> Extract coordinates
       ├─> Encode sequence
       ├─> Create MSA features (1x1xLx48)
       └─> Generate template features

2. Recycling Loop
   for cycle in range(num_recycles):
       └─> RoseTTAFoldModule forward
           ├─> Input embeddings (MSA, templates, recycling)
           ├─> IterativeSimulator (36 main blocks)
           ├─> Structure prediction (xyz coordinates)
           └─> Auxiliary predictions (pLDDT, pAE, p_bind)

       └─> Update recycling embeddings for next cycle

3. Select Best Cycle
   └─> Choose cycle with best pLDDT
   └─> Extract final coordinates and metrics

4. Compute Confidence Scores
   └─> pLDDT: Per-residue confidence
   └─> pAE: Predicted aligned error
   └─> RMSD: vs. input design (if available)

Returns: Updated pose with predictions
```

**Key Features**:
- **Recycling**: Iteratively refines prediction (default 10 cycles)
- **Confidence Metrics**: pLDDT, pAE for filtering designs
- **RMSD Calculation**: Compares design vs. prediction

##### `get_confidence_scores()`

**Purpose**: Extract confidence metrics from predictions.

**Metrics**:
```python
1. pLDDT (predicted lDDT)
   └─> Per-residue confidence score [0-100]
   └─> Higher = more confident
   └─> Mean over CDR loops used for filtering

2. pAE (predicted aligned error)
   └─> Pairwise distance error matrix [L, L]
   └─> Lower = better
   └─> Mean over binder-target interface

3. p_bind (binding confidence)
   └─> Predicted probability of binding
   └─> Trained on antibody-antigen complexes

4. RMSD
   └─> Root mean square deviation
   └─> Design backbone vs. predicted backbone
   └─> Lower = design is self-consistent
```

---

### 2.2 Network Architecture

**Location**: `src/rfantibody/rf2/network/RoseTTAFoldModel.py`

#### RoseTTAFoldModule

**Purpose**: Core neural network for structure prediction.

**Architecture**:
```
Input Embeddings
├─> MSA Embedding (d_msa=256)
├─> Extra MSA Track
├─> Template Embedding
└─> Recycling Embedding

↓

IterativeSimulator
├─> Extra Blocks (4 iterations)
│   └─> Initial MSA refinement
│
├─> Main Blocks (36 iterations)
│   ├─> MSA-to-MSA Attention
│   ├─> MSA-to-Pair Updates
│   ├─> Pair Stack
│   │   ├─> Triangle Attention
│   │   ├─> Triangle Updates
│   │   └─> Pair-to-MSA Updates
│   └─> SE(3) Updates
│       └─> Structure refinement
│
└─> Ref Blocks (4 iterations)
    └─> Final structure refinement
    └─> SE3 Transformer (equivariant)

↓

Auxiliary Predictors
├─> Distance Network (37-bin distribution)
├─> Masked Token Network (amino acid logits)
├─> LDDT Network (per-residue confidence)
├─> PAE Network (pairwise aligned error)
└─> Binder Network (binding confidence)

↓

Output
├─> xyz: (L, 27, 3) All-atom coordinates
├─> logits: (L, 20) Sequence predictions
├─> pred_lddt: (L,) Confidence scores
└─> pred_pae: (L, L) Aligned error matrix
```

**Key Components**:

1. **MSA Track**: Processes sequence information
2. **Pair Track**: Models residue-residue relationships
3. **SE(3) Transformer**: Geometrically equivariant structure updates
4. **Recycling**: Feeds outputs back as inputs

---

## 3. ProteinMPNN Core Functions

### 3.1 Sequence Design

**Location**: `src/rfantibody/proteinmpnn/util_protein_mpnn.py`

#### `sequence_optimize()`

**Purpose**: Design CDR loop sequences for RFdiffusion backbones.

**Logic Flow**:
```python
1. Parse Structure
   └─> Load PDB with CDR annotations
   └─> Extract backbone coordinates (N, CA, C, O)
   └─> Identify which positions to design

2. Featurize Structure
   └─> Compute node features (residue environments)
   └─> Compute edge features (residue-residue geometry)
   └─> Generate feature dictionary

3. ProteinMPNN Forward Pass
   └─> Encode structure
       ├─> Graph neural network
       ├─> Message passing on residue graph
       └─> Context aggregation

   └─> Decode sequence
       ├─> Autoregressive sampling
       ├─> Temperature-based stochasticity
       └─> Fixed positions remain unchanged

4. Thread Sequence onto Backbone
   └─> Assign sampled amino acids
   └─> Preserve framework sequences
   └─> Output PDB with designed CDRs

Returns: PDB with designed sequences
```

**Key Parameters**:
- **Temperature (τ)**: Controls sequence diversity (default 0.1)
- **Fixed positions**: Framework residues not designed
- **Designed positions**: CDR loops specified in config

---

## 4. Shared Utilities

### 4.1 Coordinate Transformations

**Location**: `src/rfantibody/rfdiffusion/kinematics.py`

#### `xyz_to_t2d()`

**Purpose**: Convert 3D coordinates to 2D pairwise features.

**Logic**:
```python
1. Extract Backbone Atoms
   └─> N, CA, C coordinates

2. Compute Distances
   └─> CA-CA distances
   └─> RBF encoding (Gaussian basis)

3. Compute Orientations
   └─> Dihedral angles
   └─> Unit vectors between residues
   └─> sin/cos encoding

4. Stack Features
   └─> [distance_rbf, orientations, etc.]

Returns: (L, L, d_t2d) pairwise features
```

### 4.2 Pose Management

**Location**: `src/rfantibody/rf2/modules/pose_util.py`

#### Pose Object

**Purpose**: Container for antibody-target complex structure.

**Attributes**:
```python
@dataclass
class Pose:
    xyz: torch.Tensor           # (L, 27, 3) All-atom coordinates
    seq: torch.Tensor           # (L,) Integer sequence
    atom_mask: torch.Tensor     # (L, 27) Atom presence mask
    cdrs: CDR                   # CDR loop annotations
    hotspots: torch.Tensor      # (L,) Hotspot indicators
    chain_dict: OrderedDict     # H/L/T chain masks
```

**Key Methods**:
- `pose_generator()`: Iterate over input PDB files
- `pose_to_remarked_pdblines()`: Add metrics to PDB REMARK lines
- `pose_from_RF_output()`: Convert model predictions to Pose

---

## Summary

The RFantibody pipeline coordinates three core components:

1. **RFdiffusion (AbSampler)**: Generates diverse antibody backbones via diffusion
2. **ProteinMPNN**: Rapidly designs sequences for generated backbones
3. **RoseTTAFold2 (AbPredictor)**: Validates designs with structure prediction

Each component uses sophisticated geometric and neural network operations to achieve state-of-the-art antibody design performance.

**Key Design Patterns**:
- **Diffusion Process**: Forward noise + reverse denoising
- **Self-Conditioning**: Model sees previous predictions
- **Recycling**: Multiple refinement passes
- **Geometric Equivariance**: SE(3)-invariant operations
- **Hotspot Guidance**: Target-aware design
