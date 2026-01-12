# RFantibody Inference Logic and Function Call Order

## Overview

This document provides a detailed walkthrough of the function call sequences during RFantibody inference, covering all three pipeline stages:

1. **Stage 1**: RFdiffusion backbone generation
2. **Stage 2**: ProteinMPNN sequence design
3. **Stage 3**: RoseTTAFold2 structure validation

---

## Table of Contents

- [Stage 1: RFdiffusion Inference](#stage-1-rfdiffusion-inference)
- [Stage 2: ProteinMPNN Sequence Design](#stage-2-proteinmpnn-sequence-design)
- [Stage 3: RoseTTAFold2 Validation](#stage-3-rosettafold2-validation)
- [Complete Pipeline Example](#complete-pipeline-example)

---

## Stage 1: RFdiffusion Inference

**Entry Point**: `scripts/rfdiffusion_inference.py`

### High-Level Flow

```
main()
├─> AbSampler.__init__()
├─> for i_des in range(num_designs):
│   ├─> sample_init()
│   ├─> for t in range(T, 0, -1):
│   │   └─> sample_step(t)
│   └─> Write PDB output
└─> Done
```

### Detailed Function Call Sequence

#### 1. Initialization Phase

```python
# scripts/rfdiffusion_inference.py::main()

@hydra.main(config_path='config/inference', config_name='base')
def main(conf: HydraConfig):
    """
    Entry point for RFdiffusion inference.
    """

    # 1.1 Create sampler instance
    sampler = model_runners.AbSampler(conf)

    # This triggers:
    # └─> AbSampler.__init__()
    #     └─> Sampler.initialize()
    #         ├─> load_checkpoint()
    #         │   └─> torch.load(ckpt_path)
    #         │
    #         ├─> assemble_config_from_chk()
    #         │   └─> Merge checkpoint config with inference config
    #         │
    #         ├─> load_model()
    #         │   └─> RoseTTAFoldModule.__init__()
    #         │       ├─> Create MSA embeddings
    #         │       ├─> Create template embeddings
    #         │       ├─> Create IterativeSimulator
    #         │       │   ├─> Extra blocks (4)
    #         │       │   ├─> Main blocks (36)
    #         │       │   └─> Ref blocks (4)
    #         │       └─> Create auxiliary predictors
    #         │
    #         └─> Initialize helper objects
    #             ├─> Diffuser.__init__()
    #             │   ├─> EuclideanDiffuser.__init__()
    #             │   │   └─> get_beta_schedule()
    #             │   ├─> IGSO3.__init__()
    #             │   │   └─> _calc_igso3_vals()
    #             │   │       └─> igso3.calculate_igso3() or load from cache
    #             │   └─> INTERP.__init__()
    #             └─> ComputeAllAtomCoords.to(device)
```

#### 2. Design Loop - Initialization

```python
    # 1.2 For each design to generate
    for i_des in range(num_designs):

        # 2.1 Initialize antibody complex
        x_init, seq_init = sampler.sample_init()

        # This calls:
        # └─> AbSampler.sample_init()
        #
        #     ├─> 1) Parse input PDB
        #     │   └─> ab_pose.AbPose()
        #     │       ├─> from_HLT(input_pdb) OR
        #     │       ├─> framework_from_HLT(framework_pdb)
        #     │       └─> target_from_HLT(target_pdb)
        #     │           └─> parsePDB()
        #     │               ├─> Read PDB ATOM lines
        #     │               ├─> Parse CDR REMARK lines (H1, H2, H3, L1, L2, L3)
        #     │               ├─> Identify H, L, T chains
        #     │               └─> Extract xyz, seq, atom_mask
        #     │
        #     ├─> 2) Adjust CDR loop lengths (if not partial diffusion)
        #     │   └─> pose.adjust_loop_lengths(design_loops)
        #     │       └─> For each loop (H1, H2, H3, L1, L2, L3):
        #     │           ├─> Sample target length from config range
        #     │           ├─> If too short: insert residues
        #     │           └─> If too long: delete residues
        #     │
        #     ├─> 3) Create design masks
        #     │   ├─> loop_mask = pose.parse_design_mask()
        #     │   │   └─> Boolean: True for CDR positions
        #     │   ├─> target_mask = Boolean: True for target chain
        #     │   └─> diffusion_mask = Boolean: True for fixed positions
        #     │
        #     ├─> 4) Parse hotspots
        #     │   └─> pose.parse_hotspots(hotspot_res)
        #     │       └─> Convert residue IDs to indices
        #     │
        #     ├─> 5) Setup potential manager
        #     │   └─> PotentialManager.__init__()
        #     │       └─> Initialize guiding potentials (e.g., hotspot)
        #     │
        #     ├─> 6) Forward diffusion
        #     │   └─> diffuser.diffuse_pose()
        #     │       │
        #     │       ├─> 6.1) Center and scale coordinates
        #     │       │   └─> xyz -= mean(CA)
        #     │       │   └─> xyz *= crd_scale (0.25)
        #     │       │
        #     │       ├─> 6.2) Diffuse translations
        #     │       │   └─> eucl_diffuser.diffuse_translations()
        #     │       │       └─> for t in 1..T:
        #     │       │           └─> apply_kernel(xyz, t)
        #     │       │               ├─> mean = sqrt(1-β_t) * ca_xyz
        #     │       │               ├─> var = β_t * var_scale
        #     │       │               └─> sampled = normal(mean, var)
        #     │       │
        #     │       ├─> 6.3) Diffuse rotations
        #     │       │   └─> so3_diffuser.diffuse_frames()
        #     │       │       └─> rigid_from_3_points(N, CA, C)
        #     │       │       └─> sample_vec(t_list, n_samples=L)
        #     │       │           ├─> Sample angles from IGSO(3)
        #     │       │           └─> Random unit vectors
        #     │       │       └─> Apply rotations to frames
        #     │       │           └─> R_perturbed = R_sampled @ R_true
        #     │       │
        #     │       ├─> 6.4) Diffuse torsions (if enabled)
        #     │       │   └─> torsion_diffuser.diffuse_torsions()
        #     │       │       └─> get_torsions(xyz, seq)
        #     │       │       └─> Sample random chi angles
        #     │       │       └─> Interpolate true → random
        #     │       │
        #     │       └─> 6.5) Combine all components
        #     │           └─> diffused_BB = rotated + translated
        #     │           └─> Optional: add diffused sidechains
        #     │           └─> Keep motif sidechains fixed
        #     │
        #     └─> 7) Mask CDR sequences
        #         └─> seq_T = one_hot(seq_true)
        #         └─> seq_T[loop_mask, :20] = 0
        #         └─> seq_T[loop_mask, 21] = 1  # Mask token
        #
        # Returns: (xT, seq_T)
```

#### 3. Reverse Diffusion Loop

```python
        # 2.2 Initialize state variables
        x_t = x_init.clone()
        seq_t = seq_init.clone()

        # 2.3 Reverse diffusion from T → 1
        for t in range(T, final_step-1, -1):

            px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
                t=t,
                seq_t=seq_t,
                x_t=x_t,
                seq_init=seq_init,
                final_step=final_step
            )

            # This calls:
            # └─> AbSampler.sample_step()
            #
            #     ├─> 3.1) Preprocess inputs
            #     │   └─> _preprocess(seq_t, x_t, t)
            #     │       │
            #     │       ├─> Generate time-dependent features
            #     │       │   └─> featurize(ab_item, seq, xyz, t)
            #     │       │       ├─> Create t1d features
            #     │       │       │   ├─> Sequence one-hot (22 dims)
            #     │       │       │   ├─> Timestep: (1 - t/T) for loops
            #     │       │       │   └─> Timestep: 1.0 for framework
            #     │       │       │
            #     │       │       └─> Create t2d features
            #     │       │           └─> xyz_to_t2d(xyz)
            #     │       │               ├─> CA-CA distances → RBF
            #     │       │               ├─> Orientations → sin/cos
            #     │       │               └─> Block adjacency
            #     │       │
            #     │       ├─> Generate time-invariant features
            #     │       │   ├─> MSA: one_hot(seq) + positional encoding
            #     │       │   ├─> idx_pdb: residue indices with chain jumps
            #     │       │   └─> Add hotspots to t1d
            #     │       │
            #     │       └─> Return all features batched
            #     │
            #     ├─> 3.2) Apply self-conditioning
            #     │   │
            #     │   ├─> Sequence self-conditioning (if t < T)
            #     │   │   └─> msa_prev = self.msa_prev from previous step
            #     │   │
            #     │   └─> Structure self-conditioning (if t < T)
            #     │       ├─> process_selfcond(prev_pred, t2d, xyz_t)
            #     │       │   └─> Convert prev_pred to t2d features
            #     │       │   └─> Concatenate with current t2d
            #     │       │
            #     │       └─> correct_selfcond(t2d, ab_item)
            #     │           └─> Fix self-cond for framework/target
            #     │
            #     ├─> 3.3) Model forward pass with recycling
            #     │   └─> for rec in range(num_recycles):
            #     │       │
            #     │       └─> model(msa_masked, msa_full, seq, px0, ...)
            #     │           │
            #     │           ├─> RoseTTAFoldModule.__call__()
            #     │           │   │
            #     │           │   ├─> 1) Embed inputs
            #     │           │   │   ├─> msa_emb = MSA_emb(msa_masked)
            #     │           │   │   ├─> templ_emb = Templ_emb(t1d, t2d)
            #     │           │   │   └─> If recycling:
            #     │           │   │       └─> recycle_emb = Recycling(msa_prev, pair_prev, xyz_prev)
            #     │           │   │
            #     │           │   ├─> 2) IterativeSimulator
            #     │           │   │   │
            #     │           │   │   ├─> Extra blocks (4 iterations)
            #     │           │   │   │   └─> MSA refinement
            #     │           │   │   │
            #     │           │   │   ├─> Main blocks (36 iterations)
            #     │           │   │   │   └─> For each block:
            #     │           │   │   │       ├─> MSA2MSA attention
            #     │           │   │   │       │   └─> RowAttentionWithBias
            #     │           │   │   │       │
            #     │           │   │   │       ├─> MSA2Pair updates
            #     │           │   │   │       │   └─> OuterProduct
            #     │           │   │   │       │
            #     │           │   │   │       ├─> Pair stack
            #     │           │   │   │       │   ├─> TriangleAttention
            #     │           │   │   │       │   ├─> TriangleMultiplication
            #     │           │   │   │       │   └─> Pair2MSA updates
            #     │           │   │   │       │
            #     │           │   │   │       └─> SE(3) structure updates
            #     │           │   │   │           └─> IPA (Invariant Point Attention)
            #     │           │   │   │               ├─> Query/Key/Value projections
            #     │           │   │   │               ├─> Geometric attention weights
            #     │           │   │   │               └─> Update coordinates
            #     │           │   │   │
            #     │           │   │   └─> Ref blocks (4 iterations)
            #     │           │   │       └─> SE3 Transformer refinement
            #     │           │   │
            #     │           │   ├─> 3) Auxiliary predictions
            #     │           │   │   ├─> DistanceNetwork(pair)
            #     │           │   │   │   └─> 37-bin distance distribution
            #     │           │   │   │
            #     │           │   │   ├─> MaskedTokenNetwork(msa)
            #     │           │   │   │   └─> logits (L, 20) AA predictions
            #     │           │   │   │
            #     │           │   │   ├─> LDDTNetwork(state)
            #     │           │   │   │   └─> pred_lddt (L,)
            #     │           │   │   │
            #     │           │   │   ├─> PAENetwork(pair)
            #     │           │   │   │   └─> pred_pae (L, L)
            #     │           │   │   │
            #     │           │   │   └─> BinderNetwork(pair)
            #     │           │   │       └─> p_bind (scalar)
            #     │           │   │
            #     │           │   └─> Return: (msa_prev, pair_prev, px0,
            #     │           │               state_prev, alpha, logits, plddt)
            #     │           │
            #     │           └─> If recycling and not last iteration:
            #     │               └─> Update px0 for next recycle
            #     │                   └─> xyz_t = px0
            #     │                   └─> t2d = xyz_to_t2d(xyz_t)
            #     │
            #     ├─> 3.4) Save self-conditioning for next step
            #     │   ├─> self.prev_pred = px0.clone()
            #     │   └─> self.msa_prev = msa_prev.clone()
            #     │
            #     ├─> 3.5) Convert to all-atom coordinates
            #     │   └─> allatom(seq, px0, alpha)
            #     │       └─> Reconstruct sidechains from backbone + torsions
            #     │
            #     ├─> 3.6) Sample sequence (autoregressive)
            #     │   └─> seq_probs = softmax(logits / temperature)
            #     │   └─> sampled_seq = multinomial(seq_probs)
            #     │   └─> pseq_0 = one_hot(sampled_seq)
            #     │   └─> pseq_0[mask_seq] = seq_init[mask_seq]  # Keep framework
            #     │
            #     └─> 3.7) Denoising step
            #         └─> denoiser.get_next_pose(x_t, px0, t, ...)
            #             │
            #             ├─> Denoise structure
            #             │   └─> SO(3) reverse sampling
            #             │       └─> IGSO3.reverse_sample_vectorized()
            #             │           ├─> Compute R_0t = R_t^T @ R_0
            #             │           ├─> score ≈ rotvec * score_norm / ||rotvec||
            #             │           ├─> drift = g(t)^2 * dt * score
            #             │           ├─> noise = g(t) * sqrt(dt) * z
            #             │           └─> R_{t-1} = exp(drift + noise) @ R_t
            #             │
            #             ├─> Denoise sequence
            #             │   └─> seq_{t-1} = pseq_0
            #             │
            #             ├─> Optional: align motif regions
            #             │   └─> Kabsch alignment of framework
            #             │
            #             └─> Return: (x_{t-1}, seq_{t-1}, tors_{t-1}, px0)
            #
            # Returns: (px0, x_{t-1}, seq_{t-1}, tors_{t-1}, plddt)

        # 2.4 Final coordinates are in x_t (which is now x_0)
        final_xyz = x_t
        final_seq = seq_t
```

#### 4. Output Phase

```python
        # 2.5 Compute hotspot distances (if applicable)
        if has_hotspots:
            Cb = generate_Cbeta(N=final_xyz[:,0], Ca=final_xyz[:,1], C=final_xyz[:,2])
            dist = torch.cdist(Cb[hotspots], Cb[loop_mask])
            mindist = torch.min(dist, dim=1).values
            overallmin = torch.min(mindist)

        # 2.6 Write output PDB
        pdblines = ab_write_pdblines(
            atoms=final_xyz[:,:4],
            seq=final_seq,
            chain_idx=sampler.chain_idx,
            bfacts=bfacts,
            loop_map=sampler.loop_map,
            num2aa=num2aa
        )
        # └─> stamp_pdbline() for each atom
        #     └─> Format ATOM record with chain, residue, coordinates
        #     └─> Add REMARK lines with CDR loop annotations

        with open(output_pdb, 'w') as f:
            f.write('\n'.join(pdblines))

        # 2.7 Write trajectory (if requested)
        if write_trajectory:
            writepdb_multi(traj_file, xyz_stack, ...)
```

---

## Stage 2: ProteinMPNN Sequence Design

**Entry Point**: `scripts/proteinmpnn_interface_design.py`

### Function Call Sequence

```python
# scripts/proteinmpnn_interface_design.py::main()

def main():
    """
    Optimize CDR loop sequences using ProteinMPNN.
    """

    # 1. Load RFdiffusion output
    for pdb_file in input_pdbs:

        # 2. Parse structure
        pose = load_structure(pdb_file)
        # └─> parsePDB()
        #     └─> Read ATOM lines
        #     └─> Parse CDR REMARK lines

        # 3. Setup ProteinMPNN
        runner = ProteinMPNN_runner(config)

        # 4. Design sequences
        designed_pdb = runner.sequence_optimize(pose)
        # └─> ProteinMPNN_runner.sequence_optimize()
        #     │
        #     ├─> 4.1) Prepare features
        #     │   └─> SampleFeatures.from_pose(pose)
        #     │       ├─> Extract backbone coordinates (N, CA, C, O)
        #     │       ├─> Identify designable positions (CDR loops)
        #     │       └─> Create fixed position mask (framework)
        #     │
        #     ├─> 4.2) Featurize structure
        #     │   └─> featurize_protein()
        #     │       ├─> Compute node features
        #     │       │   ├─> Backbone dihedrals (phi, psi, omega)
        #     │       │   └─> Local environments
        #     │       │
        #     │       └─> Compute edge features
        #     │           ├─> CA-CA distances
        #     │           ├─> Orientations (unit vectors)
        #     │           └─> RBF encoding
        #     │
        #     ├─> 4.3) ProteinMPNN forward pass
        #     │   └─> model.forward(features)
        #     │       │
        #     │       ├─> Encode structure
        #     │       │   └─> Graph neural network
        #     │       │       ├─> Node embedding
        #     │       │       └─> Message passing (3 layers)
        #     │       │           ├─> Gather edge messages
        #     │       │           ├─> Update node features
        #     │       │           └─> Normalize
        #     │       │
        #     │       └─> Decode sequence (autoregressive)
        #     │           └─> For each position in random order:
        #     │               ├─> Context: encoded structure + previous AAs
        #     │               ├─> Logits: MLP(context) → (20,)
        #     │               ├─> Probs: softmax(logits / temp)
        #     │               └─> Sample: categorical(probs)
        #     │
        #     ├─> 4.4) Thread sequence onto backbone
        #     │   └─> thread_mpnn_seq(pose, sampled_seq)
        #     │       └─> Update sequence field
        #     │       └─> Keep fixed positions unchanged
        #     │
        #     └─> 4.5) Write output PDB
        #         └─> Write with designed CDR sequences

        # 5. Save designed structure
        save_pdb(designed_pdb, output_file)
```

---

## Stage 3: RoseTTAFold2 Validation

**Entry Point**: `scripts/rf2_predict.py`

### Function Call Sequence

```python
# scripts/rf2_predict.py::main()

def main():
    """
    Validate ProteinMPNN designs with RF2.
    """

    # 1. Initialize predictor
    predictor = AbPredictor(config)
    # └─> AbPredictor.__init__()
    #     └─> Load RF2 model weights
    #         └─> RoseTTAFoldModule.load_state_dict()

    # 2. For each designed structure
    for pdb_file in designed_pdbs:

        # 3. Load structure
        pose = load_pose(pdb_file)
        # └─> parsePDB()
        #     └─> Read coordinates and sequence

        # 4. Predict structure
        predicted_pose = predictor(pose)
        # └─> AbPredictor.__call__()
        #     │
        #     ├─> 4.1) Preprocess inputs
        #     │   └─> pose_to_inference_RFinput(pose)
        #     │       │
        #     │       ├─> Extract coordinates (L, 27, 3)
        #     │       ├─> Encode sequence one-hot (L, 21)
        #     │       │
        #     │       ├─> Create MSA features (1, 1, L, 48)
        #     │       │   └─> Single sequence + positional encoding
        #     │       │
        #     │       ├─> Create template features
        #     │       │   ├─> t1d (1, L, 23)
        #     │       │   │   └─> Sequence + chi angles
        #     │       │   └─> t2d (1, L, L, 44)
        #     │       │       └─> xyz_to_t2d(xyz)
        #     │       │
        #     │       └─> Optional: mask hotspots (10% revealed)
        #     │
        #     ├─> 4.2) Recycling loop
        #     │   └─> for cycle in range(num_recycles):
        #     │       │
        #     │       ├─> RoseTTAFoldModule(inputs, recycle_prev)
        #     │       │   │
        #     │       │   │   [Same as Stage 1, Step 3.3]
        #     │       │   │
        #     │       │   ├─> MSA/Template/Recycle embeddings
        #     │       │   ├─> IterativeSimulator (36 main blocks)
        #     │       │   ├─> Structure prediction
        #     │       │   └─> Auxiliary predictions
        #     │       │
        #     │       ├─> Track metrics for this cycle
        #     │       │   ├─> pLDDT per residue
        #     │       │   ├─> pAE matrix
        #     │       │   └─> Mean pLDDT
        #     │       │
        #     │       └─> Update recycling inputs
        #     │           └─> msa_prev, pair_prev, xyz_prev = outputs
        #     │
        #     ├─> 4.3) Select best cycle
        #     │   └─> best_cycle = argmax(mean_plddt)
        #     │   └─> Extract coordinates and metrics from best cycle
        #     │
        #     ├─> 4.4) Compute confidence scores
        #     │   └─> get_confidence_scores(outputs)
        #     │       │
        #     │       ├─> pLDDT: mean(pred_lddt[CDR_loops])
        #     │       │
        #     │       ├─> pAE: mean(pred_pae[binder, target])
        #     │       │
        #     │       └─> RMSD: rmsd(design_bb, predicted_bb)
        #     │           └─> Kabsch alignment
        #     │           └─> sqrt(mean((x_design - x_pred)^2))
        #     │
        #     └─> 4.5) Update pose with predictions
        #         └─> predicted_pose.xyz = predicted_xyz
        #         └─> predicted_pose.plddt = pred_lddt
        #         └─> predicted_pose.pae = pred_pae

        # 5. Write output with metrics
        write_output(predicted_pose)
        # └─> pose_to_remarked_pdblines(pose)
        #     ├─> Add REMARK lines with scores
        #     │   ├─> REMARK pLDDT: 85.3
        #     │   ├─> REMARK pAE: 3.2
        #     │   └─> REMARK RMSD: 0.8
        #     └─> Write PDB with predicted coordinates
```

---

## Complete Pipeline Example

### End-to-End Workflow

```bash
# Step 1: Generate backbones with RFdiffusion
python scripts/rfdiffusion_inference.py \
    antibody.target_pdb=target.pdb \
    antibody.framework_pdb=framework.pdb \
    antibody.design_loops='{"H3": [10, 15]}' \
    ppi.hotspot_res='[A100, A101, A102]' \
    inference.num_designs=100 \
    inference.output_prefix=outputs/design

# Generates:
# - outputs/design_0.pdb, design_1.pdb, ..., design_99.pdb
# - Each PDB has:
#   * Framework from input
#   * Designed CDR H3 (length 10-15)
#   * Glycine placeholders in CDR loops

# Step 2: Design sequences with ProteinMPNN
python scripts/proteinmpnn_interface_design.py \
    --input_dir outputs/ \
    --output_dir mpnn_designs/ \
    --temperature 0.1 \
    --num_seq_per_target 4

# Generates:
# - mpnn_designs/design_0_seq0.pdb, ..., design_0_seq3.pdb
# - Each PDB has:
#   * Same backbone as input
#   * Designed amino acid sequences in CDR loops
#   * 4 sequence variants per backbone

# Step 3: Validate with RoseTTAFold2
python scripts/rf2_predict.py \
    --input_dir mpnn_designs/ \
    --output_dir validated/ \
    --num_recycles 10

# Generates:
# - validated/design_0_seq0_rf2.pdb, ...
# - Each PDB has:
#   * Predicted structure from RF2
#   * REMARK lines with pLDDT, pAE, RMSD
#   * Can filter by pLDDT > 80, RMSD < 2.0Å

# Step 4: Filter and select top designs
python scripts/filter_designs.py \
    --input_dir validated/ \
    --min_plddt 80 \
    --max_rmsd 2.0 \
    --max_pae 5.0 \
    --output top_designs/

# Final outputs: Top-scoring designs ready for experimental validation
```

### Key Function Call Dependencies

```
Stage 1: RFdiffusion
├─> sample_init()
│   ├─> parsePDB()
│   ├─> adjust_loop_lengths()
│   └─> diffuser.diffuse_pose()
│       ├─> EuclideanDiffuser.diffuse_translations()
│       ├─> IGSO3.diffuse_frames()
│       └─> INTERP.diffuse_torsions()
│
└─> sample_step() [repeated T times]
    ├─> _preprocess()
    │   └─> featurize()
    │       └─> xyz_to_t2d()
    ├─> RoseTTAFoldModule.__call__()
    │   ├─> Embeddings
    │   ├─> IterativeSimulator
    │   └─> AuxiliaryPredictors
    └─> denoiser.get_next_pose()
        └─> IGSO3.reverse_sample_vectorized()

Stage 2: ProteinMPNN
└─> sequence_optimize()
    ├─> featurize_protein()
    ├─> ProteinMPNN.forward()
    │   ├─> encode_structure()
    │   └─> decode_sequence()
    └─> thread_mpnn_seq()

Stage 3: RoseTTAFold2
└─> AbPredictor.__call__()
    ├─> pose_to_inference_RFinput()
    │   └─> xyz_to_t2d()
    ├─> for cycle in range(num_recycles):
    │   └─> RoseTTAFoldModule.__call__()
    ├─> get_confidence_scores()
    │   ├─> compute_plddt()
    │   ├─> compute_pae()
    │   └─> compute_rmsd()
    └─> pose_from_RF_output()
```

---

## Timing and Performance

### Typical Runtime (on single V100 GPU)

```
Stage 1: RFdiffusion
├─> Initialization: ~30 seconds (model loading)
├─> Per design: ~10-20 seconds (T=50 timesteps)
└─> 100 designs: ~20-30 minutes

Stage 2: ProteinMPNN
├─> Per structure: ~0.5-1 second
└─> 400 designs (100 × 4 seqs): ~5-10 minutes

Stage 3: RoseTTAFold2
├─> Per structure: ~5-10 seconds (10 recycles)
└─> 400 designs: ~40-60 minutes

Total pipeline: ~1-1.5 hours for 100 backbones → 400 designs
```

### Optimization Tips

1. **Parallel Processing**: Run multiple RFdiffusion jobs in parallel
2. **Batch Processing**: Process MPNN and RF2 in batches
3. **Early Filtering**: Filter by hotspot distances after RFdiffusion
4. **Reduce Recycles**: Use 3-5 recycles for RF2 during initial screening

---

## Summary

The RFantibody inference pipeline follows a clear three-stage process with well-defined function call orders. Each stage builds on the previous:

1. **RFdiffusion**: Generates diverse backbones via diffusion
2. **ProteinMPNN**: Rapidly designs sequences
3. **RoseTTAFold2**: Validates and filters designs

Understanding the function call flow helps with:
- **Debugging**: Trace errors to specific functions
- **Customization**: Modify specific stages
- **Optimization**: Identify bottlenecks
- **Extension**: Add new features at appropriate points
