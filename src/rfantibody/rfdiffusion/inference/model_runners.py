import os
import torch
import torch.nn.functional as nn

import numpy as np

from omegaconf import DictConfig, OmegaConf
from icecream import ic
import logging
from hydra.core.hydra_config import HydraConfig

import rfantibody.rfdiffusion.util
from rfantibody.rfdiffusion.inference import ab_pose
from rfantibody.rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
from rfantibody.rfdiffusion.kinematics import get_init_xyz, xyz_to_t2d
from rfantibody.rfdiffusion.diffusion import Diffuser
from rfantibody.rfdiffusion.chemical import seq2chars, INIT_CRDS
from rfantibody.rfdiffusion.util_module import ComputeAllAtomCoords
from rfantibody.rfdiffusion.contigs import ContigMap
from rfantibody.rfdiffusion.inference import utils as iu
from rfantibody.rfdiffusion.potentials.manager import PotentialManager
from rfantibody.rfdiffusion.inference import symmetry
from rfantibody.rfdiffusion.util import Dotdict
from rfantibody.rfdiffusion.inference.ab_util import \
    process_init_selfcond, \
    process_selfcond, \
    correct_selfcond, \
    featurize

TOR_INDICES  = rfantibody.rfdiffusion.util.torsion_indices
TOR_CAN_FLIP = rfantibody.rfdiffusion.util.torsion_can_flip
REF_ANGLES   = rfantibody.rfdiffusion.util.reference_angles


class Sampler:
    """
    Base sampler class for RFdiffusion inference.

    This class handles the core diffusion-based protein design process, including:
    - Loading pre-trained RoseTTAFold models
    - Managing diffusion schedules and denoising
    - Running reverse diffusion to generate protein structures

    The sampler coordinates between multiple components:
    - RoseTTAFoldModule: The neural network model
    - Diffuser: Manages forward/reverse diffusion of coordinates and orientations
    - Denoiser: Handles the denoising step at each timestep
    - PotentialManager: Applies guiding potentials during sampling

    Attributes:
        model: The loaded RoseTTAFoldModule neural network
        diffuser: Diffuser object for coordinate/orientation diffusion
        seq_diffuser: Optional sequence diffusion module
        device: torch.device for computation (CPU or CUDA)
        T: Total number of diffusion timesteps
    """

    def __init__(self, conf: DictConfig):
        """Initialize sampler.
        Args:
            conf: Hydra configuration object containing all inference parameters.
        """
        self.initialized = False
        self.initialize(conf)
    
    def ab_design(self):
        return False

    def initialize(self, conf: DictConfig):
        self._log = logging.getLogger(__name__)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        needs_model_reload = not self.initialized or conf.inference.ckpt_override_path != self._conf.inference.ckpt_override_path

        # Assign config to Sampler
        self._conf = conf

        # Initialize inference only helper objects to Sampler
        # JW now added automatic model selection.
        if conf.inference.ckpt_override_path is not None:
            self.ckpt_path = conf.inference.ckpt_override_path
            print("WARNING: You're overriding the checkpoint path from the defaults. Check that the model you're providing can run with the inputs you're providing.")
        else:
            if conf.contigmap.inpaint_seq is not None or conf.contigmap.provide_seq is not None:
                # use model trained for inpaint_seq
                if conf.contigmap.provide_seq is not None:
                    # this is only used for partial diffusion
                    assert conf.diffuser.partial_T is not None, "The provide_seq input is specifically for partial diffusion"
                if conf.scaffoldguided.scaffoldguided:
                    self.ckpt_path='/net/databases/diffusion/models/seq_alone_models_FoldConditioned_Jan23/BFF_4.pt'
                else:
                    self.ckpt_path = '/net/databases/diffusion/models/seq_alone_models_Dec2022/BFF_6.pt'
            elif conf.ppi.hotspot_res is not None and conf.scaffoldguided.scaffoldguided is False:
                # use complex trained model
                self.ckpt_path = '/net/databases/diffusion/models/hotspot_models/base_complex_finetuned_BFF_9.pt'
            elif conf.scaffoldguided.scaffoldguided is True:
                # use complex and secondary structure-guided model
                self.ckpt_path = '/net/databases/diffusion/models/hotspot_models/base_complex_ss_finetuned_BFF_9.pt' 
            else:
                # use default model
                self.ckpt_path = '/net/databases/diffusion/nate_tmp/new_SelfCond_crdscale0.25/models/BFF_4.pt'
        # for saving in trb file:
        assert self._conf.inference.trb_save_ckpt_path is None, "trb_save_ckpt_path is not the place to specify an input model. Specify in inference.ckpt_override_path"
        self._conf['inference']['trb_save_ckpt_path']=self.ckpt_path

        if needs_model_reload:
            # Load checkpoint, so that we can assemble the config
            self.load_checkpoint()
            self.assemble_config_from_chk()
            # Now actually load the model weights into RF
            self.model = self.load_model()
        else:
            self.assemble_config_from_chk()

        # self.initialize_sampler(conf)
        self.initialized=True

        # Assemble config from the checkpoint
        print(' ')
        print('-'*100)
        print(' ')
        print("WARNING: The following options are not currently implemented at inference. Decide if this matters.")
        print("Delete these in inference/model_runners.py once they are implemented/once you decide they are not required for inference -- JW")
        print(" -predict_previous")
        print(" -prob_self_cond")
        print(" -seqdiff_b0")
        print(" -seqdiff_bT")
        print(" -seqdiff_schedule_type")
        print(" -seqdiff")
        print(" -freeze_track_motif")
        print(" -use_motif_timestep")
        print(" ")
        print("-"*100)
        print(" ")
        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.contig_conf = self._conf.contigmap
        self.denoiser_conf = self._conf.denoiser
        self.ppi_conf = self._conf.ppi
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        self.ab_conf = self._conf.antibody
        self.diffuser = Diffuser(**self._conf.diffuser)
        # TODO: Add symmetrization RMSD check here
        if self._conf.seq_diffuser.seqdiff is None:
            ic('Doing AR Sequence Decoding')
            self.seq_diffuser = None

            assert(self._conf.preprocess.seq_self_cond is False), 'AR decoding does not make sense with sequence self cond'
            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        elif self._conf.seq_diffuser.seqdiff == 'continuous':
            ic('Doing Continuous Bit Diffusion')

            kwargs = {
                     'T': self._conf.diffuser.T,
                     's_b0': self._conf.seq_diffuser.s_b0,
                     's_bT': self._conf.seq_diffuser.s_bT,
                     'schedule_type': self._conf.seq_diffuser.schedule_type,
                     'loss_type': self._conf.seq_diffuser.loss_type
                     }
            self.seq_diffuser = seq_diffusion.ContinuousSeqDiffuser(**kwargs)

            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        else:
            sys.exit(f'Seq Diffuser of type: {self._conf.seq_diffuser.seqdiff} is not known')

        if self.inf_conf.symmetry is not None:
            self.symmetry = symmetry.SymGen(
                self.inf_conf.symmetry,
                self.inf_conf.model_only_neighbors,
                self.inf_conf.recenter,
                self.inf_conf.radius, 
            )
        else:
            self.symmetry = None

        self.allatom = ComputeAllAtomCoords().to(self.device)
        
        if not self.ab_design():
            if self.inf_conf.input_pdb is None:
                # set default pdb
                script_dir=os.path.dirname(os.path.realpath(__file__))
                self.inf_conf.input_pdb=os.path.join(script_dir, '../benchmark/input/1qys.pdb')
            self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)

        self.chain_idx = None

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        
        # Get recycle schedule    
        recycle_schedule = str(self.inf_conf.recycle_schedule) if self.inf_conf.recycle_schedule is not None else None
        self.recycle_schedule = iu.recycle_schedule(self.T, recycle_schedule, self.inf_conf.num_recycles)
        
    @property
    def T(self):
        '''
            Return the maximum number of timesteps
            that this design protocol will perform.

            Output:
                T (int): The maximum number of timesteps to perform
        '''
        return self.diffuser_conf.T

    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f'Reading checkpoint from {self.ckpt_path}')
        print('This is inf_conf.ckpt_path')
        print(self.ckpt_path)
        self.ckpt  = torch.load(
            self.ckpt_path, map_location=self.device)

    def assemble_config_from_chk(self) -> None:
        """
        Function for loading model config from checkpoint directly.

        Takes:
            - config file

        Actions:
            - Replaces all -model and -diffuser items
            - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
        
        This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
        This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.
        """
        # get overrides to re-apply after building the config from the checkpoint
        overrides = []
        if HydraConfig.initialized():
            overrides = list(HydraConfig.get().overrides.task)
            ic(overrides)
        # Added to set default T to 50
        overrides.append(f'diffuser.T={self._conf.diffuser.T}') if not any(i.startswith('diffuser.T=') for i in overrides) else None 

        # Add in the preprocess.use_selfcond_emb flag if it is set as the T_scheme
        if not any(i.startswith('preprocess.use_selfcond_emb=') for i in overrides):
            if self.ckpt['config_dict']['antibody']['T_scheme'] == 'selfcond_emb':
                overrides.append('preprocess.use_selfcond_emb=True')

        if 'config_dict' in self.ckpt.keys():
            print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

            # First, check all flags in the checkpoint config dict are in the config file
            for cat in ['model','diffuser','seq_diffuser','preprocess','antibody']:
                #assert all([i in self._conf[cat].keys() for i in self.ckpt['config_dict'][cat].keys()]), f"There are keys in the checkpoint config_dict {cat} params not in the config file"
                for key in self._conf[cat]:
                    if key == 'chi_type' and self.ckpt['config_dict'][cat][key] == 'circular':
                        ic('---------------------------------------------SKIPPPING CIRCULAR CHI TYPE')
                        continue
                    try:
                        print(f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {self.ckpt['config_dict'][cat][key]}")
                        self._conf[cat][key] = self.ckpt['config_dict'][cat][key]
                    except:
                        print(f'WARNING: config {cat}.{key} is not saved in the checkpoint. Check that conf.{cat}.{key} = {self._conf[cat][key]} is correct')
            # add back in overrides again
            for override in overrides:
                if override.split(".")[0] in ['model','diffuser','seq_diffuser','preprocess']:
                    print(f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?') 
                    mytype = type(self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
                    self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])
        else:
            print('WARNING: Model, Diffuser and Preprocess parameters are not saved in this checkpoint. Check carefully that the values specified in the config are correct for this checkpoint')     

    def load_model(self):
        """Create RosettaFold model from preloaded checkpoint."""
        
        # Now read input dimensions from checkpoint.
        self.d_t1d        = self._conf.preprocess.d_t1d
        self.d_t2d        = self._conf.preprocess.d_t2d
        self.use_selfcond_emb = self._conf.preprocess.use_selfcond_emb
        model             = RoseTTAFoldModule(**self._conf.model, d_t1d=self.d_t1d, d_t2d=self.d_t2d, use_selfcond_emb=self.use_selfcond_emb, T=self._conf.diffuser.T).to(self.device)
        model = model.eval()
        self._log.info(f'Loading checkpoint.')
        if self._conf.inference.final_state:
            model.load_state_dict(self.ckpt['final_state_dict'],strict=True)
        else:
            model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        return model

    def construct_contig(self, target_feats):
        self._log.info(f'Using contig: {self.contig_conf.contigs}')
        return ContigMap(target_feats, **self.contig_conf)

    def construct_denoiser(self, L, visible):
        """Make length-specific denoiser."""
        # TODO: Denoiser seems redundant. Combine with diffuser.
        denoise_kwargs = OmegaConf.to_container(self.diffuser_conf)
        denoise_kwargs.update(OmegaConf.to_container(self.denoiser_conf))
        aa_decode_steps = min(denoise_kwargs['aa_decode_steps'], denoise_kwargs['partial_T'] or 999)
        denoise_kwargs.update({
            'L': L,
            'diffuser': self.diffuser,
            'seq_diffuser': self.seq_diffuser,
            'potential_manager': self.potential_manager,
            'visible': visible,
            'aa_decode_steps': aa_decode_steps,
        })
        return iu.Denoise(**denoise_kwargs)

    def sample_init(self, return_forward_trajectory=False):
        """Initialize the starting state for reverse diffusion sampling.

        This method sets up the initial coordinates and sequence at timestep T (fully noised).
        Subclasses implement specific initialization logic:
        - AbSampler: Initialize antibody framework + random CDR loops
        - Standard Sampler: Initialize from contig map

        The initialization involves:
        1. Loading/constructing the input structure
        2. Running forward diffusion to timestep T
        3. Masking designed regions in the sequence

        Args:
            return_forward_trajectory: If True, return the full forward diffusion trajectory

        Returns:
            xt: (L, 14, 3) Starting backbone coordinates at timestep T (fully noised)
            seq_t: (L, 22) Starting sequence one-hot, with designed regions masked (token 21)
        """

        raise NotImplementedError('This function should be implemented in a subclass')

    def _preprocess(self, seq, xyz_t, t, repack=False):
        """Prepare all input features for the RoseTTAFold model at timestep t.

        This method converts the current structure and sequence state into the
        feature tensors required by the RoseTTAFoldModule network. Features are
        divided into time-dependent and time-invariant components.

        Input shapes:
            seq: (L, 22) one-hot encoded sequence with mask token
            xyz_t: (L, 14, 3) current backbone coordinates (diffused)
            t: int, current timestep (1 to T)

        Output feature tensors:
            msa_masked: (1, 1, L, 48) MSA features with positional encoding
            msa_full: (1, 1, L, 25) Full MSA track (single sequence)
            seq: (1, L, 22) Sequence one-hot for network input
            xyz_t: (1, L, 14, 3) Template coordinates
            idx_pdb: (1, L) Residue indices with chain breaks
            t1d: (1, L, 23+) 1D features per residue:
                - Sequence one-hot (22 dims: 20 AAs + gap + mask)
                - Global timestep: (1-t/T) for designed regions, 1 for fixed (1 dim)
                - Hotspot indicator (1 dim)
                - Optional: SS prediction, chi angle timestep
            t2d: (1, L, L, 44+) Pairwise 2D features:
                - Distance-based RBF features
                - Orientation features (sin/cos of angles)
                - Self-conditioning structure (from previous step prediction)
                - Block adjacency matrix (last channel)

        Args:
            seq: (L, 22) Current sequence state
            xyz_t: (L, 14, 3) Current structure state
            t: Current timestep
            repack: Whether this is a repack step

        Returns:
            Tuple of feature tensors ready for model forward pass
        """
        raise NotImplementedError('This function should be implemented in a subclass')

        
    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step, return_extra=False):
        '''Execute one reverse diffusion step from timestep t to t-1.

        This is the core sampling loop that:
        1. Preprocesses inputs into model features
        2. Runs the RoseTTAFold model forward pass
        3. Predicts the denoised structure (px0)
        4. Samples the sequence via autoregressive decoding
        5. Updates coordinates for the next timestep using the denoiser

        The method implements:
        - Self-conditioning: Uses previous step's prediction to inform current step
        - Sequence masking: Only designs specified regions
        - Structure alignment: Keeps motif regions fixed

        Args:
            t (int): Current timestep (counting down from T to 1)
            seq_t (torch.tensor): (L, 22) Sequence state at timestep t
            x_t (torch.tensor): (L, 14, 3) Backbone coordinates at timestep t
            seq_init (torch.tensor): (L, 22) Initial sequence (motif regions fixed)
            final_step (int): When to stop diffusion (usually 1)
            return_extra (bool): Whether to return additional debug info

        Returns:
            px0: (L, 14, 3) Model's prediction of the final denoised structure
            x_t_1: (L, 14, 3) Updated coordinates for timestep t-1
            seq_t_1: (L, 22) Updated sequence for timestep t-1
            tors_t_1: (L, 10, 2) Updated torsion angles for timestep t-1
            plddt: (L,) Per-residue predicted lDDT confidence scores
        '''

        raise NotImplementedError('This function should be implemented in a subclass')

class AbSampler(Sampler):
    '''Antibody-specific sampler for designing CDR loops with RFdiffusion.

    This class extends the base Sampler to handle antibody-target complex design:
    - Parses HLT format PDBs (Heavy, Light, Target chains)
    - Identifies and designs specific CDR loops (H1, H2, H3, L1, L2, L3)
    - Supports hotspot-guided design targeting specific epitope residues
    - Implements antibody-specific self-conditioning schemes

    Key Features:
    1. CDR Loop Design: Flexible loop length sampling and design
    2. Framework Conservation: Maintains constant antibody framework
    3. Hotspot Targeting: Guides CDR loops toward specified target residues
    4. Partial Diffusion: Can refine existing CDR designs

    Attributes:
        pose (AbPose): Antibody-target complex structure handler
        ab_item (Dotdict): Contains antibody-specific metadata:
            - loop_mask: Boolean mask of CDR positions being designed
            - target_mask: Boolean mask of target chain positions
            - hotspots: Target residue positions to target
            - interchain_mask: Mask for interface residues
        loop_map (dict): Maps CDR names (H1, H2, etc.) to residue indices
        binderlen (int): Length of antibody binder region (H+L chains)
    '''

    def ab_design(self):
        """Flag indicating this is an antibody design sampler."""
        return True

    def sample_init(self):
        '''Initialize antibody-target complex for diffusion sampling.

        This method sets up the antibody design problem by:
        1. Parsing the input PDB structure(s) in HLT format
        2. Adjusting CDR loop lengths if specified
        3. Identifying which residues to design
        4. Running forward diffusion to timestep T
        5. Masking CDR loop sequences

        The method handles two input modes:
        - Single HLT PDB: Complete antibody-target complex
        - Separate framework + target PDBs: Assembled during initialization

        Process Flow:
        1. Parse input structure(s) into AbPose object
        2. Adjust CDR loop lengths (insert/delete residues)
        3. Create design masks (which loops to design)
        4. Parse target hotspot residues
        5. Run forward diffusion on binder region
        6. Mask CDR loop sequences (set to unknown token 21)

        Returns:
            xT: (L, 14, 3) Fully diffused backbone coordinates at timestep T
            seq_T: (L, 22) Sequence one-hot with CDR loops masked
        '''

        #### 1) Parse pdb to an ab_pose that can be easily manipulated
        ####################################################################
        self.pose = ab_pose.AbPose()

        # Determine which format the input structure has been provided
        if self.inf_conf.input_pdb is not None:
            assert(self.ab_conf.target_pdb is None and self.ab_conf.framework_pdb is None), \
                    "Both inference.input_pdb and antibody.target + antibody.framework_pdb cannot be active at the same time."

            self.pose.from_HLT(self.inf_conf.input_pdb)

        assert(~((self.ab_conf.target_pdb is None) ^ (self.ab_conf.framework_pdb is None))), \
                "Having antibody.target and not antibody.framework_pdb or vice versa is not currently supported."

        if self.ab_conf.target_pdb is not None and self.ab_conf.framework_pdb is not None:
            assert(self.diffuser_conf.partial_T is None), \
                    "Partial diffusion is only supported when using inference.input_pdb"

            assert(self.inf_conf.input_pdb is None), \
                    "Both inference.input_pdb and antibody.target + antibody.framework_pdb cannot be active at the same time."

            self.pose.framework_from_HLT(self.ab_conf.framework_pdb)
            self.pose.target_from_HLT(self.ab_conf.target_pdb)


        #### 2) Adjust the length of the CDR loops in the AbPose
        ####################################################################################################

        # If we are doing partial diffusion, we will skip the loop length adjustment step
        # Since we are just resampling from the starting scaffold
        if self.diffuser_conf.partial_T:
           print("Partial diffusion detected, skipping loop length adjustment step") 

        else:
            # We are doing full diffusion, so we need to adjust the loop lengths
            ic(self.pose.length())
            ic(self.pose.binder_len())
            ic(self.pose.L.seq)
            self.pose.adjust_loop_lengths(self.ab_conf.design_loops)
            ic(self.pose.length())
            ic(self.pose.binder_len())
            ic(self.pose.L.seq)


        #### 3) Assemble the ab_item for use downstream. Also determine which residues we should design
        ####################################################################################################
        self.ab_item = Dotdict()
        self.ab_item.loop_mask = self.pose.parse_design_mask(self.ab_conf.design_loops)

        assert(torch.any(self.ab_item.loop_mask)), "Received input with no diffused region. Exiting"

        self.L = self.pose.length()
        self.diffusion_mask = torch.ones(self.L).bool() # True is not diffused
        if self.ab_conf.T_scheme == 'fixed_dock':
            # Only diffuse loops in fixed dock case
            self.diffusion_mask[self.ab_item.loop_mask] = False 

        elif self.ab_conf.T_scheme in {'single_T', 'single_T_better_confidence', 'no_T', 'single_T_correct_selfcond', 'noSeq_single_T_correct_selfcond', 'selfcond_emb'}:
            # Diffuse binder region
            self.diffusion_mask[:self.pose.binder_len()] = False 

        else:
            raise NotImplementedError()

        # Needed for downstream processing
        self.ab_item.target = True

        self.ab_item.target_mask = torch.zeros(self.L).bool()
        self.ab_item.target_mask[self.pose.binder_len():] = True

        self.ab_item.interchain_mask = self.pose.get_interchain_mask()

        self.binderlen = self.pose.binder_len()

        #### 4) Parse hotspots
        ##########################################
        self.ab_item.hotspots = self.pose.parse_hotspots(self.ppi_conf.hotspot_res)

        self.ab_item.inputs   = self.pose.to_diffusion_inputs()

        # Assign chain_idx, this is used by run_inference.py to assign chain letters upon writing to disk
        self.chain_idx = self.pose.get_chain_idx()
        self.loop_map  = self.pose.get_loop_map()

        #### 5) Setup Potential Manager, there are not yet any Ab potentials
        ########################################################################
        self.potential_manager = PotentialManager(self.potential_conf,
                                                  self.ppi_conf,
                                                  self.diffuser_conf,
                                                  self.inf_conf,
                                                  self.ab_item.hotspots,
                                                  self.pose.binder_len())

        # These are necessary for compatibility with the parent sample_step function
        self.mask_seq = torch.clone(~self.ab_item.loop_mask)
        #self.mask_seq = torch.clone(self.diffusion_mask)
        self.mask_str = torch.clone(self.diffusion_mask)

        # Determine the timesteps to use for diffusion
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)

        t_list = np.arange(1, self.t_step_input+1)

        #### 6) Diffuse the CDR loop regions to timestep T
        #############################################
        # Run forward diffusion on the binder (antibody) region
        # - Framework and target remain fixed (diffusion_mask=True)
        # - CDR loops are progressively noised (diffusion_mask=False)
        # Returns coordinates at all timesteps 1..T
        fa_stack, _, _ = self.diffuser.diffuse_pose(
            self.ab_item.inputs.xyz_true,
            self.ab_item.inputs.seq_true,
            self.ab_item.inputs.atom_mask,
            diffusion_mask=self.diffusion_mask.squeeze(),
            t_list=t_list,
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)

        # Extract backbone coordinates at final timestep T (fully noised)
        xT = torch.clone(fa_stack[-1].squeeze()[:,:14])

        #### 7) Mask the input sequence of the CDR loops
        ####################################################
        # Convert sequence to one-hot encoding
        seq_T = nn.one_hot(self.ab_item.inputs.seq_true, num_classes=22).float()
        # Zero out CDR loop sequences and set to mask token (21)
        seq_T[~self.mask_seq,:20] = 0  # Zero out all amino acid channels
        seq_T[~self.mask_seq,21]  = 1  # Set mask token channel to 1

        self.denoiser = self.construct_denoiser(self.L, visible=self.diffusion_mask)

        ic(self.ab_item.loop_mask)
        ic(self.ab_item.hotspots)
        return xT, seq_T

    def _preprocess(self, seq, xyz_t, t):
        '''
        xyz_t [T,L,27,3]
        msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t

        '''

        L = xyz_t.shape[0]

        ## 1) Generate the time-dependent features
        ################################################

        tmp_xyz = torch.full((L,27,3), np.nan)
        tmp_xyz[:,:14] = xyz_t # [L,27,3]

        # Featurize expects xyz to have 27 atom dimensions
        features = featurize(
                             self.ab_item,
                             seq,
                             tmp_xyz,
                             self.preprocess_conf.d_t1d,
                             self.preprocess_conf.hotspot_dim,
                             self.ab_conf.T_scheme,
                             1 - (t / self.T),
                             ~self.preprocess_conf.motif_sidechain_input,
                             ~self.ab_conf.no_bugfix_t1d_mask
                            )

        ## 2) Now generate the time-invariant features
        ################################################
        
        ## idx_pdb ##
        #############

        idx_pdb = torch.arange(L) # (L)
        if self.ab_item.target:
            idx_pdb[self.ab_item.target_mask] += 200 # Do idx jump at chainbreak
        
        ## Add hotspots to t1d ##
        #########################
        features['t1d'][...,22] = self.ab_item.hotspots[None,None]
        
        retval = (
                  features['msa_masked'],
                  features['msa_full'],
                  features['seq'],
                  features['xyz_prev'].unsqueeze(0),
                  idx_pdb.unsqueeze(0),
                  features['t1d'].unsqueeze(0),
                  features['t2d'].unsqueeze(0),
                  features['xyz_t'].unsqueeze(0),
                  features['alpha_t'].unsqueeze(0)
                 )

        # Send all inputs to device
        retval = [i.to(self.device) for i in retval]

        return retval

    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step):

        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
            seq_t, x_t, t)

        B,N,L = xyz_t.shape[:3]

        ##################################
        ######## Seq Self Cond ###########
        ##################################
        if (t < self.diffuser.T) and (t != self.diffuser_conf.partial_T) \
            and self.preprocess_conf.selfcondition_msaprev and self.preprocess_conf.msaprev_bugfix:

            ic('Providing Ab Seq Self Cond')
            msa_prev = self.msa_prev

        else:
            msa_prev = None 

        ##################################
        ######## Str Self Cond ###########
        ##################################
        if (t < self.diffuser.T) and (t != self.diffuser_conf.partial_T):

            ic('Providing Ab Str Self Cond')
            xyz_t, t2d, xyz_sc, sc2d = process_selfcond(self.prev_pred, t2d, xyz_t, self.ab_conf, xyz_t.device)

            # Correct selfcond now will detect from the checkpoint file whether it
            # should be run or not
            t2d = correct_selfcond(
                                    t2d,
                                    self.ab_conf,
                                    self.ab_item.inputs.xyz_true,
                                    self.ab_item.loop_mask,
                                    self.ab_item.target_mask,
                                    self.ab_item.interchain_mask
                                  )

        else:
            # The non-selfcond step for antibodies is to just leave the input as-is
            sc2d, xyz_sc = process_init_selfcond(t2d, xyz_t, self.ab_conf, xyz_t.device)
        
        print('Monitoring target centering')
        ic(xyz_t[0,0,self.diffusion_mask,1].mean(dim=0))
        ic(xt_in[0,self.diffusion_mask,1].mean(dim=0))

        with torch.no_grad():
            px0=xt_in
            for rec in range(self.recycle_schedule[t-1]):
                msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(msa_masked,
                                    msa_full,
                                    seq_in,
                                    px0,
                                    idx_pdb,
                                    t1d=t1d,
                                    t2d=t2d,
                                    sc2d=sc2d,
                                    xyz_sc=xyz_sc,
                                    xyz_t=xyz_t,
                                    alpha_t=alpha_t,
                                    msa_prev = msa_prev,
                                    pair_prev = None,
                                    state_prev = None,
                                    t=torch.tensor(t),
                                    return_infer=True,
                                    motif_mask=self.diffusion_mask.squeeze().to(self.device))   

                # To permit 'recycling' within a timestep, in a manner akin to how this model was trained
                # Aim is to basically just replace the xyz_t with the model's last px0, and to *not* recycle the state, pair or msa embeddings
                if rec < self.recycle_schedule[t-1] -1:
                    zeros = torch.zeros(B,1,L,24,3).float().to(xyz_t.device)
                    xyz_t = torch.cat((px0.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]
                    t2d   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]
                    px0=xt_in

        self.prev_pred = torch.clone(px0)
        self.msa_prev  = torch.clone(msa_prev)

        # prediction of X0
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0     = px0.squeeze()[:,:14]

        # Default method of decoding sequence
        seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
        sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position

        pseq_0 = torch.nn.functional.one_hot(
            sampled_seq, num_classes=22).to(self.device).float()

        pseq_0[self.mask_seq.squeeze()] = seq_init[self.mask_seq.squeeze()].to(self.device) # [L,22]
        
        if t > final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                seq_diffusion_mask=self.mask_seq.squeeze(),
                seq_t=seq_t,
                pseq0=pseq_0,
                diffuse_sidechains=self.preprocess_conf.sidechain_input,
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input
            )
            self._log.info(
                    f'Timestep {t}, input to next step: { seq2chars(torch.argmax(seq_t_1, dim=-1).tolist())}')
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            seq_t_1 = pseq_0

            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.mask_str.shape[-1], 10, 2))
            px0 = px0.to(x_t.device)

        return px0, x_t_1, seq_t_1, tors_t_1, plddt



