import multiprocessing
import os
import pickle
import platform
import shutil
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from torchview import draw_graph

from src.boltz.main import (
    download_boltz1,
    download_boltz2,
    check_inputs,
    process_inputs,
    filter_inputs_structure,
    BoltzProcessedInput,
    Boltz2DiffusionParams,
    PairformerArgsV2,
    BoltzDiffusionParams,
    PairformerArgs,
    MSAModuleArgs,
    BoltzSteeringParams,
    filter_inputs_affinity,
    get_cache_path,
)

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from rdkit import Chem

from src.boltz.data import const
from src.boltz.data.module.inference import BoltzInferenceDataModule
from src.boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from src.boltz.data.types import (
    Manifest,
)
from src.boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
from src.boltz.model.models.boltz1 import Boltz1
from src.boltz.model.models.boltz2 import Boltz2
from src.utils import get_project_root


@click.group()
def cli() -> None:
    """This Boltz has been modded to remove pockets that mess up the affinity
    predictions"""
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help=(
        "The directory where to download the data and model. "
        "Default is ~/.boltz, or $BOLTZ_CACHE if set."
    ),
    default=get_cache_path,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--max_parallel_samples",
    type=int,
    help="The maximum number of samples to predict in parallel. Default is None.",
    default=5,
)
@click.option(
    "--step_scale",
    type=float,
    help=(
        "The step size is related to the temperature at "
        "which the diffusion process samples the distribution. "
        "The lower the higher the diversity among samples "
        "(recommended between 1 and 2). "
        "Default is 1.638 for Boltz-1 and 1.5 for Boltz-2. "
        "If not provided, the default step size will be used."
    ),
    default=None,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help=(
        "Pairing strategy to use. Used only if --use_msa_server is set. "
        "Options are 'greedy' and 'complete'"
    ),
    default="greedy",
)
@click.option(
    "--use_potentials",
    is_flag=True,
    help="Whether to not use potentials for steering. Default is False.",
)
@click.option(
    "--model",
    default="boltz2",
    type=click.Choice(["boltz1", "boltz2"]),
    help="The model to use for prediction. Default is boltz2.",
)
@click.option(
    "--method",
    type=str,
    help="The method to use for prediction. Default is None.",
    default=None,
)
@click.option(
    "--preprocessing-threads",
    type=int,
    help="The number of threads to use for preprocessing. Default is 1.",
    default=multiprocessing.cpu_count(),
)
@click.option(
    "--affinity_mw_correction",
    is_flag=True,
    type=bool,
    help="Whether to add the Molecular Weight correction to the affinity value head.",
)
@click.option(
    "--sampling_steps_affinity",
    type=int,
    help="The number of sampling steps to use for affinity prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples_affinity",
    type=int,
    help="The number of diffusion samples to use for affinity prediction. Default is 5.",
    default=5,
)
@click.option(
    "--affinity_checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--max_msa_seqs",
    type=int,
    help="The maximum number of MSA sequences to use for prediction. Default is 8192.",
    default=8192,
)
@click.option(
    "--subsample_msa",
    is_flag=True,
    help="Whether to subsample the MSA. Default is True.",
)
@click.option(
    "--num_subsampled_msa",
    type=int,
    help="The number of MSA sequences to subsample. Default is 1024.",
    default=1024,
)
@click.option(
    "--no_kernels",
    is_flag=True,
    help="Whether to disable the kernels. Default False",
)
def predict(  # noqa: C901, PLR0915, PLR0912
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    affinity_checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 3,
    max_parallel_samples: Optional[int] = None,
    step_scale: Optional[float] = None,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    use_potentials: bool = False,
    model: Literal["boltz1", "boltz2"] = "boltz2",
    method: Optional[str] = None,
    affinity_mw_correction: Optional[bool] = False,
    preprocessing_threads: int = 1,
    max_msa_seqs: int = 8192,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    no_kernels: bool = False,
) -> None:
    """Run predictions with Boltz."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Supress some lightning warnings
    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set rdkit pickle logic
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        # Disable kernel tuning by default,
        # but do not modify envvar if already set by caller
        os.environ[key] = os.environ.get(key, "1")

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    if model == "boltz1":
        download_boltz1(cache)
    elif model == "boltz2":
        download_boltz2(cache)
    else:
        msg = f"Model {model} not supported. Supported: boltz1, boltz2."
        raise ValueError(f"Model {model} not supported.")

    # Validate inputs
    data = check_inputs(data)

    # Check method
    if method is not None:
        if model == "boltz1":
            msg = "Method conditioning is not supported for Boltz-1."
            raise ValueError(msg)
        if method.lower() not in const.method_types_ids:
            method_names = list(const.method_types_ids.keys())
            msg = f"Method {method} not supported. Supported: {method_names}"
            raise ValueError(msg)

    print(f"Using method: {method}")

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        boltz2=model == "boltz2",
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=max_msa_seqs,
    )

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    # Filter out existing predictions
    filtered_manifest = filter_inputs_structure(
        manifest=manifest,
        outdir=out_dir,
        override=override,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        start_method = "fork" if platform.system() != "win32" else "spawn"
        strategy = DDPStrategy(start_method=start_method)
        if len(filtered_manifest.records) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions, taking the minimum."
            )
            click.echo(msg)
            if isinstance(devices, list):
                devices = devices[: max(1, len(filtered_manifest.records))]
            else:
                devices = max(1, min(len(filtered_manifest.records), devices))

    # Set up model parameters
    if model == "boltz2":
        diffusion_params = Boltz2DiffusionParams()
        step_scale = 1.5 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgsV2()
    else:
        diffusion_params = BoltzDiffusionParams()
        step_scale = 1.638 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgs()

    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=model == "boltz2",
    )

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
        boltz2=model == "boltz2",
    )

    # Set up trainer
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32 if model == "boltz1" else "bf16-mixed",
    )

    if filtered_manifest.records:
        msg = f"Running structure prediction for {len(filtered_manifest.records)} input"
        msg += "s." if len(filtered_manifest.records) > 1 else "."
        click.echo(msg)

        # Create data module
        if model == "boltz2":
            data_module = Boltz2InferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                mol_dir=mol_dir,
                num_workers=num_workers,
                constraints_dir=processed.constraints_dir,
                template_dir=processed.template_dir,
                extra_mols_dir=processed.extra_mols_dir,
                override_method=method,
            )
        else:
            data_module = BoltzInferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                num_workers=num_workers,
                constraints_dir=processed.constraints_dir,
            )

        # Load model
        if checkpoint is None:
            if model == "boltz2":
                checkpoint = cache / "boltz2_conf.ckpt"
            else:
                checkpoint = cache / "boltz1_conf.ckpt"

        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": write_full_pae,
            "write_full_pde": write_full_pde,
        }

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = use_potentials
        steering_args.guidance_update = use_potentials

        model_cls = Boltz2 if model == "boltz2" else Boltz1
        model_module = model_cls.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not no_kernels,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        model_module.eval()

        # TODO: REMOVE THIS
        # example_input = {
        #     "token_index": torch.randint(25, size=(1, 397), dtype=torch.int64),
        #     "residue_index": torch.randint(25, size=(1, 397), dtype=torch.int64),
        #     "asym_id": torch.randint(25, size=(1, 397), dtype=torch.int64),
        #     "entity_id": torch.randint(25, size=(1, 397), dtype=torch.int64),
        #     "sym_id": torch.randint(25, size=(1, 397), dtype=torch.int64),
        #     "mol_type": torch.randint(25, size=(1, 397), dtype=torch.int64),
        #     "res_type": torch.randint(50, size=(1, 397, 33), dtype=torch.int64),
        #     "disto_center": torch.rand(1, 397, 3, dtype=torch.float32),
        #     "token_bonds": torch.rand(1, 397, 397, 1, dtype=torch.float32),
        #     "type_bonds": torch.randint(50, size=(1, 397, 397), dtype=torch.int64),
        #     "token_pad_mask": torch.rand(1, 397, dtype=torch.float32),
        #     "token_resolved_mask": torch.rand(1, 397, dtype=torch.float32),
        #     "token_disto_mask": torch.rand(1, 397, dtype=torch.float32),
        #     "contact_conditioning": torch.randint(
        #         50, size=(1, 397, 397, 5), dtype=torch.int64
        #     ),
        #     "contact_threshold": torch.rand(1, 397, 397, dtype=torch.float32),
        #     "method_feature": torch.randint(25, size=(1, 397), dtype=torch.int64),
        #     "modified": torch.randint(25, size=(1, 397), dtype=torch.int64),
        #     "cyclic_period": torch.zeros(1, 397, dtype=torch.float32),
        #     "affinity_token_mask": torch.rand(1, 397, dtype=torch.float32),
        #     "ref_pos": torch.rand(1, 2976, 3, dtype=torch.float32),
        #     "atom_resolved_mask": torch.randint(2, size=(1, 2976), dtype=torch.bool),
        #     "ref_atom_name_chars": torch.randint(
        #         20, size=(1, 2976, 4, 64), dtype=torch.int64
        #     ),
        #     "ref_element": torch.randint(25, size=(1, 2976, 128), dtype=torch.int64),
        #     "ref_charge": torch.rand(1, 2976, dtype=torch.float32),
        #     "ref_chirality": torch.randint(25, size=(1, 2976), dtype=torch.int64),
        #     "atom_backbone_feat": torch.randint(
        #         25, size=(1, 2976, 17), dtype=torch.int64
        #     ),
        #     "ref_space_uid": torch.randint(25, size=(1, 2976), dtype=torch.int64),
        #     "coords": torch.rand(1, 1, 2976, 3, dtype=torch.float32),
        #     "atom_pad_mask": torch.rand(1, 2976, dtype=torch.float32),
        #     "atom_to_token": torch.randint(25, size=(1, 2976, 397), dtype=torch.int64),
        #     "token_to_rep_atom": torch.randint(
        #         25, size=(1, 397, 2976), dtype=torch.int64
        #     ),
        #     "r_set_to_rep_atom": torch.randint(
        #         25, size=(1, 384, 2976), dtype=torch.int64
        #     ),
        #     "token_to_center_atom": torch.randint(
        #         25, size=(1, 397, 2976), dtype=torch.int64
        #     ),
        #     "disto_target": torch.rand(1, 397, 397, 1, 64, dtype=torch.float32),
        #     "disto_coords_ensemble": torch.rand(1, 1, 397, 3, dtype=torch.float32),
        #     "bfactor": torch.rand(1, 2976, dtype=torch.float32),
        #     "plddt": torch.rand(1, 2976, dtype=torch.float32),
        #     "frames_idx": torch.randint(10, size=(1, 1, 397, 3), dtype=torch.int64),
        #     "frame_resolved_mask": torch.randint(2, size=(1, 1, 397), dtype=torch.bool),
        #     "msa": torch.randint(10, size=(1, 2129, 397), dtype=torch.int64),
        #     "msa_paired": torch.rand(1, 2129, 397, dtype=torch.float32),
        #     "deletion_value": torch.rand(1, 2129, 397, dtype=torch.float32),
        #     "has_deletion": torch.randint(2, size=(1, 2129, 397), dtype=torch.bool),
        #     "deletion_mean": torch.rand(1, 397, dtype=torch.float32),
        #     "profile": torch.rand(1, 397, 33, dtype=torch.float32),
        #     "msa_mask": torch.randint(10, size=(1, 2129, 397), dtype=torch.int64),
        #     "template_restype": torch.randint(
        #         10, size=(1, 1, 397, 33), dtype=torch.int64
        #     ),
        #     "template_frame_rot": torch.rand(1, 1, 397, 3, 3, dtype=torch.float32),
        #     "template_frame_t": torch.rand(1, 1, 397, 3, dtype=torch.float32),
        #     "template_cb": torch.rand(1, 1, 397, 3, dtype=torch.float32),
        #     "template_ca": torch.rand(1, 1, 397, 3, dtype=torch.float32),
        #     "template_mask_cb": torch.rand(1, 1, 397, dtype=torch.float32),
        #     "template_mask_frame": torch.rand(1, 1, 397, dtype=torch.float32),
        #     "template_mask": torch.rand(1, 1, 397, dtype=torch.float32),
        #     "query_to_template": torch.randint(10, size=(1, 1, 397), dtype=torch.int64),
        #     "visibility_ids": torch.rand(1, 1, 397, dtype=torch.float32),
        #     "ensemble_ref_idxs": torch.randint(25, size=(1, 1), dtype=torch.int64),
        #     "rdkit_bounds_index": torch.randint(25, size=(1, 2, 78), dtype=torch.int64),
        #     "rdkit_bounds_bond_mask": torch.randint(2, size=(1, 78), dtype=torch.bool),
        #     "rdkit_bounds_angle_mask": torch.randint(2, size=(1, 78), dtype=torch.bool),
        #     "rdkit_upper_bounds": torch.rand(1, 78, dtype=torch.float32),
        #     "rdkit_lower_bounds": torch.rand(1, 78, dtype=torch.float32),
        #     "chiral_atom_index": torch.randint(10, size=(1, 4, 0), dtype=torch.int64),
        #     "chiral_reference_mask": torch.randint(2, size=(1, 0), dtype=torch.bool),
        #     "chiral_atom_orientations": torch.randint(2, size=(1, 0), dtype=torch.bool),
        #     "stereo_bond_index": torch.randint(100, size=(1, 4, 0), dtype=torch.int64),
        #     "stereo_reference_mask": torch.randint(2, size=(1, 0), dtype=torch.bool),
        #     "stereo_bond_orientations": torch.randint(2, size=(1, 0), dtype=torch.bool),
        #     "planar_bond_index": torch.randint(10, size=(1, 6, 0), dtype=torch.int64),
        #     "planar_ring_5_index": torch.randint(10, size=(1, 5, 0), dtype=torch.int64),
        #     "planar_ring_6_index": torch.randint(10, size=(1, 6, 1), dtype=torch.int64),
        #     "connected_chain_index": torch.randint(
        #         10, size=(1, 2, 0), dtype=torch.int64
        #     ),
        #     "connected_atom_index": torch.randint(
        #         10, size=(1, 2, 0), dtype=torch.int64
        #     ),
        #     "symmetric_chain_index": torch.randint(
        #         10, size=(1, 2, 0), dtype=torch.int64
        #     ),
        #     "record": [
        #         Record(
        #             id="affinity",
        #             structure=StructureInfo(),
        #             chains=[
        #                 ChainInfo(
        #                     chain_id=0,
        #                     chain_name="A",
        #                     mol_type=0,
        #                     cluster_id=-1,
        #                     msa_id="affinity_0",
        #                     num_residues=384,
        #                     valid=True,
        #                     entity_id=0,
        #                 ),
        #                 ChainInfo(
        #                     chain_id=1,
        #                     chain_name="B",
        #                     mol_type=3,
        #                     cluster_id=-1,
        #                     msa_id=-1,
        #                     num_residues=1,
        #                     valid=True,
        #                     entity_id=1,
        #                 ),
        #             ],
        #             interfaces=[],
        #             inference_options=InferenceOptions(
        #                 pocket_constraints=[(1, [(0, 25)], 6.0)]
        #             ),
        #             templates=[],
        #             md=None,
        #             affinity=AffinityInfo(chain_id=1, mw=171.11099999999996),
        #         )
        #     ],
        # }

        # print("Visualizing model architecture...")
        # model2 = model_cls.load_from_checkpoint(
        #     checkpoint,
        #     strict=True,
        #     predict_args=predict_args,
        #     map_location="cpu",
        #     diffusion_process_args=asdict(diffusion_params),
        #     ema=False,
        #     use_kernels=False,
        #     pairformer_args=asdict(pairformer_args),
        #     msa_args=asdict(msa_args),
        #     steering_args=asdict(steering_args),
        # )
        # model2.eval()
        # model_graph = draw_graph(
        #     model2,
        #     device="cuda",
        #     input_data={
        #         "feats": example_input,
        #     },
        # )
        # model_graph.visual_graph.render("boltz_architecture", format="png")
        # print("Model architecture visualized and saved as 'boltz_architecture.png'.")

        print("Loading picked input")
        with open(get_project_root() / "output" / "feats.pkl", "rb") as f:
            example_input = pickle.load(f)

        print("Visualizing model architecture...")
        model2 = model_cls.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        model2.eval()
        model_graph = draw_graph(
            model2,
            device="cpu",
            input_data={
                "feats": example_input,
            },
        )
        model_graph.visual_graph.render("boltz_architecture", format="png")
        print("Model architecture visualized and saved as 'boltz_architecture.png'.")

        # Compute structure predictions
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )

    # Check if affinity predictions are needed
    if any(r.affinity for r in manifest.records):
        # Print header
        click.echo("\nPredicting property: affinity\n")

        # Validate inputs
        manifest_filtered = filter_inputs_affinity(
            manifest=manifest,
            outdir=out_dir,
            override=override,
        )
        if not manifest_filtered.records:
            click.echo("Found existing affinity predictions for all inputs, skipping.")
            return

        msg = f"Running affinity prediction for {len(manifest_filtered.records)} input"
        msg += "s." if len(manifest_filtered.records) > 1 else "."
        click.echo(msg)

        pred_writer = BoltzAffinityWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
        )

        data_module = Boltz2InferenceDataModule(
            manifest=manifest_filtered,
            target_dir=out_dir / "predictions",
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method="other",
            affinity=True,
        )

        print("SETTINGS:")
        print("--" * 20)
        print(data_module)

        print("**" * 20)
        for attr_name in [
            "affinity",
            "mol_dir",
            "num_workers",
            "manifest",
            "target_dir",
            "msa_dir",
            "mol_dir",
            "constraints_dir",
            "template_dir",
            "extra_mols_dir",
            "override_method",
            "affinity",
        ]:
            attr_value = getattr(data_module, attr_name)
            print(f"{attr_name}: {attr_value}")

        print("--" * 20)

        predict_affinity_args = {
            "recycling_steps": 5,
            "sampling_steps": sampling_steps_affinity,
            "diffusion_samples": diffusion_samples_affinity,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        # Load affinity model
        if affinity_checkpoint is None:
            affinity_checkpoint = cache / "boltz2_aff.ckpt"
            click.echo(f"Loading affinity checkpoint from {affinity_checkpoint}")

        model_module = Boltz2.load_from_checkpoint(
            affinity_checkpoint,
            strict=True,
            predict_args=predict_affinity_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args={"fk_steering": False, "guidance_update": False},
            affinity_mw_correction=affinity_mw_correction,
        )
        model_module.eval()

        trainer.callbacks[0] = pred_writer

        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )


if __name__ == "__main__":
    EXAMPLE_TARGET_DIR = (
        Path(__file__).parent.parent / "output" / "boltz_results_affinity"
    )

    if EXAMPLE_TARGET_DIR.exists():
        print(f"Removing existing example output directory: {EXAMPLE_TARGET_DIR}")
        shutil.rmtree(EXAMPLE_TARGET_DIR)
    else:
        print(
            f"Example output directory does not exist: {EXAMPLE_TARGET_DIR}. Proceeding."
        )

    cli()
