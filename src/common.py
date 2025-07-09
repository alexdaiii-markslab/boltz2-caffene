import os
import warnings
from pathlib import Path
from typing import Literal, NamedTuple

import torch
from boltz.data import const
from boltz.data.types import Manifest
from boltz.main import (
    download_boltz1,
    check_inputs,
    download_boltz2,
    process_inputs,
    filter_inputs_structure,
    BoltzProcessedInput,
)
from pytorch_lightning import seed_everything
from rdkit import Chem


def setup_environment(
    seed: int | None = None,
):
    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )
    #
    # # Set no grad
    torch.set_grad_enabled(False)
    #
    # # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")
    #
    # # Set rdkit pickle logic
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    #
    # # Set seed if desired
    if seed is not None:
        seed_everything(seed)
    #
    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        # Disable kernel tuning by default,
        # but do not modify envvar if already set by caller
        os.environ[key] = os.environ.get(key, "1")


class SetupIoOutput(NamedTuple):
    out_dir: Path
    cache: Path
    data: Path


def setup_io(
    *,
    data: str,
    cache: str = "~/.boltz",
    out_dir: str,
    model: Literal["boltz1", "boltz2"],
) -> SetupIoOutput:
    #
    # # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    #
    # # Create output directories
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

    return SetupIoOutput(
        out_dir=out_dir,
        cache=cache,
        data=data,
    )


def full_validate_inputs(
    *,
    data: Path,
    method: str | None,
    model: Literal["boltz1", "boltz2"],
    cache: Path,
    out_dir: Path,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    preprocessing_threads: int,
    max_msa_seqs: int,
):
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


class LoadModelSetupRt(NamedTuple):
    processed: BoltzProcessedInput
    manifest: Manifest
    filtered_manifest: Manifest


def load_model_setup(
    *,
    out_dir: Path,
    override: bool,
) -> LoadModelSetupRt:
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

    return LoadModelSetupRt(
        processed=processed,
        manifest=manifest,
        filtered_manifest=filtered_manifest,
    )
