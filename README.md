# Boltz2 Binding for caffeine

## Intriduction

We are looking at the binding of caffeine to RYR1, which has a caffeine
binding site. We will be testing different xanthine derivatives to see
if they bind and which is predicted to bind the best. 

We are using Boltz2 to perform in-silico experiments to predict the binding of
the xanthine derivatives to RYR2. It will give 2 predictions:

- **affinity_pred_value** - measure the specific affinity of different binders and 
  how this changes with small modifications of the molecule. 
  This should be used in ligand optimization stages such as hit-to-lead and 
  lead-optimization. It reports a binding affinity value as `log(IC50)`, 
  derived from an IC50 measured in μM
- **affinity_probability_binary**: used to detect binders from decoys, 
  for example in a hit-discovery stage. It's value ranges from 0 to 1 and 
  represents the predicted probability that the ligand is a binder

The inputs for this are a yaml file. See the Boltz2 
[prediction instructions](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md)
for more details. The most important parts are the `sequence` and an optional
`constraints` section.

```yaml
sequences:
    - ENTITY_TYPE:
        id: CHAIN_ID 
        sequence: SEQUENCE    # only for protein, dna, rna
        smiles: 'SMILES'        # only for ligand, exclusive with ccd
        ccd: CCD              # only for ligand, exclusive with smiles
        msa: MSA_PATH         # only for protein
        modifications:
          - position: RES_IDX   # index of residue, starting from 1
            ccd: CCD            # CCD code of the modified residue
        cyclic: false
    - ENTITY_TYPE:
        id: [CHAIN_ID, CHAIN_ID]    # multiple ids in case of multiple identical entities
        ...
constraints:
    - bond:
        atom1: [CHAIN_ID, RES_IDX, ATOM_NAME]
        atom2: [CHAIN_ID, RES_IDX, ATOM_NAME]
    - pocket:
        binder: CHAIN_ID
        contacts: [[CHAIN_ID, RES_IDX/ATOM_NAME], [CHAIN_ID, RES_IDX/ATOM_NAME]]
        max_distance: DIST_ANGSTROM
    - contact:
        token1: [CHAIN_ID, RES_IDX/ATOM_NAME]
        token2: [CHAIN_ID, RES_IDX/ATOM_NAME]
        max_distance: DIST_ANGSTROM

templates:
    - cif: CIF_PATH  # if only a path is provided, Boltz will find the best matchings
    - cif: CIF_PATH
      chain_id: CHAIN_ID   # optional, specify which chain to find a template for
    - cif: CIF_PATH
      chain_id: [CHAIN_ID, CHAIN_ID]  # can be more than one
      template_id: [TEMPLATE_CHAIN_ID, TEMPLATE_CHAIN_ID]
properties:
    - affinity:
        binder: CHAIN_ID
```

The `sequence` section will contain an `ENTITY_TYPE` that will be `protein` or
`ligand` in our case. When using `protein`, the sequence is required and the 
`msa` key is required by default but can be omited by passing the `--use_msa_server` 
flag which will auto-generate the MSA using the mmseqs2 server. When using
a precomputed MSA, use the `msa` attribute with `MSA_PATH`
indicating the path to the `.a3m` file containing the MSA for that protein.
The `ligand` should be specified using either the `smiles` or `ccd` key.

`constraints` is an optional field that can optionally contain a `pocket` field
for the ligand - so the model does not need to figure it out itself. The pocket 
constraint specifies the residues associated with a ligand, where 
binder refers to the chain binding to the pocket 
(which can be a molecule, protein, DNA or RNA) a
nd contacts is the list of chain and residue indices 
(starting from 1) associated with the pocket. 
The model currently only supports the specification of a single binder chain 
(and any number of contacts residues in other chains).

To calculate affinity, the properties section should contain an `affinity` with
the protein the affinity is calculated for. 

## Installation

To install, you can use a conda or pip venv. For Boltz2 2.1.1, there is a 
requirements.txt file that can be used to install the dependencies. The python
version should be exactly 3.12.X. The conda environment yaml may or may not
work depending on your system dependencies.

```bash
mamba create -n boltz2 python=3.12 pip
mamba activate boltz2
pip boltz2
```

## Running

### On Mondrian

This is a dual GPU system - 1 of them is used for display (GPU 0) and the other
is used for computation (GPU 1). To run the code, you can use the following command:

```bash
export CUDA_VISIBLE_DEVICES=1
boltz predict <YOUR_YAML_FILE> --out_dir output --use_msa_server
```

The export makes sure that the Boltz2 code uses the second GPU for computation.
To get the id of the GPU, you can use the `nvidia-smi` command. Choose the one
that is not used for display (since that uses VRAM to display the desktop).

## The A6000 Machine

```bash
boltz predict <YOUR_YAML_FILE> --out_dir output --use_msa_server
```
