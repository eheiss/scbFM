"""
Generate KPGT drug embeddings for GDSC compounds.

Pipeline:
  1. Read IC50 CSV → extract unique (drug_id, smiles) pairs
  2. Build KPGT-compatible input (molecular graphs, fingerprints, descriptors)
  3. Run pretrained KPGT model → 2304-dim embeddings per drug
  4. Save drug_features.npz in the GDSC directory

Usage:
    python generate_drug_features.py \
        --ic50_path "$SCBFM_ROOT_DIR/datasets/GDSC/drug_response_prediction_IC50.csv" \
        --output_dir "$SCBFM_ROOT_DIR/datasets/GDSC" \
        --kpgt_root "$SCBFM_ROOT_DIR/other/KPGT" \
        --model_path "$SCBFM_ROOT_DIR/other/KPGT/pretrained/base/base.pth"

Output:
    {output_dir}/drug_features.npz
        drug_ids : (n_drugs,) str array  — drug IDs matching IC50 CSV
        features : (n_drugs, 2304) float32 array — KPGT embeddings
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic50_path", type=str, required=True,
                        help="Path to drug_response_prediction_IC50.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write drug_features.npz")
    parser.add_argument("--kpgt_root", type=str, required=True,
                        help="Path to the KPGT repository root (e.g. .../other/KPGT)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained KPGT checkpoint (.pth); "
                             "defaults to {kpgt_root}/pretrained/base/base.pth")
    parser.add_argument("--config", type=str, default="base",
                        help="KPGT model config name (default: base)")
    parser.add_argument("--drug_id_col", type=str, default="Drug ID",
                        help="Column name for drug IDs in IC50 CSV")
    parser.add_argument("--smiles_col", type=str, default="smiles",
                        help="Column name for SMILES strings")
    parser.add_argument("--smiles_path", type=str, default=None,
                        help="Optional separate CSV with drug ID + SMILES columns. "
                             "If not given, SMILES are read from the IC50 CSV itself.")
    parser.add_argument("--dataset_col", type=str, default="Dataset Version",
                        help="Column name for GDSC version (GDSC1/GDSC2)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--path_length", type=int, default=5)
    return parser.parse_args()


def extract_unique_drugs(ic50_path, drug_id_col, smiles_col, dataset_col, smiles_path=None):
    """
    Return unique (drug_id, smiles) pairs.

    If smiles_path is given, SMILES are loaded from that separate CSV and
    joined onto the IC50 drug IDs. Otherwise SMILES are read directly from
    the IC50 CSV (requires a SMILES column there).
    """
    df = pd.read_csv(ic50_path)
    print(f"Loaded IC50 CSV: {len(df)} rows, columns: {list(df.columns)}")
    df[drug_id_col] = df[drug_id_col].astype(str)

    # Prefer GDSC2 over GDSC1 when both exist for the same drug
    df_sorted = df.sort_values(
        dataset_col,
        key=lambda s: s.map({"GDSC1": 0, "GDSC2": 1}).fillna(0),
        ascending=True,
    )
    unique_ids = (
        df_sorted.drop_duplicates(subset=[drug_id_col], keep="last")[[drug_id_col]]
        .reset_index(drop=True)
    )

    if smiles_path is not None:
        smiles_df = pd.read_csv(smiles_path)
        print(f"Loaded SMILES CSV: {len(smiles_df)} rows, columns: {list(smiles_df.columns)}")
        smiles_df[drug_id_col] = smiles_df[drug_id_col].astype(str)
        unique_drugs = unique_ids.merge(
            smiles_df[[drug_id_col, smiles_col]], on=drug_id_col, how="inner"
        )
    else:
        # SMILES expected as a column in the IC50 CSV itself
        unique_drugs = (
            df_sorted.drop_duplicates(subset=[drug_id_col], keep="last")
            [[drug_id_col, smiles_col]]
            .reset_index(drop=True)
        )

    unique_drugs = unique_drugs.dropna(subset=[smiles_col]).reset_index(drop=True)
    unique_drugs[drug_id_col] = unique_drugs[drug_id_col].astype(str)
    print(f"Unique drugs with SMILES: {len(unique_drugs)}")
    return unique_drugs


def build_kpgt_input(unique_drugs, drug_id_col, smiles_col, kpgt_data_dir, n_jobs, path_length):
    """
    Write KPGT-compatible CSV and run preprocessing to build graphs,
    fingerprints, and molecular descriptors.
    """
    from dgl.data.utils import save_graphs
    from dgllife.utils.io import pmap
    from multiprocessing import Pool
    from rdkit import Chem
    from scipy import sparse as sp

    from src.data.featurizer import smiles_to_graph_tune
    from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
    import dgl.backend as F

    dataset_name = "gdsc"
    dataset_dir = os.path.join(kpgt_data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Write input CSV — KPGT expects a 'smiles' column + at least one label column
    kpgt_csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
    kpgt_df = unique_drugs[[smiles_col]].rename(columns={smiles_col: "smiles"})
    kpgt_df["drug_id"] = unique_drugs[drug_id_col].values
    kpgt_df["dummy_label"] = 0.0
    kpgt_df.to_csv(kpgt_csv_path, index=False)
    print(f"Wrote KPGT input CSV: {kpgt_csv_path}")

    smiless = kpgt_df["smiles"].tolist()

    # Build molecular graphs
    cache_path = os.path.join(dataset_dir, f"{dataset_name}_{path_length}.pkl")
    if not os.path.exists(cache_path):
        print("Building molecular graphs ...")
        graphs = pmap(
            smiles_to_graph_tune,
            smiless,
            max_length=path_length,
            n_virtual_nodes=2,
            n_jobs=n_jobs,
        )
        valid_ids, valid_graphs = [], []
        for i, g in enumerate(graphs):
            if g is not None:
                valid_ids.append(i)
                valid_graphs.append(g)
        dummy_labels = F.zerocopy_from_numpy(
            np.zeros((len(valid_ids), 1), dtype=np.float32)
        )
        save_graphs(cache_path, valid_graphs, labels={"labels": dummy_labels})
        print(f"  Saved {len(valid_graphs)} graphs → {cache_path}")
    else:
        print(f"Graph cache already exists: {cache_path}")

    # RDKit fingerprints
    fp_path = os.path.join(dataset_dir, "rdkfp1-7_512.npz")
    if not os.path.exists(fp_path):
        print("Extracting RDKit fingerprints ...")
        FP_list = []
        for smiles in smiless:
            mol = Chem.MolFromSmiles(smiles)
            FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
        sp.save_npz(fp_path, sp.csc_matrix(np.array(FP_list)))
        print(f"  Saved fingerprints → {fp_path}")
    else:
        print(f"Fingerprints already exist: {fp_path}")

    # Molecular descriptors
    md_path = os.path.join(dataset_dir, "molecular_descriptors.npz")
    if not os.path.exists(md_path):
        print("Extracting molecular descriptors ...")
        generator = RDKit2DNormalized()
        features_map = Pool(n_jobs).imap(generator.process, smiless)
        arr = np.array(list(features_map))
        np.savez_compressed(md_path, md=arr[:, 1:])
        print(f"  Saved descriptors → {md_path}")
    else:
        print(f"Descriptors already exist: {md_path}")

    return kpgt_data_dir, dataset_name


def extract_features(kpgt_data_dir, dataset_name, model_path, config_name, batch_size, path_length):
    """Run the KPGT model to generate per-molecule embeddings."""
    from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
    from src.data.finetune_dataset import MoleculeDataset
    from src.data.collator import Collator_tune
    from src.model.light import LiGhTPredictor as LiGhT
    from src.model_config import config_dict

    config = config_dict[config_name]
    # DGL in the container is CPU-only; torch tensors run on CPU accordingly.
    device = torch.device("cpu")
    print(f"Running KPGT on {device}")

    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    collator = Collator_tune(config["path_length"])
    mol_dataset = MoleculeDataset(
        root_path=kpgt_data_dir,
        dataset=dataset_name,
        dataset_type=None,
        path_length=path_length,
    )
    print(f"MoleculeDataset: {len(mol_dataset)} molecules")
    loader = DataLoader(
        mol_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collator,
    )

    model = LiGhT(
        d_node_feats=config["d_node_feats"],
        d_edge_feats=config["d_edge_feats"],
        d_g_feats=config["d_g_feats"],
        d_hpath_ratio=config["d_hpath_ratio"],
        n_mol_layers=config["n_mol_layers"],
        path_length=config["path_length"],
        n_heads=config["n_heads"],
        n_ffn_dense_layers=config["n_ffn_dense_layers"],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size,
    ).to(device)
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in torch.load(model_path, map_location=device).items()}
    )
    model.eval()

    fps_list = []
    with torch.no_grad():
        for batch_idx, batched_data in enumerate(loader):
            _, g, ecfp, md, _ = batched_data
            ecfp = ecfp.to(device)
            md = md.to(device)
            g = g.to(device)
            fps = model.generate_fps(g, ecfp, md)
            fps_list.extend(fps.detach().cpu().numpy().tolist())
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} molecules ...")

    return np.array(fps_list, dtype=np.float32)


def main():
    args = parse_args()

    # Add KPGT to Python path
    sys.path.insert(0, args.kpgt_root)

    # Resolve default model path
    if args.model_path is None:
        args.model_path = os.path.join(args.kpgt_root, "pretrained", "base", "base.pth")
    print(f"Using model: {args.model_path}")

    # Step 1: extract unique drugs from IC50 CSV
    unique_drugs = extract_unique_drugs(
        args.ic50_path, args.drug_id_col, args.smiles_col, args.dataset_col,
        smiles_path=args.smiles_path,
    )
    drug_ids = unique_drugs[args.drug_id_col].values

    # Step 2: build KPGT input in a temp subdirectory inside output_dir
    kpgt_data_dir = os.path.join(args.output_dir, "kpgt_tmp")
    kpgt_data_dir, dataset_name = build_kpgt_input(
        unique_drugs, args.drug_id_col, args.smiles_col,
        kpgt_data_dir, args.n_jobs, args.path_length,
    )

    # Step 3: extract KPGT features
    features = extract_features(
        kpgt_data_dir, dataset_name, args.model_path,
        args.config, args.batch_size, args.path_length,
    )
    print(f"Extracted features shape: {features.shape}")

    if len(features) != len(drug_ids):
        raise ValueError(
            f"Alignment mismatch: {len(features)} feature rows vs "
            f"{len(drug_ids)} drug_ids. Some molecules may have failed "
            f"graph construction — delete the kpgt_tmp cache and re-run."
        )

    # Step 4: save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "drug_features.npz")
    np.savez_compressed(out_path, drug_ids=drug_ids, features=features)
    print(f"Saved drug features → {out_path}")
    print(f"  drug_ids : {drug_ids.shape}")
    print(f"  features : {features.shape}  (dtype: {features.dtype})")


if __name__ == "__main__":
    main()
