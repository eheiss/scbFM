from __future__ import annotations

from pathlib import Path
import gc
import json
import re

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


# =========================
# Paths
# =========================

GENE_LIST_PATH = Path(__file__).resolve().parent / "gene_list.txt"
GTEX_PATH = Path("/cluster/work/boeva/eheiss/datasets/GTEx/gtex.h5ad")
ARCHS4_PATH = Path("/cluster/work/boeva/eheiss/datasets/ARCHS4/human_gene_v2.latest.h5")

OUT_DIR = Path("/cluster/work/boeva/eheiss/datasets/bulk")
ARCHS4_CHUNK_DIR = OUT_DIR / "archs4_RAW_chunks"
ARCHS4_MERGE_TMP_DIR = OUT_DIR / "archs4_RAW_merge_tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARCHS4_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
ARCHS4_MERGE_TMP_DIR.mkdir(parents=True, exist_ok=True)

GTEX_OUT = OUT_DIR / "gtex_RAW.h5ad"
ARCHS4_OUT = OUT_DIR / "archs4_RAW.h5ad"
PRETRAIN_OUT = OUT_DIR / "pretraining_bulk_RAW.h5ad"
PREADAPT_OUT = OUT_DIR / "preadapt_bulk_RAW.h5ad"
ARCHS4_GTEX_DONOR_HITS_OUT = OUT_DIR / "archs4_gtex_donor_hits_RAW.csv"
ARCHS4_DOWNSTREAM_HITS_OUT = OUT_DIR / "archs4_downstream_hits_RAW.csv"


# =========================
# Settings
# =========================

MIN_GENES = 200
ARCHS4_CHUNK_SIZE = 2000  # samples per chunk before cell filtering
MERGE_BATCH_SIZE = 16
PRETRAIN_SAMPLE_COUNT = 700_000
RANDOM_SEED = 42
GTEX_DONOR_PATTERN = re.compile(r"GTEX-[A-Z0-9]+")
METADATA_TOKEN_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9._:-]{2,}")

DOWNSTREAM_DATASET_PATHS = {
    "TCGA": Path("/cluster/work/boeva/eheiss/datasets/TCGA/tcga.h5ad"),
    "DepMap": Path("/cluster/work/boeva/eheiss/datasets/DepMap/depmap.h5ad"),
    "GDSC": Path("/cluster/work/boeva/eheiss/datasets/GDSC/gdsc.h5ad"),
    "DiSignAtlas": Path("/cluster/work/boeva/eheiss/datasets/DiSignAtlas/disignatlas.h5ad"),
}

DOWNSTREAM_GCTX_PATHS = {
    "LINCS": Path("/cluster/customapps/biomed/boeva/eheiss/downloads/level5_beta_all_n1201944x12328.gctx"),
}

DOWNSTREAM_DATASET_TERMS = {
    "TCGA": (
        "TCGA",
        "The Cancer Genome Atlas",
        "Cancer Genome Atlas",
    ),
    "DepMap": (
        "DepMap",
        "CCLE",
        "Cancer Cell Line Encyclopedia",
    ),
    "GDSC": (
        "GDSC",
        "Genomics of Drug Sensitivity in Cancer",
        "CancerRxGene",
        "Cancer RX Gene",
    ),
    "DiSignAtlas": (
        "DiSignAtlas",
        "Disease Signature Atlas",
    ),
    "LINCS": (
        "LINCS",
        "L1000",
        "Connectivity Map",
        "CMAP",
    ),
}

DOWNSTREAM_ID_PATTERNS = {
    "TCGA": (
        re.compile(r"\bTCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9]{2,4}){0,4}\b"),
    ),
    "DepMap": (
        re.compile(r"\bACH-\d{6}\b"),
    ),
    "LINCS": (
        re.compile(r"\b(?:LINCS|L1000)[-_:][A-Z0-9._:-]+\b"),
    ),
}


# =========================
# Helpers
# =========================

def read_gene_list(path: Path) -> list[str]:
    with open(path) as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def decode_bytes_array(arr) -> list[str]:
    out = []
    for x in arr:
        if hasattr(x, "decode"):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out


def decode_value(value) -> str:
    if hasattr(value, "decode"):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def gtex_donor_id(sample_id: str) -> str:
    parts = sample_id.upper().split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else sample_id.upper()


def extract_gtex_donors(value: str, valid_donors: set[str]) -> set[str]:
    return {match for match in GTEX_DONOR_PATTERN.findall(value.upper()) if match in valid_donors}


def normalize_metadata_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def term_in_metadata_value(value_upper: str, term: str) -> bool:
    term_upper = term.upper()
    if len(term_upper) <= 5 and term_upper.replace("-", "").isalnum():
        return re.search(rf"(?<![A-Z0-9]){re.escape(term_upper)}(?![A-Z0-9])", value_upper) is not None
    return term_upper in value_upper


def metadata_tokens_from_adata(path: Path) -> set[str]:
    if not path.exists():
        print(f"Downstream dataset not found, skipping ID extraction: {path}")
        return set()

    print(f"Loading downstream sample IDs from {path}...")
    adata = ad.read_h5ad(path, backed="r")
    try:
        tokens = {normalize_metadata_token(str(idx)) for idx in adata.obs_names}
    finally:
        adata.file.close()
    tokens = {token for token in tokens if len(token) >= 4}
    print(f"Loaded {len(tokens)} downstream sample IDs from {path.name}")
    return tokens


def hdf5_string_values(dataset: h5py.Dataset) -> list[str]:
    try:
        values = dataset.asstr()[:]
    except (AttributeError, TypeError):
        values = dataset[:]
    return [decode_value(value) for value in values]


def metadata_tokens_from_gctx(path: Path) -> set[str]:
    if not path.exists():
        print(f"Downstream GCTX not found, skipping ID extraction: {path}")
        return set()

    print(f"Loading downstream sample IDs from {path}...")
    candidate_names = {"ID", "SIG_ID", "SAMPLE_ID", "SAMPLE", "GEO_ID", "DISTIL_ID"}
    tokens: set[str] = set()

    with h5py.File(path, "r") as f:
        dataset_names: list[str] = []

        def collect_candidate(name: str, obj) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            upper_name = name.upper()
            base_name = upper_name.rsplit("/", 1)[-1]
            if "META/COL" in upper_name and base_name in candidate_names and obj.ndim == 1:
                dataset_names.append(name)

        f.visititems(collect_candidate)

        for dataset_name in dataset_names:
            values = hdf5_string_values(f[dataset_name])
            for value in values:
                token = normalize_metadata_token(value)
                if len(token) >= 4:
                    tokens.add(token)

    print(f"Loaded {len(tokens)} downstream sample IDs from {path.name}")
    return tokens


def build_downstream_id_tokens() -> dict[str, set[str]]:
    tokens = {
        dataset: metadata_tokens_from_adata(path)
        for dataset, path in DOWNSTREAM_DATASET_PATHS.items()
    }
    for dataset, path in DOWNSTREAM_GCTX_PATHS.items():
        tokens.setdefault(dataset, set()).update(metadata_tokens_from_gctx(path))
    return tokens


def extract_downstream_matches(
    value: str,
    downstream_id_tokens: dict[str, set[str]],
) -> dict[str, dict[str, set[str]]]:
    value_upper = value.upper()
    value_tokens = {normalize_metadata_token(token) for token in METADATA_TOKEN_PATTERN.findall(value_upper)}
    value_tokens = {token for token in value_tokens if len(token) >= 4}
    matches: dict[str, dict[str, set[str]]] = {}

    for dataset, terms in DOWNSTREAM_DATASET_TERMS.items():
        matched_terms = {term for term in terms if term_in_metadata_value(value_upper, term)}
        matched_ids = set()

        for pattern in DOWNSTREAM_ID_PATTERNS.get(dataset, ()):
            matched_ids.update(pattern.findall(value_upper))

        dataset_tokens = downstream_id_tokens.get(dataset, set())
        if dataset_tokens and value_tokens:
            matched_ids.update(sorted(value_tokens.intersection(dataset_tokens)))

        if matched_terms or matched_ids:
            matches[dataset] = {
                "matched_terms": matched_terms,
                "matched_ids": matched_ids,
            }

    return matches


def build_reindexer(source_gene_ids: list[str] | pd.Index, target_gene_list: list[str]) -> tuple[list[int], list[int], list[str]]:
    """
    Returns:
        src_pos: positions in source
        tgt_pos: positions in target gene list
        missing: target genes not present in source
    Preserves exact order of target_gene_list.
    """
    first_pos = {}
    for i, g in enumerate(source_gene_ids):
        if g not in first_pos:
            first_pos[g] = i

    src_pos = []
    tgt_pos = []
    missing = []

    for j, g in enumerate(target_gene_list):
        if g in first_pos:
            src_pos.append(first_pos[g])
            tgt_pos.append(j)
        else:
            missing.append(g)

    return src_pos, tgt_pos, missing


def place_into_target_order(x_present: np.ndarray, tgt_pos: list[int], total_genes: int, dtype=np.float32) -> np.ndarray:
    """
    x_present: (n_cells, n_present_genes)
    returns dense array (n_cells, total_genes), zeros for missing genes,
    with columns in exact order of target gene list.
    """
    out = np.zeros((x_present.shape[0], total_genes), dtype=dtype)
    out[:, tgt_pos] = x_present
    return out


def filter_raw_dense_block(
    x: np.ndarray,
    min_genes: int = MIN_GENES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    x: dense float array, shape (cells, genes)
    returns:
        x_raw_float32: shape (kept_cells, genes)
        keep_mask: bool mask over original cells
    """
    n_genes_by_cell = (x > 0).sum(axis=1)
    keep_mask = n_genes_by_cell >= min_genes
    x = x[keep_mask]

    if x.shape[0] == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32), keep_mask

    return x.astype(np.float32, copy=False), keep_mask


def write_manifest(paths: list[Path], out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump([str(p) for p in paths], f, indent=2)


def merge_h5ad_group(paths: list[Path], out_path: Path) -> Path:
    adatas = []
    for p in paths:
        adatas.append(ad.read_h5ad(p))

    merged = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)
    merged.obs_names_make_unique()
    merged.write(out_path)

    del adatas, merged
    gc.collect()

    return out_path


# =========================
# GTEx
# =========================

def preprocess_gtex(gene_list: list[str]) -> tuple[Path, set[str]]:
    print("Loading GTEx...")
    adata = ad.read_h5ad(GTEX_PATH)
    gtex_donors = {gtex_donor_id(sample_id) for sample_id in adata.obs_names.astype(str)}

    gtex_gene_ids = adata.var_names.astype(str)
    src_pos, tgt_pos, missing = build_reindexer(gtex_gene_ids, gene_list)

    print(f"GTEx genes present: {len(src_pos)} / {len(gene_list)}")
    print(f"GTEx genes missing: {len(missing)}")
    print(f"GTEx donor IDs: {len(gtex_donors)}")

    if sparse.issparse(adata.X):
        x_present = adata.X[:, src_pos].toarray().astype(np.float32)
    else:
        x_present = np.asarray(adata.X[:, src_pos], dtype=np.float32)

    x = place_into_target_order(x_present, tgt_pos, len(gene_list), dtype=np.float32)
    del x_present
    gc.collect()

    x, keep_mask = filter_raw_dense_block(x)

    obs = adata.obs.iloc[np.where(keep_mask)[0]].copy()
    obs["dataset"] = "GTEx"

    var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))

    out = ad.AnnData(X=sparse.csr_matrix(x), obs=obs, var=var)
    out.var_names = pd.Index(gene_list, dtype=str)

    out.write(GTEX_OUT)

    with open(OUT_DIR / "gtex_missing_genes_RAW.json", "w") as f:
        json.dump(missing, f)

    print(f"Saved {GTEX_OUT}")
    return GTEX_OUT, gtex_donors


# =========================
# ARCHS4 -> chunks
# =========================

def find_archs4_gtex_donor_hits(gtex_donors: set[str]) -> set[int]:
    print("Scanning ARCHS4 metadata for GTEx donor IDs...")
    hits: dict[int, dict[str, set[str]]] = {}

    with h5py.File(ARCHS4_PATH, "r") as f:
        sample_group = f["meta/samples"]
        keys = list(sample_group.keys())
        n_samples = len(sample_group["sample"])

        for key in keys:
            values = sample_group[key][:]
            for i, raw in enumerate(values):
                value = decode_value(raw)
                if "GTEX" not in value.upper():
                    continue
                matched_donors = extract_gtex_donors(value, gtex_donors)
                if not matched_donors:
                    continue
                record = hits.setdefault(
                    i,
                    {"matched_fields": set(), "matched_donors": set()},
                )
                record["matched_fields"].add(key)
                record["matched_donors"].update(matched_donors)

        rows = []
        for i, record in sorted(hits.items()):
            row = {
                "archs4_row": int(i),
                "matched_fields": ";".join(sorted(record["matched_fields"])),
                "matched_donors": ";".join(sorted(record["matched_donors"])),
            }
            for key in keys:
                row[key] = decode_value(sample_group[key][i])
            rows.append(row)

    pd.DataFrame(rows).to_csv(ARCHS4_GTEX_DONOR_HITS_OUT, index=False)
    print(
        f"ARCHS4 samples with GTEx donor ID hits: {len(hits)} / {n_samples}; "
        f"details saved to {ARCHS4_GTEX_DONOR_HITS_OUT}"
    )
    return set(hits)


def find_archs4_downstream_hits(
    downstream_id_tokens: dict[str, set[str]],
) -> set[int]:
    print("Scanning ARCHS4 metadata for downstream dataset leakage...")
    hits: dict[int, dict[str, set[str]]] = {}

    with h5py.File(ARCHS4_PATH, "r") as f:
        sample_group = f["meta/samples"]
        keys = list(sample_group.keys())
        n_samples = len(sample_group["sample"])

        for key in keys:
            values = sample_group[key][:]
            for i, raw in enumerate(values):
                matches = extract_downstream_matches(decode_value(raw), downstream_id_tokens)
                if not matches:
                    continue

                record = hits.setdefault(
                    i,
                    {
                        "matched_fields": set(),
                        "matched_datasets": set(),
                        "matched_terms": set(),
                        "matched_ids": set(),
                    },
                )
                record["matched_fields"].add(key)
                for dataset, match in matches.items():
                    record["matched_datasets"].add(dataset)
                    record["matched_terms"].update(
                        f"{dataset}:{term}" for term in match["matched_terms"]
                    )
                    record["matched_ids"].update(
                        f"{dataset}:{matched_id}" for matched_id in match["matched_ids"]
                    )

        rows = []
        for i, record in sorted(hits.items()):
            row = {
                "archs4_row": int(i),
                "matched_fields": ";".join(sorted(record["matched_fields"])),
                "matched_datasets": ";".join(sorted(record["matched_datasets"])),
                "matched_terms": ";".join(sorted(record["matched_terms"])),
                "matched_ids": ";".join(sorted(record["matched_ids"])),
            }
            for key in keys:
                row[key] = decode_value(sample_group[key][i])
            rows.append(row)

    pd.DataFrame(rows).to_csv(ARCHS4_DOWNSTREAM_HITS_OUT, index=False)
    print(
        f"ARCHS4 samples with downstream metadata hits: {len(hits)} / {n_samples}; "
        f"details saved to {ARCHS4_DOWNSTREAM_HITS_OUT}"
    )
    return set(hits)


def preprocess_archs4_to_chunks(
    gene_list: list[str],
    gtex_donor_hits: set[int],
    downstream_hits: set[int],
    chunk_size: int = ARCHS4_CHUNK_SIZE,
) -> list[Path]:
    print("Preparing ARCHS4 mappings...")
    with h5py.File(ARCHS4_PATH, "r") as f:
        archs4_gene_ids = decode_bytes_array(f["meta/genes/ensembl_gene"][:])
        sc_prob = np.asarray(f["meta/samples/singlecellprobability"][:], dtype=np.float32)
        sample_ids = decode_bytes_array(f["meta/samples/sample"][:])

    src_pos, tgt_pos, missing = build_reindexer(archs4_gene_ids, gene_list)
    bulk_like_idx = np.where(sc_prob < 0.5)[0]   # keep bulk-like samples
    bulk_like_before_filter = len(bulk_like_idx)
    if gtex_donor_hits:
        gtex_hit_mask = np.isin(bulk_like_idx, np.fromiter(gtex_donor_hits, dtype=np.int64))
        excluded_bulk_like_count = int(gtex_hit_mask.sum())
        bulk_like_idx = bulk_like_idx[~gtex_hit_mask]
    else:
        excluded_bulk_like_count = 0
    if downstream_hits:
        downstream_hit_mask = np.isin(bulk_like_idx, np.fromiter(downstream_hits, dtype=np.int64))
        excluded_downstream_count = int(downstream_hit_mask.sum())
        bulk_like_idx = bulk_like_idx[~downstream_hit_mask]
    else:
        excluded_downstream_count = 0

    print(f"ARCHS4 genes present: {len(src_pos)} / {len(gene_list)}")
    print(f"ARCHS4 genes missing: {len(missing)}")
    print(
        "ARCHS4 kept samples "
        f"(singlecellprobability < 0.5, before GTEx donor filtering): "
        f"{bulk_like_before_filter} / {len(sc_prob)}"
    )
    print(f"ARCHS4 bulk-like samples excluded by GTEx donor IDs: {excluded_bulk_like_count}")
    print(f"ARCHS4 bulk-like samples excluded by downstream metadata hits: {excluded_downstream_count}")
    print(f"ARCHS4 kept samples after leakage filtering: {len(bulk_like_idx)}")

    with open(OUT_DIR / "archs4_missing_genes_RAW.json", "w") as f:
        json.dump(missing, f)

    written_chunks: list[Path] = []

    with h5py.File(ARCHS4_PATH, "r") as f:
        expr = f["data/expression"]  # shape: (genes, samples)

        for chunk_id, start in enumerate(range(0, len(bulk_like_idx), chunk_size)):
            end = min(start + chunk_size, len(bulk_like_idx))
            cols = bulk_like_idx[start:end]

            print(f"ARCHS4 chunk {chunk_id}: source samples {start}:{end}")

            # Read only the selected sample block first to avoid materializing
            # a huge intermediate array over all ARCHS4 samples.
            sample_block = np.asarray(expr[:, cols], dtype=np.float32)
            x_present = sample_block[src_pos, :].T
            del sample_block

            # Put into exact target gene order
            x = place_into_target_order(x_present, tgt_pos, len(gene_list), dtype=np.float32)
            del x_present
            gc.collect()

            # Keep the same expressed-gene filter as the binned pipeline, but
            # leave counts otherwise untouched.
            x, keep_mask = filter_raw_dense_block(x)

            kept_cols = cols[np.where(keep_mask)[0]]
            obs = pd.DataFrame(index=pd.Index([sample_ids[i] for i in kept_cols], name="sample_id"))
            obs["singlecellprobability"] = sc_prob[kept_cols]
            obs["dataset"] = "ARCHS4"

            var = pd.DataFrame(index=pd.Index(gene_list, name="ensembl_id"))
            adata_chunk = ad.AnnData(X=sparse.csr_matrix(x), obs=obs, var=var)
            adata_chunk.var_names = pd.Index(gene_list, dtype=str)

            out_path = ARCHS4_CHUNK_DIR / f"archs4_RAW_chunk_{chunk_id:05d}.h5ad"
            adata_chunk.write(out_path)
            written_chunks.append(out_path)

            del x, obs, var, adata_chunk
            gc.collect()

    write_manifest(written_chunks, OUT_DIR / "archs4_RAW_chunk_manifest.json")
    print(f"Wrote {len(written_chunks)} ARCHS4 chunks")
    return written_chunks


# =========================
# ARCHS4 chunks -> one h5ad
# =========================

def merge_archs4_chunks(chunk_paths: list[Path]) -> Path:
    print("Merging ARCHS4 chunks...")
    if len(chunk_paths) == 0:
        raise ValueError("No ARCHS4 chunk files found.")

    current_paths = list(chunk_paths)
    round_id = 0

    while len(current_paths) > 1:
        next_paths: list[Path] = []
        for batch_id, start in enumerate(range(0, len(current_paths), MERGE_BATCH_SIZE)):
            batch_paths = current_paths[start:start + MERGE_BATCH_SIZE]
            out_path = ARCHS4_MERGE_TMP_DIR / f"archs4_RAW_merge_r{round_id:02d}_b{batch_id:04d}.h5ad"
            print(
                f"Merging ARCHS4 batch round {round_id}, batch {batch_id}: "
                f"{len(batch_paths)} files"
            )
            next_paths.append(merge_h5ad_group(batch_paths, out_path))
        current_paths = next_paths
        round_id += 1

    final_merged = ad.read_h5ad(current_paths[0])
    final_merged.obs_names_make_unique()
    final_merged.write(ARCHS4_OUT)

    del final_merged
    gc.collect()

    print(f"Saved {ARCHS4_OUT}")
    return ARCHS4_OUT


# =========================
# GTEx + ARCHS4 -> pretraining and pre-adaptation h5ad files
# =========================

def merge_and_split_bulk_datasets(gtex_path: Path, archs4_path: Path) -> tuple[Path, Path]:
    print("Merging GTEx + filtered ARCHS4...")
    gtex = ad.read_h5ad(gtex_path)
    archs4 = ad.read_h5ad(archs4_path)

    merged = ad.concat([gtex, archs4], axis=0, join="outer", merge="same", index_unique=None)
    merged.obs_names_make_unique()
    print(f"Merged bulk data: {merged.n_obs} samples x {merged.n_vars} genes")

    if merged.n_obs < PRETRAIN_SAMPLE_COUNT:
        raise ValueError(
            f"Cannot sample {PRETRAIN_SAMPLE_COUNT} pretraining samples from only "
            f"{merged.n_obs} merged bulk samples."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(merged.n_obs)
    pretrain_idx = np.sort(shuffled[:PRETRAIN_SAMPLE_COUNT])
    preadapt_idx = np.sort(shuffled[PRETRAIN_SAMPLE_COUNT:])

    pretrain = merged[pretrain_idx].copy()
    preadapt = merged[preadapt_idx].copy()
    pretrain.obs["bulk_split"] = "pretrain"
    preadapt.obs["bulk_split"] = "preadapt"

    pretrain.write(PRETRAIN_OUT)
    preadapt.write(PREADAPT_OUT)

    print(f"Pretraining bulk samples: {pretrain.n_obs}; saved {PRETRAIN_OUT}")
    print(f"Pre-adaptation bulk samples: {preadapt.n_obs}; saved {PREADAPT_OUT}")

    del gtex, archs4, merged, pretrain, preadapt
    gc.collect()

    return PRETRAIN_OUT, PREADAPT_OUT


# =========================
# Main
# =========================

def main():
    gene_list = read_gene_list(GENE_LIST_PATH)
    print(f"Gene list length: {len(gene_list)}")

    gtex_path, gtex_donors = preprocess_gtex(gene_list)
    gtex_donor_hits = find_archs4_gtex_donor_hits(gtex_donors)
    downstream_id_tokens = build_downstream_id_tokens()
    downstream_hits = find_archs4_downstream_hits(downstream_id_tokens)
    archs4_chunks = preprocess_archs4_to_chunks(
        gene_list,
        gtex_donor_hits,
        downstream_hits,
        chunk_size=ARCHS4_CHUNK_SIZE,
    )
    archs4_path = merge_archs4_chunks(archs4_chunks)
    merge_and_split_bulk_datasets(gtex_path, archs4_path)


if __name__ == "__main__":
    main()
