import inspect
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ..base_logger import logger
from ..constants import Labels


ESM_MODEL = "esm2_t30_150M_UR50D"
PREPROCESSING = "quantile_clipping_log1p_standardization_gaussian_blur"
TOKEN_PATCH_SIZE_PX = 8
SEGMENT_LARGE_TISSUE_THRESHOLD = 1500
SEGMENT_LARGE_DETECTION_SIZE = 20

SUPPORTED_MODELS = {
    "virtues-sp32": {
        "repo_id": "bunnelab/virtues",
        "backbone_checkpoint": "virtues-sp32/model.safetensors",
        "segmentation_checkpoint": "virtues-sp32/segmentation-head.safetensors",
    }
}

VIRTUES_CELL_TYPE_MAPPING = {
    0: "None",
    1: "B cell",
    2: "CD4 T cell",
    3: "CD8 T cell",
    4: "Fibroblasts /Stroma",
    5: "Myeloid",
    6: "Tumor",
    7: "Unknown",
    8: "Vessel / Endo.",
}


def _add_virtues_path(virtues_path=None):
    if virtues_path is None:
        return

    virtues_path = str(Path(virtues_path).expanduser())
    if virtues_path not in sys.path:
        sys.path.insert(0, virtues_path)


def _optional_import_error(name: str, exc: ImportError) -> ImportError:
    error = ImportError(
        f"sp.tl.virtues(...) requires optional VirTues dependency {name!r}. "
        f"Original import error: {exc}"
    )
    error.__cause__ = exc
    return error


def _load_base_config(virtues_path=None) -> dict:
    import importlib.util
    import yaml

    if virtues_path is not None:
        virtues_package_root = Path(virtues_path)
    else:
        spec = importlib.util.find_spec("virtues")
        if spec is None or spec.submodule_search_locations is None:
            raise ImportError(
                "sp.tl.virtues(...) requires an importable local VirTues package."
            )
        virtues_package_root = Path(list(spec.submodule_search_locations)[0]).parent
    config_path = virtues_package_root / "configs" / "base_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_and_normalize_protein_mapping(
    channels: list[str],
    protein_mapping: pd.DataFrame,
    marker_col: str,
    protein_id_col: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Validate and normalize the marker-to-protein mapping for VirTues.

    Parameters
    ----------
    channels : list[str]
        Image channel names in their original order.
    protein_mapping : pd.DataFrame
        A dataframe with one row per marker to use for VirTues.
    marker_col : str
        The column in ``protein_mapping`` containing marker names.
    protein_id_col : str
        The column in ``protein_mapping`` containing UniProt IDs.
    model_name : str
        The VirTues model name.

    Returns
    -------
    pd.DataFrame
        A normalized mapping table with columns ``marker_name``, ``protein_id``, ``channel_index``,
        and ``marker_embedding_index``.

    Raises
    ------
    ValueError
        If the model name is unsupported, required columns are missing, protein IDs are empty,
        marker names are duplicated, or no mapped markers are present in the image channels.
    TypeError
        If ``protein_mapping`` is not a pandas DataFrame.

    Notes
    -----
    UniProt IDs are never inferred from marker names. Markers that are present in ``protein_mapping``
    but absent from the image channels are dropped with a warning.
    Marker names and protein IDs are converted to strings and stripped of surrounding whitespace.
    """

    # check model name, protein mapping type, required columns, non-empty protein IDs, unique marker names, presence of markers in the image channels
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported VirTues model_name {model_name!r}. Supported models: {list(SUPPORTED_MODELS)}.")

    if protein_mapping is None:
        raise ValueError("protein_mapping must be supplied. UniProt IDs are not inferred from marker names.")
    elif not isinstance(protein_mapping, pd.DataFrame):
        raise TypeError("protein_mapping must be a pandas DataFrame.")

    missing = [col for col in [marker_col, protein_id_col] if col not in protein_mapping.columns]

    if missing:
        raise ValueError(f"protein_mapping must contain columns {marker_col!r} and {protein_id_col!r}.")

    mapping = protein_mapping.loc[:, [marker_col, protein_id_col]].copy()
    mapping.columns = ["marker_name", "protein_id"]

    if mapping["protein_id"].isna().any():
        raise ValueError("Protein IDs must be non-empty.")

    mapping["marker_name"] = mapping["marker_name"].astype(str).str.strip() # strip whitespace from marker names
    mapping["protein_id"] = mapping["protein_id"].astype(str).str.strip() # strip whitespace from protein IDs

    if (mapping["protein_id"] == "").any():
        raise ValueError("Protein IDs must be non-empty.")
    
    if mapping["marker_name"].duplicated().any():
        duplicated = mapping.loc[mapping["marker_name"].duplicated(), "marker_name"].tolist()
        raise ValueError(f"Marker names must be unique: {duplicated}.")

    missing_markers = [marker for marker in mapping["marker_name"] if marker not in channels]
    if missing_markers: # intersect to markers that are both in the channels and in the mapping
        logger.warning(f"Dropping markers from protein_mapping that are not present in the image channels: {missing_markers}.")
        mapping = mapping[mapping["marker_name"].isin(channels)].copy()

    if len(mapping) == 0:
        raise ValueError("No protein_mapping markers are present in the image channels.")

    mapping["channel_index"] = [channels.index(marker) for marker in mapping["marker_name"]]
    mapping["marker_embedding_index"] = np.arange(len(mapping), dtype=int)
    mapping.attrs.update(
        {
            "model_name": model_name,
            "esm_model": ESM_MODEL,
            "model_marker_order": mapping["marker_name"].tolist(),
            "model_protein_id_order": mapping["protein_id"].tolist(),
        }
    )

    logger.info(f"Using {len(mapping)} markers for VirTues.")
    if len(mapping) < len(channels):
        logger.warning(
            "VirTues protein_mapping contains only a subset of image channels; segmentation quality may be worse "
            "than with the full intended marker panel."
        )
    return mapping


def _download_uniprot_fasta(protein_id: str) -> str:
    try:
        import requests
    except ImportError as exc:
        raise _optional_import_error("requests", exc)
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta", timeout=60)
    if response.status_code != 200:
        raise ValueError(f"Could not download FASTA sequence for UniProt ID {protein_id!r}.")
    sequence = "".join(line.strip() for line in response.text.splitlines() if line and not line.startswith(">"))
    if sequence == "":
        raise ValueError(f"Downloaded FASTA sequence for UniProt ID {protein_id!r} was empty.")
    return sequence


def _compute_virtues_marker_embeddings(protein_ids: list[str], device: str = "cuda") -> np.ndarray:
    """Compute ESM priors in the supplied protein order, returning float32 CPU arrays."""
    try:
        import torch
    except ImportError as exc:
        raise _optional_import_error("torch", exc)

    try:
        from virtues.utils.compute_esm_embeddings import ESMWrapper
    except ImportError as exc:
        raise _optional_import_error("virtues.utils.compute_esm_embeddings", exc)

    esm_model = ESMWrapper(model_name=ESM_MODEL, device=device)
    embeddings = []
    for protein_id in protein_ids:
        sequence = _download_uniprot_fasta(protein_id)
        with torch.no_grad():
            embedding = esm_model.embed_sequence(sequence).detach().cpu().numpy()
        embeddings.append(embedding.astype(np.float32))
    embeddings = np.stack(embeddings, axis=0)

    return embeddings


def _resolve_mean_std(mean: Optional[pd.Series], 
                      std: Optional[pd.Series], 
                      markers: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    if (mean is None) != (std is None):
        raise ValueError("mean and std must either both be supplied or both be None.")
    if mean is None:
        return None, None, "image"

    if not isinstance(mean, pd.Series) or not isinstance(std, pd.Series):
        raise TypeError("mean and std must be pandas Series indexed by marker name.")

    missing_mean = [marker for marker in markers if marker not in mean.index]
    missing_std = [marker for marker in markers if marker not in std.index]
    if missing_mean:
        raise ValueError(f"mean is missing markers: {missing_mean}.")
    if missing_std:
        raise ValueError(f"std is missing markers: {missing_std}.")

    mean_series = mean.loc[markers].astype(float)
    std_series = std.loc[markers].astype(float)
    return mean_series.to_numpy(), std_series.to_numpy(), "dataset"


def preprocess_for_virtues(
    image: np.ndarray,
    mapping: pd.DataFrame,
    quantile: float = 0.99,
    mean: Optional[pd.Series] = None,
    std: Optional[pd.Series] = None,
    gaussian_blur: bool = True,
    gaussian_blur_sigma: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Preprocess image channels for VirTues model input.

    Parameters
    ----------
    image : np.ndarray
        Image with shape [mapped_channels, y, x], already selected in mapping order.
    mapping : pd.DataFrame
        The normalized marker-to-protein mapping defining marker order.
    quantile : float, optional
        The per-channel image quantile used for clipping.
    mean : pd.Series, optional
        Dataset-level per-marker means, indexed by marker name.
    std : pd.Series, optional
        Dataset-level per-marker standard deviations, indexed by marker name.
    gaussian_blur : bool, optional
        If True, apply a Gaussian blur after standardization.
    gaussian_blur_sigma : float, optional
        The Gaussian blur sigma.

    Returns
    -------
    tuple[np.ndarray, dict]
        The preprocessed image with shape ``[channels, y, x]`` and a dictionary with preprocessing metadata.

    Notes
    -----
    The selected image is clipped to ``[0, q]``, transformed with ``log1p``,
    standardized, and optionally blurred. If ``mean`` and ``std`` are not supplied, image-level
    values are used with a warning. Negative values are set to zero by the clipping step.
    """
    if not np.isfinite(quantile) or quantile < 0 or quantile > 1:
        raise ValueError("quantile must be a finite value between 0 and 1.")

    markers = mapping["marker_name"].tolist()
    protein_ids = mapping["protein_id"].tolist()

    if image.ndim != 3 or image.shape[0] != len(mapping):
        raise ValueError("Image must have shape [mapped_channels, y, x] matching protein_mapping rows.")
    image = image.astype(np.float32, copy=True)

    # clip to specified quantile
    quantiles = np.quantile(image, quantile, axis=(1, 2)).astype(np.float32)
    if np.any(~np.isfinite(quantiles)) or np.any(quantiles < 0):
        raise ValueError("Computed image-level VirTues quantiles must be finite and non-negative.")
    for idx, q in enumerate(quantiles):
        image[idx] = np.clip(image[idx], 0, q)

    # log1p transform
    image = np.log1p(image)

    # get and validate mean/std, using either dataset-level or image-level
    mean_values, std_values, mean_std_level = _resolve_mean_std(mean, std, markers)
    if mean_values is None:
        logger.warning(
            "No dataset-level VirTues mean/std supplied. Falling back to image-level mean/std; results may differ "
            "from the VirTues paper preprocessing."
        )
        mean_values = image.mean(axis=(1, 2))
        std_values = image.std(axis=(1, 2))

    if np.any(~np.isfinite(std_values)) or np.any(std_values <= 0):
        raise ValueError("VirTues standard deviations must be finite and > 0.")
    if np.any(~np.isfinite(mean_values)):
        raise ValueError("VirTues means must be finite.")

    # standardize the image
    image = (image - mean_values[:, np.newaxis, np.newaxis]) / std_values[:, np.newaxis, np.newaxis]
    image = image.astype(np.float32)

    # apply Gaussian blur if requested
    if gaussian_blur:
        for idx in range(image.shape[0]):
            image[idx] = gaussian_filter(image[idx], sigma=gaussian_blur_sigma, truncate=1.0)

    attrs = {
        "preprocessing": PREPROCESSING,
        "quantile": quantile,
        "quantile_level": "image",
        "mean_std_level": mean_std_level,
        "gaussian_blur": bool(gaussian_blur),
        "gaussian_blur_sigma": gaussian_blur_sigma,
        "markers_used": markers,
        "protein_ids_used": protein_ids,
    }
    return image, attrs


def load_virtues_model(marker_embeddings, model_name: str, cache_dir=None, virtues_path=None, device: str = "cuda", load_segmentation_head: bool = True):
    """Load the backbone from NumPy priors and optionally load its segmentation head.

    Returns (backbone, head), with head=None when segmentation components are not requested.
    """
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from virtues.modules.multiplex_virtues import MultiplexVirtues
    except ImportError as exc:
        raise _optional_import_error("torch/huggingface_hub/safetensors/virtues", exc)

    registry = SUPPORTED_MODELS[model_name]
    backbone_path = hf_hub_download(
        repo_id=registry["repo_id"],
        filename=registry["backbone_checkpoint"],
        cache_dir=cache_dir,
    )
    marker_embeddings = torch.as_tensor(marker_embeddings, dtype=torch.float32, device=device)
    conf = _load_base_config(virtues_path=virtues_path)["model"]
    model = MultiplexVirtues(
        use_default_config=False,
        custom_config=None,
        prior_bias_embeddings=marker_embeddings,
        prior_bias_embedding_type="esm",
        prior_bias_embedding_fusion_type="add",
        patch_size=conf["patch_size"],
        model_dim=conf["model_dim"],
        feedforward_dim=conf["feedforward_dim"],
        encoder_pattern=conf["encoder_pattern"],
        num_encoder_heads=conf["num_encoder_heads"],
        decoder_pattern=conf["decoder_pattern"],
        num_decoder_heads=conf["num_decoder_heads"],
        num_hidden_layers=conf["num_decoder_hidden_layers"],
        positional_embedding_type=conf["positional_embedding_type"],
        dropout=conf["dropout"],
        group_layers=conf["group_layers"],
        norm_after_encoder_decoder=conf["norm_after_encoder_decoder"],
        verbose=False,
    )
    model.load_state_dict(load_file(backbone_path, device="cpu"))
    model = model.to(device).eval()

    segmentation_model = None
    if load_segmentation_head:
        try:
            from virtues.modules.segmentation.unet import VirtuesSegmentationHead
        except ImportError as exc:
            raise _optional_import_error("virtues.modules.segmentation.unet", exc)

        segmentation_path = hf_hub_download(
            repo_id=registry["repo_id"],
            filename=registry["segmentation_checkpoint"],
            cache_dir=cache_dir,
        )
        segmentation_model = VirtuesSegmentationHead(
            virtues_model=model, dim_out=5, num_celltypes=9
        )
        segmentation_model.load_state_dict(
            load_file(segmentation_path, device=device), strict=False
        )
        segmentation_model = segmentation_model.to(device).eval()
    return model, segmentation_model


def _virtues_aggregate_patch_embeddings(
    patch_embeddings: np.ndarray,
    patch_coords_yx: np.ndarray,
    segmentation: np.ndarray,
    cell_ids: np.ndarray,
    tile_size_px: int,
) -> np.ndarray:
    """Aggregate crop-local tokens by pixel overlap, preserving the supplied cell order."""
    if patch_embeddings.ndim != 4 or patch_coords_yx.shape != (patch_embeddings.shape[0], 2):
        raise ValueError("Patch embeddings and coordinates must have shapes [crop, token_y, token_x, embedding] and [crop, 2].")
    if not np.issubdtype(patch_coords_yx.dtype, np.number) or not np.isfinite(patch_coords_yx).all():
        raise ValueError("Crop coordinates must be finite numeric top-left (y, x) positions.")
    if np.any(patch_coords_yx < 0) or not np.array_equal(patch_coords_yx, np.rint(patch_coords_yx)):
        raise ValueError("Crop coordinates must be non-negative integers.")
    if segmentation.ndim != 2 or cell_ids.ndim != 1:
        raise ValueError("Segmentation must be two-dimensional and cell_ids one-dimensional.")
    if not isinstance(tile_size_px, (int, np.integer)) or tile_size_px <= 0 or tile_size_px % TOKEN_PATCH_SIZE_PX:
        raise ValueError("tile_size_px must be a positive multiple of the fixed 8-pixel token size.")
    if patch_embeddings.shape[1:3] != (tile_size_px // TOKEN_PATCH_SIZE_PX, tile_size_px // TOKEN_PATCH_SIZE_PX):
        raise ValueError("Patch token grid does not match tile_size_px and the fixed 8-pixel token size.")

    patch_embeddings = patch_embeddings.astype(np.float32)
    patch_coords = patch_coords_yx
    tile_size = tile_size_px
    token_patch_size = token_stride = TOKEN_PATCH_SIZE_PX
    embedding_dim = patch_embeddings.shape[-1]

    cell_tokens = {}
    weights = {}
    for crop_tokens, (crop_y, crop_x) in zip(patch_embeddings, patch_coords):
        crop_y = int(crop_y)
        crop_x = int(crop_x)
        crop_mask = segmentation[
            crop_y : crop_y + tile_size,
            crop_x : crop_x + tile_size,
        ]

        for i in range(crop_tokens.shape[0]):
            for j in range(crop_tokens.shape[1]):
                patch_mask = crop_mask[
                    i * token_stride : i * token_stride + token_patch_size,
                    j * token_stride : j * token_stride + token_patch_size,
                ]
                if patch_mask.size == 0:
                    continue
                unique, counts = np.unique(patch_mask, return_counts=True)
                for cell_id, overlap_pixels in zip(unique, counts):
                    if cell_id == 0:
                        continue
                    if cell_id not in weights:
                        weights[cell_id] = []
                    weights[cell_id].append(overlap_pixels)
                    if cell_id not in cell_tokens:
                        cell_tokens[cell_id] = []
                    cell_tokens[cell_id].append(crop_tokens[i, j, :])

    cells = cell_ids
    values = np.full((len(cells), embedding_dim), np.nan, dtype=np.float32)
    for row, cell_id in enumerate(cells):
        if cell_id in cell_tokens:
            tokens = np.stack(cell_tokens[cell_id])
            weights_array = np.asarray(weights[cell_id], dtype=tokens.dtype)
            weights_array = weights_array / weights_array.sum()
            values[row] = np.sum(tokens * weights_array[:, None], axis=0).astype(
                np.float32
            )

    return values


def _project_virtues_cell_types_to_segmentation(
    cell_type_prediction: np.ndarray,
    segmentation: np.ndarray,
    cell_ids: np.ndarray,
) -> np.ndarray:
    """Assign pixel-wise class predictions to cells by majority vote, in the supplied cell-ID order.

    All supplied class IDs participate, including 0 if present; ties choose the lowest class ID.
    """
    if segmentation.ndim != 2 or cell_type_prediction.shape != segmentation.shape or cell_ids.ndim != 1:
        raise ValueError("Cell type predictions and segmentation must have matching two-dimensional shapes and cell_ids must be one-dimensional.")
    cell_types = []
    for cell_id in cell_ids:
        mask = segmentation == cell_id
        if np.sum(mask) == 0:
            raise ValueError(f"Cell {cell_id!r} is absent from the segmentation.")
        cell_type = np.bincount(cell_type_prediction[mask]).argmax()
        cell_types.append(VIRTUES_CELL_TYPE_MAPPING.get(cell_type, Labels.UNLABELED))

    return np.asarray(cell_types)


def _run_virtues_segmentation(
    image: np.ndarray,
    segmentation_model,
    tile_size: int,
    segment_overlap: int,
    batch_size: int,
    return_cell_type_prediction: bool = False,
    return_patch_embeddings: bool = False,
    device: str = "cuda",
) -> dict:
    """Run segmentation on an image array and return requested CPU arrays and computational metadata.

    Optional predictions and patch outputs are None when not requested. No backend data is accessed or stored.
    """
    if image.ndim != 3:
        raise ValueError("Image must have shape [channels, y, x].")
    try:
        import torch
    except ImportError as exc:
        raise _optional_import_error("torch", exc)

    kwargs = {}
    if return_patch_embeddings:
        signature = inspect.signature(segmentation_model.segment_tissue)
        if "return_patch_embeddings" not in signature.parameters:
            raise RuntimeError(
                "The imported VirTues segment_tissue does not support return_patch_embeddings. "
                "Install the adapted local VirTues checkout or set return_patch_embeddings=False."
            )
        kwargs["return_patch_embeddings"] = True

    # on `torch.arange` (vs. `midxs`): The wrapper builds a compact marker embedding matrix 
    # in mapping order, so channel ids are dense row indices into that compact matrix.
    with torch.inference_mode():
        output = segmentation_model.segment_tissue(
            torch.as_tensor(image, dtype=torch.float32, device=device),
            torch.arange(image.shape[0], dtype=torch.long, device=device),
            tile_size=tile_size,
            overlap=segment_overlap,
            batch_size=batch_size,
            **kwargs,
        )
    if return_patch_embeddings:
        segmentation, semantic_logits, patch_embeddings, patch_coords_yx = output
    else:
        segmentation, semantic_logits = output

    segmentation = segmentation.cpu().numpy().astype(np.int32)

    cell_type_prediction = None
    if return_cell_type_prediction:
        #-------- suppress cell types None and Unknown, following the VirTues tutorial. 
        reverse_cell_type_mapping = {v: k for k, v in VIRTUES_CELL_TYPE_MAPPING.items()}
        suppressed_cell_type_ids = [
            reverse_cell_type_mapping[cell_type]
            for cell_type in ["None", "Unknown"]
        ]
        semantic_logits = semantic_logits.cpu().clone()
        semantic_logits[suppressed_cell_type_ids, :, :] = float("-inf")
        #-------- 
        cell_type_prediction = semantic_logits.argmax(0).numpy().astype(np.int16)

    attrs = {
        "source": "segment_tissue",
        "tile_size_px": tile_size,
        "segment_overlap_px": segment_overlap,
        "batch_size": batch_size,
    }
    if return_cell_type_prediction:
        attrs["cell_type_suppressed_classes"] = ["None", "Unknown"]
    patch_attrs = None
    if return_patch_embeddings:
        # segment patch geometry
        if max(image.shape[-2:]) > SEGMENT_LARGE_TISSUE_THRESHOLD:
            crop_overlap = 2 * (segment_overlap + SEGMENT_LARGE_DETECTION_SIZE)
        else:
            crop_overlap = 2 * segment_overlap
        crop_stride = max(tile_size - crop_overlap, 0)

        patch_attrs = {
            "source": "segment_tissue",
            "tile_size_px": tile_size,
            "crop_stride_px": crop_stride,
            "crop_overlap_px": crop_overlap,
            "token_patch_size_px": TOKEN_PATCH_SIZE_PX,
            "token_stride_px": TOKEN_PATCH_SIZE_PX,
        }
        patch_embeddings = patch_embeddings.cpu().numpy()
        patch_coords_yx = patch_coords_yx.cpu().numpy()
    return {
        "segmentation": segmentation,
        "cell_type_prediction": cell_type_prediction,
        "patch_embeddings": patch_embeddings if return_patch_embeddings else None,
        "patch_coords_yx": patch_coords_yx if return_patch_embeddings else None,
        "segmentation_metadata": attrs,
        "patch_metadata": patch_attrs,
    }


def _stack_crop_tokens(crop_tokens, indices):
    if len(crop_tokens) != len(indices):
        raise ValueError("compute_cell_tokens returned different numbers of crop tokens and crop coordinates.")
    if len(crop_tokens) == 0:
        raise ValueError("compute_cell_tokens returned no crop tokens.")
    for idx in indices:
        if len(idx) != 2:
            raise ValueError("compute_cell_tokens indices must be top-left (y, x) coordinate pairs.")
    ordered = sorted(
        zip(indices, crop_tokens), key=lambda item: (int(item[0][0]), int(item[0][1]))
    )
    tokens = [
        token.detach().cpu().numpy() if hasattr(token, "detach") else np.asarray(token)
        for _, token in ordered
    ]
    coords = np.asarray(
        [[int(idx[0]), int(idx[1])] for idx, _ in ordered], dtype=np.int32
    )
    if np.any(coords < 0):
        raise ValueError("compute_cell_tokens indices must be non-negative.")
    if any(token.ndim != 3 for token in tokens):
        raise ValueError("compute_cell_tokens crop_tokens must be [token_y, token_x, embedding] arrays.")
    return np.stack(tokens, axis=0), coords


def _run_virtues_cell_embedding(
    image: np.ndarray,
    segmentation: np.ndarray,
    virtues_model,
    tile_size: int,
    cell_token_stride: int,
    batch_size: int,
    return_patch_embeddings: bool = False,
    device: str = "cuda",
) -> dict:
    """Extract cell embeddings from arrays; optionally return crop tokens without storing outputs.

    Upstream compute_cell_tokens still computes crop tokens internally when they are not returned here.
    """
    if image.ndim != 3 or segmentation.ndim != 2 or image.shape[-2:] != segmentation.shape:
        raise ValueError("Image [channels, y, x] and segmentation [y, x] must have matching spatial shapes.")
    if image.shape[1] < tile_size or image.shape[2] < tile_size:
        raise ValueError("VirTues compute_cell_tokens is not well-defined for images smaller than tile_size in v0.1.")

    try:
        from virtues.utils.cell_tokens import compute_cell_tokens
    except ImportError as exc:
        raise _optional_import_error("virtues.utils.cell_tokens", exc)

    try:
        import torch
    except ImportError as exc:
        raise _optional_import_error("torch", exc)

    segmentation = torch.as_tensor(segmentation, device="cpu")
    cell_ids, cell_tokens, crop_tokens, indices = compute_cell_tokens(
        virtues_model,
        torch.as_tensor(image, dtype=torch.float32, device=device),
        torch.arange(image.shape[0], dtype=torch.long, device=device),
        segmentation,
        device=device,
        tile_size=tile_size,
        patch_size=TOKEN_PATCH_SIZE_PX,
        stride=cell_token_stride,
        chunk_size=batch_size,
    )
    if len(cell_ids) != cell_tokens.shape[0]:
        raise ValueError("compute_cell_tokens returned cell_ids and cell_tokens with incompatible lengths.")
    attrs = {
        "source": "compute_cell_tokens",
        "tile_size_px": tile_size,
        "cell_token_stride_px": cell_token_stride,
        "token_patch_size_px": TOKEN_PATCH_SIZE_PX,
    }
    patch_embeddings = patch_coords = patch_attrs = None
    if return_patch_embeddings:
        patch_embeddings, patch_coords = _stack_crop_tokens(crop_tokens, indices)
        patch_attrs = {
            "source": "compute_cell_tokens",
            "tile_size_px": tile_size,
            "crop_stride_px": cell_token_stride,
            "crop_overlap_px": max(tile_size - cell_token_stride, 0),
            "token_patch_size_px": TOKEN_PATCH_SIZE_PX,
            "token_stride_px": TOKEN_PATCH_SIZE_PX,
        }
    return {
        "cell_ids": cell_ids.cpu().numpy(),
        "cell_embeddings": cell_tokens.cpu().numpy().astype(np.float32),
        "patch_embeddings": patch_embeddings,
        "patch_coords_yx": patch_coords,
        "cell_embedding_metadata": attrs,
        "patch_metadata": patch_attrs,
    }


def _virtues(
    image: np.ndarray,
    mapping: pd.DataFrame,
    mode: str = "segment",
    segmentation: Optional[np.ndarray] = None,
    marker_embeddings: Optional[np.ndarray] = None,
    model_name: str = "virtues-sp32",
    cache_dir=None,
    virtues_path=None,
    quantile: float = 0.99,
    mean: Optional[pd.Series] = None,
    std: Optional[pd.Series] = None,
    gaussian_blur: bool = True,
    gaussian_blur_sigma: float = 1.0,
    tile_size: int = 128,
    segment_overlap: int = 32,
    cell_token_stride: int = 42,
    batch_size=None,
    return_cell_type_prediction: bool = False,
    return_patch_embeddings: bool = False,
    device: str = "cuda",
) -> dict:
    """Run the shared VirTues workflow on selected arrays and return outputs plus newly computed priors.

    Image channels and supplied priors must already match the normalized mapping order. The outputs
    dictionary contains the selected runner's arrays and model/preprocessing metadata. Newly computed
    priors are returned separately for backend storage, or None when existing priors were supplied.
    """
    if mode not in {"segment", "embed_cells"}:
        raise ValueError("mode must be either 'segment' or 'embed_cells'.")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported VirTues model_name {model_name!r}.")
    if device != "cuda":
        raise ValueError("Only device='cuda' is supported for sp.tl.virtues(...) in v0.1.")
    if mode == "embed_cells" and segmentation is None:
        raise ValueError("mode='embed_cells' requires a segmentation array.")
    _add_virtues_path(virtues_path)
    try:
        import torch
    except ImportError as exc:
        raise _optional_import_error("torch", exc)
    if not torch.cuda.is_available():
        raise RuntimeError("sp.tl.virtues(...) requires CUDA for v0.1 model execution.")
    if batch_size is None:
        batch_size = 4 if mode == "segment" else 32

    image, preprocessing_attrs = preprocess_for_virtues(
        image, 
        mapping, 
        quantile=quantile, 
        mean=mean, 
        std=std,
        gaussian_blur=gaussian_blur, 
        gaussian_blur_sigma=gaussian_blur_sigma,
    )
    new_marker_embeddings = None
    if marker_embeddings is None:
        marker_embeddings = _compute_virtues_marker_embeddings(mapping["protein_id"].tolist(), device=device)
        new_marker_embeddings = marker_embeddings
    virtues_model, segmentation_model = load_virtues_model(
        marker_embeddings=marker_embeddings, 
        model_name=model_name, 
        cache_dir=cache_dir,
        virtues_path=virtues_path, 
        device=device, 
        load_segmentation_head=(mode == "segment"),
    )
    if mode == "segment":
        outputs = _run_virtues_segmentation(
            image, 
            segmentation_model, 
            tile_size=tile_size, 
            segment_overlap=segment_overlap,
            batch_size=batch_size, 
            return_cell_type_prediction=return_cell_type_prediction,
            return_patch_embeddings=return_patch_embeddings, 
            device=device,
        )
    else:
        outputs = _run_virtues_cell_embedding(
            image, 
            segmentation, 
            virtues_model, 
            tile_size=tile_size, 
            cell_token_stride=cell_token_stride,
            batch_size=batch_size, 
            return_patch_embeddings=return_patch_embeddings, 
            device=device,
        )
    outputs = dict(outputs)
    for key in ["segmentation_metadata", "cell_embedding_metadata", "patch_metadata"]:
        if outputs.get(key) is not None:
            attrs = dict(outputs[key])
            attrs.update({"model_name": model_name, **preprocessing_attrs})
            outputs[key] = attrs
    return {"outputs": outputs, "new_marker_embeddings": new_marker_embeddings}
