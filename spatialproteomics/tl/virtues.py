import inspect
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from ..base_logger import logger
from ..constants import Dims, Labels, Layers


ESM_MODEL = "esm2_t30_150M_UR50D"
PREPROCESSING = "quantile_clipping_log1p_standardization_gaussian_blur"
MARKER_METADATA_DIM = "virtues_marker_metadata"
VIRTUES_MARKERS_DIM = "virtues_markers"
VIRTUES_MARKER_EMBEDDING_DIM = "virtues_marker_embedding"
VIRTUES_PATCHES_DIM = "virtues_patches"
TOKEN_Y_DIM = "token_y"
TOKEN_X_DIM = "token_x"
VIRTUES_EMBEDDING_DIM = "virtues_embedding"
YX_DIM = "yx"
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


def _check_key_available(obj: xr.Dataset, key: str, overwrite: bool) -> None:
    if key in obj and not overwrite:
        raise ValueError(f"Layer {key} already exists. Use overwrite=True or choose another key.")


def _drop_if_overwrite(obj: xr.Dataset, key: str, overwrite: bool) -> xr.Dataset:
    _check_key_available(obj, key, overwrite)
    if key in obj and overwrite:
        return obj.drop_vars(key)
    return obj


def validate_and_normalize_protein_mapping(
    obj: xr.Dataset,
    protein_mapping: pd.DataFrame,
    marker_col: str,
    protein_id_col: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Validate and normalize the marker-to-protein mapping for VirTues.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object containing the image channel coordinates.
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
        marker names are duplicated, or no mapped markers are present in the object.
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

    channels = obj.coords[Dims.CHANNELS].values.astype(str).tolist()
    missing_markers = [marker for marker in mapping["marker_name"] if marker not in channels]
    if missing_markers: # intersect to markers that are both in the channels and in the mapping
        logger.warning(f"Dropping markers from protein_mapping that are not present in obj.coords['channels']: {missing_markers}.")
        mapping = mapping[mapping["marker_name"].isin(channels)].copy()

    if len(mapping) == 0:
        raise ValueError("No protein_mapping markers are present in obj.coords['channels'].")

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


def _marker_embeddings_from_layer(obj: xr.Dataset, key: str, mapping: pd.DataFrame):
    values = obj[key]
    if values.dims[0] == Dims.CHANNELS:
        return values.sel(
            {Dims.CHANNELS: mapping["marker_name"].tolist()}
        ).values.astype(np.float32)
    if values.dims[0] == VIRTUES_MARKERS_DIM:
        return values.sel(
            {VIRTUES_MARKERS_DIM: mapping["marker_name"].tolist()}
        ).values.astype(np.float32)
    raise ValueError(f"Stored marker embeddings {key!r} use unsupported first dimension {values.dims[0]!r}.")


def add_marker_embeddings(
    obj: xr.Dataset,
    embeddings: np.ndarray,
    mapping: pd.DataFrame,
    key_added: str,
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Add VirTues marker embeddings and their metadata to the object.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    embeddings : np.ndarray
        Marker embeddings in the order of the normalized mapping table.
    mapping : pd.DataFrame
        The normalized marker-to-protein mapping.
    key_added : str
        The key under which marker embeddings will be stored.
    overwrite : bool, optional
        If True, overwrite existing marker embedding and metadata layers.

    Returns
    -------
    xr.Dataset
        The updated spatialproteomics object.

    Notes
    -----
    If all image channels are represented, embeddings are stored on the native ``channels`` dimension
    in image-channel order. If only a subset is represented, embeddings are stored on the
    ``virtues_markers`` dimension in mapping order.
    A companion metadata layer is added under ``{key_added}_metadata``.
    """
    if embeddings.shape[0] != len(mapping):
        raise ValueError("Number of marker embeddings must match protein_mapping rows.")

    metadata_key = f"{key_added}_metadata"
    obj = _drop_if_overwrite(obj, key_added, overwrite)
    obj = _drop_if_overwrite(obj, metadata_key, overwrite)

    channels = obj.coords[Dims.CHANNELS].values.astype(str).tolist()
    full_mapping = set(mapping["marker_name"]) == set(channels) and len(mapping) == len(
        channels
    )
    attrs = {
        "model_name": mapping.attrs.get("model_name", "virtues-sp32"),
        "esm_model": mapping.attrs.get("esm_model", ESM_MODEL),
        "model_marker_order": mapping["marker_name"].tolist(),
        "model_protein_id_order": mapping["protein_id"].tolist(),
    }

    if full_mapping:
        order = [mapping.index[mapping["marker_name"] == channel][0] for channel in channels]
        stored_embeddings = embeddings[order]
        stored_mapping = mapping.loc[order].reset_index(drop=True)
        dim_name = Dims.CHANNELS
        dim_values = obj.coords[Dims.CHANNELS].values
    else:
        stored_embeddings = embeddings
        stored_mapping = mapping.reset_index(drop=True)
        dim_name = VIRTUES_MARKERS_DIM
        dim_values = mapping["marker_name"].to_numpy()

    attrs.update(
        {
            "stored_marker_order": stored_mapping["marker_name"].tolist(),
            "stored_protein_id_order": stored_mapping["protein_id"].tolist(),
        }
    )
    embeddings_array = xr.DataArray(
        stored_embeddings.astype(np.float32),
        dims=[dim_name, VIRTUES_MARKER_EMBEDDING_DIM],
        coords={
            dim_name: dim_values,
            VIRTUES_MARKER_EMBEDDING_DIM: np.arange(stored_embeddings.shape[1]),
        },
        attrs=attrs,
        name=key_added,
    )
    
    fields = ["marker_name", "protein_id", "channel_index", "marker_embedding_index"]
    metadata = stored_mapping.loc[:, fields].astype(str).to_numpy()
    metadata_array = xr.DataArray(
        metadata,
        dims=[dim_name, MARKER_METADATA_DIM],
        coords={dim_name: dim_values, MARKER_METADATA_DIM: fields},
        name=f"{key_added}_metadata",
    )
    return xr.merge([obj, embeddings_array, metadata_array], join="outer", compat="no_conflicts")


def get_or_create_marker_embeddings(
    obj: xr.Dataset,
    mapping: pd.DataFrame,
    marker_embedding_key_added: str,
    overwrite: bool = False,
    device: str = "cuda",
):
    """
    Reuse stored marker embeddings or create them with the VirTues ESM wrapper.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    mapping : pd.DataFrame
        The normalized marker-to-protein mapping.
    marker_embedding_key_added : str
        The key under which marker embeddings are stored or will be added.
    overwrite : bool, optional
        If True, replace existing marker embeddings whose metadata do not match.
    device : str, optional
        The device used for the ESM wrapper and returned torch tensor.

    Returns
    -------
    tuple[xr.Dataset, torch.Tensor]
        The updated spatialproteomics object and a tensor of marker embeddings in mapping order.

    Raises
    ------
    ValueError
        If an existing marker embedding layer has incompatible metadata and ``overwrite`` is False.
    """
    try:
        import torch
    except ImportError as exc:
        raise _optional_import_error("torch", exc)

    if marker_embedding_key_added in obj:
        layer = obj[marker_embedding_key_added]
        metadata_matches = (
            layer.attrs.get("model_marker_order") == mapping["marker_name"].tolist()
            and layer.attrs.get("model_protein_id_order") == mapping["protein_id"].tolist()
            and layer.attrs.get("model_name") == mapping.attrs.get("model_name")
            and layer.attrs.get("esm_model") == mapping.attrs.get("esm_model")
        )
        if metadata_matches:
            embeddings = _marker_embeddings_from_layer(obj, marker_embedding_key_added, mapping)
            return obj, torch.as_tensor(embeddings, dtype=torch.float32, device=device)
        if not overwrite:
            raise ValueError(
                f"Marker embedding layer {marker_embedding_key_added!r} exists but metadata does not match the "
                "requested protein mapping. Use overwrite=True or choose another key."
            )

    try:
        from virtues.utils.compute_esm_embeddings import ESMWrapper
    except ImportError as exc:
        raise _optional_import_error("virtues.utils.compute_esm_embeddings", exc)

    esm_model = ESMWrapper(model_name=ESM_MODEL, device=device)
    embeddings = []
    for protein_id in mapping["protein_id"].tolist():
        sequence = _download_uniprot_fasta(protein_id)
        with torch.no_grad():
            embedding = esm_model.embed_sequence(sequence).detach().cpu().numpy()
        embeddings.append(embedding.astype(np.float32))
    embeddings = np.stack(embeddings, axis=0)

    obj = add_marker_embeddings(obj, embeddings, mapping, marker_embedding_key_added, overwrite=overwrite)
    return obj, torch.as_tensor(embeddings, dtype=torch.float32, device=device)


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
    obj: xr.Dataset,
    mapping: pd.DataFrame,
    image_key: str,
    quantile: float = 0.99,
    mean: Optional[pd.Series] = None,
    std: Optional[pd.Series] = None,
    gaussian_blur: bool = True,
    gaussian_blur_sigma: float = 1.0,
):
    """
    Preprocess image channels for VirTues model input.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    mapping : pd.DataFrame
        The normalized marker-to-protein mapping defining marker order.
    image_key : str
        The image layer to preprocess.
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
    The image is subset to mapped markers, clipped to ``[0, q]``, transformed with ``log1p``,
    standardized, and optionally blurred. If ``mean`` and ``std`` are not supplied, image-level
    values are used with a warning. Negative values are set to zero by the clipping step.
    """
    if image_key not in obj:
        raise KeyError(f"Image layer {image_key!r} not found.")
    if not np.isfinite(quantile) or quantile < 0 or quantile > 1:
        raise ValueError("quantile must be a finite value between 0 and 1.")

    markers = mapping["marker_name"].tolist()
    protein_ids = mapping["protein_id"].tolist()

    # check that dimensions are correct
    image_layer = obj[image_key]
    expected_dims = [Dims.CHANNELS, Dims.Y, Dims.X]
    if list(image_layer.dims) != expected_dims:
        raise ValueError(
            f"Image layer {image_key!r} must have dimensions {expected_dims}; found {list(image_layer.dims)}."
        )

    # subset to markers from the mapping table and convert to numpy array
    image = (obj[image_key].sel({Dims.CHANNELS: markers}).values.astype(np.float32, copy=True))

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
        "image_key": image_key,
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


def load_virtues_model(marker_embeddings, model_name: str, cache_dir=None, virtues_path=None, device: str = "cuda"):
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from virtues.modules.multiplex_virtues import MultiplexVirtues
        from virtues.modules.segmentation.unet import VirtuesSegmentationHead
    except ImportError as exc:
        raise _optional_import_error("huggingface_hub/safetensors/virtues", exc)

    registry = SUPPORTED_MODELS[model_name]
    backbone_path = hf_hub_download(
        repo_id=registry["repo_id"],
        filename=registry["backbone_checkpoint"],
        cache_dir=cache_dir,
    )
    segmentation_path = hf_hub_download(
        repo_id=registry["repo_id"],
        filename=registry["segmentation_checkpoint"],
        cache_dir=cache_dir,
    )
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

    segmentation_model = VirtuesSegmentationHead(
        virtues_model=model, dim_out=5, num_celltypes=9
    )
    segmentation_model.load_state_dict(
        load_file(segmentation_path, device=device), strict=False
    )
    segmentation_model = segmentation_model.to(device).eval()
    return model, segmentation_model


def add_patch_embeddings(
    obj: xr.Dataset,
    patch_embeddings: np.ndarray,
    patch_coords_yx: np.ndarray,
    patch_embedding_key_added: str,
    patch_coord_key_added: str,
    attrs: dict,
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Add VirTues patch embeddings and their crop coordinates to the object.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    patch_embeddings : np.ndarray
        Patch embeddings with shape ``[patch, token_y, token_x, embedding]``.
    patch_coords_yx : np.ndarray
        Top-left crop coordinates with shape ``[patch, 2]`` in ``(y, x)`` order.
    patch_embedding_key_added : str
        The key under which patch embeddings will be stored.
    patch_coord_key_added : str
        The key under which patch coordinates will be stored.
    attrs : dict
        Metadata to store on the patch embedding layer.
    overwrite : bool, optional
        If True, overwrite existing patch embedding and coordinate layers.

    Returns
    -------
    xr.Dataset
        The updated spatialproteomics object.

    Notes
    -----
    Patch embeddings are stored by crop and token position. Coordinates are not stitched into image
    ``x``/``y`` space. Embeddings are stored as ``float16``; coordinates are stored as integer
    top-left ``(y, x)`` positions and include attrs pointing back to the paired embedding layer.
    """
    if patch_embeddings.ndim != 4:
        raise ValueError("patch_embeddings must have shape [patch, token_y, token_x, embedding].")

    if patch_coords_yx.shape != (patch_embeddings.shape[0], 2):
        raise ValueError("patch_coords_yx must have shape [patch, 2].")
    if not np.issubdtype(patch_coords_yx.dtype, np.number):
        raise ValueError("patch_coords_yx must contain numeric top-left coordinates.")
    if not np.isfinite(patch_coords_yx).all():
        raise ValueError("patch_coords_yx must contain finite top-left coordinates.")
    if np.any(patch_coords_yx < 0):
        raise ValueError("patch_coords_yx must contain non-negative top-left coordinates.")
    if not np.array_equal(patch_coords_yx, np.rint(patch_coords_yx)):
        raise ValueError("patch_coords_yx must contain integer top-left coordinates.")

    if patch_embedding_key_added in obj and not overwrite:
        existing_source = obj[patch_embedding_key_added].attrs.get("source")
        if existing_source is not None and existing_source != attrs.get("source"):
            raise ValueError("Existing patch embedding source differs from requested source.")

    if patch_coord_key_added in obj and not overwrite:
        paired = obj[patch_coord_key_added].attrs.get("paired_embedding_key")
        if paired is not None and paired != patch_embedding_key_added:
            raise ValueError("Existing patch coordinates are paired with a different embedding key.")

    n_patches, token_y, token_x, embedding_dim = patch_embeddings.shape
    coord_values = {
        VIRTUES_PATCHES_DIM: np.arange(n_patches),
        TOKEN_Y_DIM: np.arange(token_y),
        TOKEN_X_DIM: np.arange(token_x),
        VIRTUES_EMBEDDING_DIM: np.arange(embedding_dim),
        YX_DIM: np.asarray(["y", "x"]),
    }

    obj = _drop_if_overwrite(obj, patch_embedding_key_added, overwrite)
    obj = _drop_if_overwrite(obj, patch_coord_key_added, overwrite)
    for coord in [
        VIRTUES_PATCHES_DIM,
        TOKEN_Y_DIM,
        TOKEN_X_DIM,
        VIRTUES_EMBEDDING_DIM,
        YX_DIM,
    ]:
        if coord in obj.coords and not any(
            coord in obj[var].dims for var in obj.data_vars
        ):
            obj = obj.drop_vars(coord)

    coord_attrs = {
        "coordinate_convention": "top-left pixel (y, x)",
        "paired_embedding_key": patch_embedding_key_added,
    }
    patch_da = xr.DataArray(
        patch_embeddings.astype(np.float16),
        dims=[VIRTUES_PATCHES_DIM, TOKEN_Y_DIM, TOKEN_X_DIM, VIRTUES_EMBEDDING_DIM],
        coords={
            VIRTUES_PATCHES_DIM: coord_values[VIRTUES_PATCHES_DIM],
            TOKEN_Y_DIM: coord_values[TOKEN_Y_DIM],
            TOKEN_X_DIM: coord_values[TOKEN_X_DIM],
            VIRTUES_EMBEDDING_DIM: coord_values[VIRTUES_EMBEDDING_DIM],
        },
        attrs=attrs,
        name=patch_embedding_key_added,
    )
    coord_da = xr.DataArray(
        patch_coords_yx.astype(np.int32),
        dims=[VIRTUES_PATCHES_DIM, YX_DIM],
        coords={
            VIRTUES_PATCHES_DIM: coord_values[VIRTUES_PATCHES_DIM],
            YX_DIM: coord_values[YX_DIM],
        },
        attrs=coord_attrs,
        name=patch_coord_key_added,
    )
    return xr.merge([obj, patch_da, coord_da], join="outer", compat="no_conflicts")


def add_virtues_segmentation_outputs(
    obj: xr.Dataset,
    segmentation: np.ndarray,
    cell_type_mask: np.ndarray,
    segmentation_key_added: str,
    cell_type_mask_key_added: str,
    attrs: dict,
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Add VirTues segmentation and cell type mask layers to the object.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    segmentation : np.ndarray
        VirTues instance segmentation with shape ``[y, x]``.
    cell_type_mask : np.ndarray
        VirTues cell type mask with shape ``[y, x]``.
    segmentation_key_added : str
        The key under which the VirTues segmentation will be stored.
    cell_type_mask_key_added : str
        The key under which the VirTues cell type mask will be stored.
    attrs : dict
        Metadata to store on both output layers.
    overwrite : bool, optional
        If True, overwrite existing output layers.

    Returns
    -------
    xr.Dataset
        The updated spatialproteomics object.

    Notes
    -----
    The cell type mask stores the VirTues cell type mapping in its attrs.
    """
    if segmentation.shape != (obj.sizes[Dims.Y], obj.sizes[Dims.X]):
        raise ValueError("VirTues segmentation shape must match image y/x shape.")
    if cell_type_mask.shape != segmentation.shape:
        raise ValueError("VirTues cell type mask shape must match segmentation shape.")

    obj = _drop_if_overwrite(obj, segmentation_key_added, overwrite)
    obj = _drop_if_overwrite(obj, cell_type_mask_key_added, overwrite)

    seg_da = xr.DataArray(
        segmentation.astype(np.int32),
        dims=[Dims.Y, Dims.X],
        coords={Dims.Y: obj.coords[Dims.Y], Dims.X: obj.coords[Dims.X]},
        attrs=attrs,
        name=segmentation_key_added,
    )
    mask_attrs = dict(attrs)
    mask_attrs["cell_type_mapping"] = VIRTUES_CELL_TYPE_MAPPING
    mask_da = xr.DataArray(
        cell_type_mask.astype(np.int16),
        dims=[Dims.Y, Dims.X],
        coords={Dims.Y: obj.coords[Dims.Y], Dims.X: obj.coords[Dims.X]},
        attrs=mask_attrs,
        name=cell_type_mask_key_added,
    )
    return xr.merge([obj, seg_da, mask_da], join="outer", compat="no_conflicts")


def add_virtues_embedding_outputs(
    obj: xr.Dataset,
    cell_ids: np.ndarray,
    cell_embeddings: np.ndarray,
    key_added: str,
    attrs: dict,
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Add VirTues cell embeddings aligned to the object's cells coordinate.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    cell_ids : np.ndarray
        Cell IDs corresponding to rows of ``cell_embeddings``.
    cell_embeddings : np.ndarray
        Cell embeddings with shape ``[cells, embedding]``.
    key_added : str
        The key under which cell embeddings will be stored.
    attrs : dict
        Metadata to store on the embedding layer.
    overwrite : bool, optional
        If True, overwrite an existing embedding layer.

    Returns
    -------
    xr.Dataset
        The updated spatialproteomics object.

    Notes
    -----
    Embeddings are aligned to ``obj.coords['cells']``. Cells without a returned
    VirTues embedding are filled with ``nan``.
    """
    if Dims.CELLS not in obj.coords:
        raise ValueError("Cannot store VirTues cell embeddings without a cells coordinate.")
    if len(cell_ids) != cell_embeddings.shape[0]:
        raise ValueError("cell_ids and cell_embeddings must have the same first dimension.")

    obj = _drop_if_overwrite(obj, key_added, overwrite)
    cells = obj.coords[Dims.CELLS].values
    values = np.full((len(cells), cell_embeddings.shape[1]), np.nan, dtype=np.float32)
    cell_to_pos = {cell: idx for idx, cell in enumerate(cells)}
    for idx, cell_id in enumerate(cell_ids):
        if cell_id in cell_to_pos:
            values[cell_to_pos[cell_id]] = cell_embeddings[idx].astype(np.float32)

    da = xr.DataArray(
        values,
        dims=[Dims.CELLS, VIRTUES_EMBEDDING_DIM],
        coords={
            Dims.CELLS: cells,
            VIRTUES_EMBEDDING_DIM: np.arange(cell_embeddings.shape[1]),
        },
        attrs=attrs,
        name=key_added,
    )
    return xr.merge([obj, da], join="outer", compat="no_conflicts")


def virtues_aggregate_patch_embeddings(
    obj: xr.Dataset,
    patch_embedding_key: str = "_virtues_patch_embeddings",
    patch_coord_key: str = "_virtues_patch_coords",
    segmentation_key: str = Layers.SEGMENTATION,
    key_added: str = "_virtues_cell_embeddings",
    overwrite: bool = False,
    virtues_path: str = None,
) -> xr.Dataset:
    """
    Aggregate stored VirTues patch embeddings into cell embeddings.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object containing patch embeddings, patch coordinates, and a segmentation layer.
    patch_embedding_key : str, optional
        The key of the stored patch embeddings.
    patch_coord_key : str, optional
        The key of the stored patch top-left coordinates.
    segmentation_key : str, optional
        The segmentation layer used to assign tokens to cells.
    key_added : str, optional
        The key under which aggregated cell embeddings will be stored.
    overwrite : bool, optional
        If True, overwrite an existing cell embedding layer.
    virtues_path : str, optional
        Reserved for API compatibility.

    Returns
    -------
    xr.Dataset
        The updated spatialproteomics object.

    Notes
    -----
    Each token contributes to every overlapping cell, weighted by the number of overlapping
    segmentation pixels. This mirrors the cell-token aggregation logic used by VirTues
    as closely as possible.
    """

    for key in [patch_embedding_key, patch_coord_key, segmentation_key]:
        if key not in obj:
            raise KeyError(f"Layer {key!r} not found.")

    _check_key_available(obj, key_added, overwrite)

    patch_da = obj[patch_embedding_key]
    coords_da = obj[patch_coord_key]
    required_attrs = [
        "tile_size_px",
        "crop_stride_px",
        "crop_overlap_px",
        "token_patch_size_px",
        "token_stride_px",
        "source",
    ]
    missing_attrs = [attr for attr in required_attrs if attr not in patch_da.attrs]
    if missing_attrs:
        raise ValueError(
            f"Patch embeddings are missing required attrs: {missing_attrs}."
        )

    patch_embeddings = patch_da.values.astype(np.float32)
    patch_coords = coords_da.values.astype(int)
    segmentation = obj[segmentation_key].values
    tile_size = int(patch_da.attrs["tile_size_px"])
    token_patch_size = int(patch_da.attrs["token_patch_size_px"])
    token_stride = int(patch_da.attrs["token_stride_px"])
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

    cells = (
        obj.coords[Dims.CELLS].values
        if Dims.CELLS in obj.coords
        else np.unique(segmentation[segmentation > 0])
    )
    values = np.full((len(cells), embedding_dim), np.nan, dtype=np.float32)
    for row, cell_id in enumerate(cells):
        if cell_id in cell_tokens:
            tokens = np.stack(cell_tokens[cell_id])
            weights_array = np.asarray(weights[cell_id], dtype=tokens.dtype)
            weights_array = weights_array / weights_array.sum()
            values[row] = np.sum(tokens * weights_array[:, None], axis=0).astype(
                np.float32
            )

    attrs = {
        "source": f"aggregate:{patch_da.attrs['source']}",
        "patch_embedding_key": patch_embedding_key,
        "patch_coord_key": patch_coord_key,
        "segmentation_key": segmentation_key,
    }
    return add_virtues_embedding_outputs(
        obj,
        cell_ids=cells,
        cell_embeddings=values,
        key_added=key_added,
        attrs=attrs,
        overwrite=overwrite,
    )


def project_virtues_cell_types_to_existing_segmentation(
    obj: xr.Dataset,
    cell_type_mask: np.ndarray,
    segmentation_key: str = Layers.SEGMENTATION,
    cell_type_label_key_added: str = "_virtues_cell_type",
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Project VirTues cell type predictions onto an existing segmentation.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    cell_type_mask : np.ndarray
        Pixel-wise VirTues cell type mask.
    segmentation_key : str, optional
        The existing segmentation layer whose cells should receive labels.
    cell_type_label_key_added : str, optional
        The column added to ``_la_layers``.
    overwrite : bool, optional
        If True, overwrite an existing label layer column with the same name.

    Returns
    -------
    xr.Dataset
        The updated spatialproteomics object.

    Notes
    -----
    For each existing cell, the assigned VirTues cell type is the most frequent value of
    ``cell_type_mask`` under that cell's segmentation mask. Ties follow ``np.bincount(...).argmax()``
    and therefore choose the lowest tied integer cell type ID.
    """
    if segmentation_key not in obj:
        return obj
    segmentation = obj[segmentation_key].values

    if cell_type_mask.shape != segmentation.shape:
        raise ValueError("cell_type_mask shape must match existing segmentation shape.")

    if Dims.CELLS not in obj.coords:
        raise ValueError("Cannot project VirTues cell types without a cells coordinate.")

    cell_types = []
    for cell_id in obj.coords[Dims.CELLS].values:
        mask = segmentation == cell_id
        if np.sum(mask) == 0:
            raise ValueError(f"Cell {cell_id!r} is present in coords['cells'] but absent from {segmentation_key!r}.")
        cell_type = np.bincount(cell_type_mask[mask]).argmax()
        cell_types.append(VIRTUES_CELL_TYPE_MAPPING.get(cell_type, Labels.UNLABELED))

    new_df = pd.DataFrame({cell_type_label_key_added: cell_types}, index=obj.coords[Dims.CELLS].values)

    if Layers.LA_LAYERS in obj:
        existing = obj[Layers.LA_LAYERS].to_pandas()
        if cell_type_label_key_added in existing.columns and not overwrite:
            raise ValueError(f"Label layer column {cell_type_label_key_added} already exists.")
        existing[cell_type_label_key_added] = new_df[cell_type_label_key_added]
        df = existing
        obj = obj.drop_vars(Layers.LA_LAYERS)
    else:
        df = new_df

    da = xr.DataArray(
        df.values.astype(object),
        dims=[Dims.CELLS, Dims.LA_FEATURES],
        coords={
            Dims.CELLS: obj.coords[Dims.CELLS].values,
            Dims.LA_FEATURES: df.columns.to_numpy(),
        },
        name=Layers.LA_LAYERS,
    )
    return xr.merge([obj, da], join="outer", compat="no_conflicts")


def run_virtues_segmentation(
    obj: xr.Dataset,
    image: np.ndarray,
    mapping: pd.DataFrame,
    segmentation_model,
    model_name: str,
    preprocessing_attrs: dict,
    tile_size: int,
    segment_overlap: int,
    batch_size: int,
    segmentation_key_used: str,
    segmentation_key_added: str,
    cell_type_mask_key_added: str,
    cell_type_label_key_added: str,
    patch_embedding_key_added: str,
    patch_coord_key_added: str,
    store_patch_embeddings: bool,
    overwrite: bool,
    device: str,
):
    """
    Run VirTues segmentation and store the requested outputs.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    image : np.ndarray
        Preprocessed image with shape ``[channels, y, x]``.
    mapping : pd.DataFrame
        The normalized marker-to-protein mapping.
    segmentation_model
        The VirTues segmentation model.
    model_name : str
        The VirTues model name.
    preprocessing_attrs : dict
        Metadata returned by ``preprocess_for_virtues``.
    tile_size : int
        Tile size passed to VirTues segmentation.
    segment_overlap : int
        Tile overlap passed to VirTues segmentation.
    batch_size : int
        Batch size passed to VirTues segmentation.
    segmentation_key_used : str
        Existing segmentation used for optional cell type projection.
    segmentation_key_added : str
        Key for the VirTues segmentation output.
    cell_type_mask_key_added : str
        Key for the VirTues cell type mask output.
    cell_type_label_key_added : str
        Label column used when projecting VirTues cell types.
    patch_embedding_key_added : str
        Key for optional patch embeddings.
    patch_coord_key_added : str
        Key for optional patch coordinates.
    store_patch_embeddings : bool
        If True, store patch embeddings returned by the segmentation model.
    overwrite : bool
        If True, overwrite existing output layers.
    device : str
        Device used for model execution.

    Returns
    -------
    xr.Dataset
        The updated spatialproteomics object.
    """
    try:
        from virtues.utils.segmentation import assign_cell_types
    except ImportError as exc:
        raise _optional_import_error("virtues.utils.segmentation", exc)

    try:
        import torch
    except ImportError as exc:
        raise _optional_import_error("torch", exc)

    kwargs = {}
    if store_patch_embeddings:
        signature = inspect.signature(segmentation_model.segment_tissue)
        if "return_patch_embeddings" not in signature.parameters:
            raise RuntimeError(
                "The imported VirTues segment_tissue does not support return_patch_embeddings. "
                "Install the adapted local VirTues checkout or set store_patch_embeddings=False."
            )
        kwargs["return_patch_embeddings"] = True

    # on `torch.arange` (vs. `midxs`): The wrapper builds a compact marker embedding matrix 
    # in mapping order, so channel ids are dense row indices into that compact matrix.
    with torch.inference_mode():
        output = segmentation_model.segment_tissue(
            torch.as_tensor(image, dtype=torch.float32, device=device),
            torch.arange(len(mapping), dtype=torch.long, device=device),
            tile_size=tile_size,
            overlap=segment_overlap,
            batch_size=batch_size,
            **kwargs,
        )
    if store_patch_embeddings:
        segmentation, semantic_logits, patch_embeddings, patch_coords_yx = output
    else:
        segmentation, semantic_logits = output


    segmentation = segmentation.cpu().numpy().astype(np.int32)

    #-------- suppress cell types None and Unknown, following the VirTues tutorial. 
    reverse_cell_type_mapping = {v: k for k, v in VIRTUES_CELL_TYPE_MAPPING.items()}
    suppressed_cell_type_ids = [
        reverse_cell_type_mapping[cell_type]
        for cell_type in ["None", "Unknown"]
    ]
    semantic_logits = semantic_logits.cpu().clone()
    semantic_logits[suppressed_cell_type_ids, :, :] = float("-inf")
    #-------- 
    semantic_prediction = semantic_logits.argmax(0).numpy().astype(np.int32)
    cell_type_mask = assign_cell_types(segmentation, semantic_prediction).astype(np.int16)    

    attrs = {
        "model_name": model_name,
        "source": "segment_tissue",
        "tile_size_px": tile_size,
        "segment_overlap_px": segment_overlap,
        "batch_size": batch_size,
        "cell_type_suppressed_classes": ["None", "Unknown"],
    }
    attrs.update(preprocessing_attrs)
    obj = add_virtues_segmentation_outputs(
        obj,
        segmentation,
        cell_type_mask,
        segmentation_key_added=segmentation_key_added,
        cell_type_mask_key_added=cell_type_mask_key_added,
        attrs=attrs,
        overwrite=overwrite,
    )
    if segmentation_key_used in obj:
        obj = project_virtues_cell_types_to_existing_segmentation(
            obj,
            cell_type_mask,
            segmentation_key=segmentation_key_used,
            cell_type_label_key_added=cell_type_label_key_added,
            overwrite=overwrite,
        )
    if store_patch_embeddings:
        # segment patch geometry
        if max(image.shape[-2:]) > SEGMENT_LARGE_TISSUE_THRESHOLD:
            crop_overlap = 2 * (segment_overlap + SEGMENT_LARGE_DETECTION_SIZE)
        else:
            crop_overlap = 2 * segment_overlap
        crop_stride = max(tile_size - crop_overlap, 0)

        patch_attrs = {
            "model_name": model_name,
            "source": "segment_tissue",
            "tile_size_px": tile_size,
            "crop_stride_px": crop_stride,
            "crop_overlap_px": crop_overlap,
            "token_patch_size_px": TOKEN_PATCH_SIZE_PX,
            "token_stride_px": TOKEN_PATCH_SIZE_PX,
        }
        patch_attrs.update(preprocessing_attrs) #TODO: check what this does actually -- are there duplicates from the one before or how is it handled? 

        obj = add_patch_embeddings(
            obj,
            patch_embeddings.cpu().numpy(),
            patch_coords_yx.cpu().numpy(),
            patch_embedding_key_added,
            patch_coord_key_added,
            attrs=patch_attrs,
            overwrite=overwrite,
        )
    return obj


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


def run_virtues_cell_embedding(
    obj: xr.Dataset,
    image: np.ndarray,
    mapping: pd.DataFrame,
    virtues_model,
    model_name: str,
    preprocessing_attrs: dict,
    tile_size: int,
    cell_token_stride: int,
    batch_size: int,
    segmentation_key_used: str,
    cell_embedding_key_added: str,
    patch_embedding_key_added: str,
    patch_coord_key_added: str,
    store_patch_embeddings: bool,
    overwrite: bool,
    device: str,
):
    """
    Run VirTues cell embedding extraction and store cell-level embeddings.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    image : np.ndarray
        Preprocessed image with shape ``[channels, y, x]``.
    mapping : pd.DataFrame
        The normalized marker-to-protein mapping.
    virtues_model
        The VirTues backbone model.
    model_name : str
        The VirTues model name.
    preprocessing_attrs : dict
        Metadata returned by ``preprocess_for_virtues``.
    tile_size : int
        Tile size passed to ``compute_cell_tokens``.
    cell_token_stride : int
        Crop stride passed to ``compute_cell_tokens``.
    batch_size : int
        Chunk size passed to ``compute_cell_tokens``.
    segmentation_key_used : str
        Existing segmentation layer used to define cells.
    cell_embedding_key_added : str
        Key for the cell embedding output.
    patch_embedding_key_added : str
        Key for optional patch embeddings.
    patch_coord_key_added : str
        Key for optional patch coordinates.
    store_patch_embeddings : bool
        If True, store crop tokens and coordinates returned by ``compute_cell_tokens``.
    overwrite : bool
        If True, overwrite existing output layers.
    device : str
        Device used for model execution.

    Returns
    -------
    xr.Dataset
        The updated spatialproteomics object.
    """
    if segmentation_key_used not in obj:
        raise ValueError(f"mode='embed_cells' requires an existing segmentation layer {segmentation_key_used!r}.")
    if obj.sizes[Dims.Y] < tile_size or obj.sizes[Dims.X] < tile_size:
        raise ValueError(
            "VirTues compute_cell_tokens is not well-defined for images smaller than tile_size in v0.1."
        )

    try:
        from virtues.utils.cell_tokens import compute_cell_tokens
    except ImportError as exc:
        raise _optional_import_error("virtues.utils.cell_tokens", exc)

    try:
        import torch
    except ImportError as exc:
        raise _optional_import_error("torch", exc)

    segmentation = torch.as_tensor(obj[segmentation_key_used].values, device="cpu")
    cell_ids, cell_tokens, crop_tokens, indices = compute_cell_tokens(
        virtues_model,
        torch.as_tensor(image, dtype=torch.float32, device=device),
        torch.arange(len(mapping), dtype=torch.long, device=device),
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
        "model_name": model_name,
        "source": "compute_cell_tokens",
        "segmentation_key_used": segmentation_key_used,
        "tile_size_px": tile_size,
        "cell_token_stride_px": cell_token_stride,
        "token_patch_size_px": TOKEN_PATCH_SIZE_PX,
    }
    attrs.update(preprocessing_attrs)
    obj = add_virtues_embedding_outputs(
        obj,
        cell_ids=cell_ids.cpu().numpy(),
        cell_embeddings=cell_tokens.cpu().numpy().astype(np.float32),
        key_added=cell_embedding_key_added,
        attrs=attrs,
        overwrite=overwrite,
    )
    if store_patch_embeddings:
        patch_embeddings, patch_coords = _stack_crop_tokens(crop_tokens, indices)
        patch_attrs = {
            "model_name": model_name,
            "source": "compute_cell_tokens",
            "tile_size_px": tile_size,
            "crop_stride_px": cell_token_stride,
            "crop_overlap_px": max(tile_size - cell_token_stride, 0),
            "token_patch_size_px": TOKEN_PATCH_SIZE_PX,
            "token_stride_px": TOKEN_PATCH_SIZE_PX,
        }
        patch_attrs.update(preprocessing_attrs)
        obj = add_patch_embeddings(
            obj,
            patch_embeddings,
            patch_coords,
            patch_embedding_key_added,
            patch_coord_key_added,
            attrs=patch_attrs,
            overwrite=overwrite,
        )
    return obj


def _validate_virtues_arguments(
    obj: xr.Dataset,
    mode: str,
    store_patch_embeddings: bool,
    segmentation_key_used: str,
    segmentation_key_added: str,
    cell_type_mask_key_added: str,
    cell_type_label_key_added: str,
    cell_embedding_key_added: str,
    patch_embedding_key_added: str,
    patch_coord_key_added: str,
    overwrite: bool,
) -> None:
    if mode == "embed_cells":
        if segmentation_key_used not in obj:
            raise ValueError(f"mode='embed_cells' requires an existing segmentation layer {segmentation_key_used!r}.")
        _check_key_available(obj, cell_embedding_key_added, overwrite)
    else:
        _check_key_available(obj, segmentation_key_added, overwrite)
        _check_key_available(obj, cell_type_mask_key_added, overwrite)
        if segmentation_key_used in obj and Layers.LA_LAYERS in obj:
            existing = obj[Layers.LA_LAYERS].to_pandas()
            if cell_type_label_key_added in existing.columns and not overwrite:
                raise ValueError(f"Label layer column {cell_type_label_key_added} already exists.")

    if store_patch_embeddings:
        _check_key_available(obj, patch_embedding_key_added, overwrite)
        _check_key_available(obj, patch_coord_key_added, overwrite)


def virtues(
    obj: xr.Dataset,
    mode: str = "segment",
    protein_mapping: pd.DataFrame = None,
    marker_col: str = "marker_name",
    protein_id_col: str = "protein_id",
    image_key: str = Layers.IMAGE,
    model_name: str = "virtues-sp32",
    cache_dir=None,
    virtues_path=None,
    quantile: float = 0.99,
    mean: Optional[pd.Series] = None,
    std: Optional[pd.Series] = None,
    gaussian_blur: bool = True,
    gaussian_blur_sigma: float = 1.0,
    store_patch_embeddings: bool = False,
    tile_size: int = 128,
    segment_overlap: int = 32,
    cell_token_stride: int = 42,
    batch_size=None,
    segmentation_key_used: str = Layers.SEGMENTATION,
    segmentation_key_added: str = "_virtues_segmentation",
    cell_type_mask_key_added: str = "_virtues_cell_type_mask",
    cell_type_label_key_added: str = "_virtues_cell_type",
    cell_embedding_key_added: str = "_virtues_cell_embeddings",
    marker_embedding_key_added: str = "_virtues_marker_embeddings",
    patch_embedding_key_added: str = "_virtues_patch_embeddings",
    patch_coord_key_added: str = "_virtues_patch_coords",
    overwrite: bool = False,
    device: str = "cuda",
) -> xr.Dataset:
    """
    Run VirTues segmentation or cell embedding extraction.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    mode : {"segment", "embed_cells"}, optional
        The VirTues workflow to run. ``"segment"`` stores a VirTues segmentation and cell type mask.
        ``"embed_cells"`` stores cell-level embeddings for an existing segmentation.
    protein_mapping : pd.DataFrame
        A dataframe mapping image marker names to UniProt IDs.
    marker_col : str, optional
        Column in ``protein_mapping`` containing marker names.
    protein_id_col : str, optional
        Column in ``protein_mapping`` containing UniProt IDs.
    image_key : str, optional
        Image layer used as VirTues input.
    model_name : str, optional
        VirTues model name.
    cache_dir : str, optional
        Hugging Face cache directory used for model downloads.
    virtues_path : str, optional
        Local VirTues checkout to prepend to ``sys.path`` before importing VirTues.
    quantile : float, optional
        Per-channel image quantile used for clipping.
    mean : pd.Series, optional
        Dataset-level per-marker means, indexed by marker name.
    std : pd.Series, optional
        Dataset-level per-marker standard deviations, indexed by marker name.
    gaussian_blur : bool, optional
        If True, apply a Gaussian blur after standardization.
    gaussian_blur_sigma : float, optional
        Gaussian blur sigma.
    store_patch_embeddings : bool, optional
        If True, store patch embeddings and top-left crop coordinates.
    tile_size : int, optional
        VirTues tile size.
    segment_overlap : int, optional
        Tile overlap used for segmentation.
    cell_token_stride : int, optional
        Crop stride used for cell embedding extraction.
    batch_size : int, optional
        Batch size for segmentation or chunk size for cell embedding extraction.
    segmentation_key_used : str, optional
        Existing segmentation used for cell embedding extraction and cell type projection.
    segmentation_key_added : str, optional
        Key for the VirTues segmentation output.
    cell_type_mask_key_added : str, optional
        Key for the VirTues cell type mask output.
    cell_type_label_key_added : str, optional
        Label column used when projecting VirTues cell types onto an existing segmentation.
    cell_embedding_key_added : str, optional
        Key for the VirTues cell embedding output.
    marker_embedding_key_added : str, optional
        Key for marker embeddings.
    patch_embedding_key_added : str, optional
        Key for optional patch embeddings.
    patch_coord_key_added : str, optional
        Key for optional patch coordinates.
    overwrite : bool, optional
        If True, overwrite existing output layers.
    device : str, optional
        Device used for model execution. Only ``"cuda"`` is supported in v0.1.

    Returns
    -------
    xr.Dataset
        The updated spatialproteomics object.

    Notes
    -----
    UniProt IDs must be supplied explicitly in ``protein_mapping``. DAPI or Histone H3 are only used
    if they are represented in the mapping. The canonical ``_segmentation`` layer is not overwritten
    by default; VirTues segmentation is stored under ``_virtues_segmentation``. Optional VirTues
    dependencies are imported only when this function is called.
    """
    if mode not in {"segment", "embed_cells"}:
        raise ValueError("mode must be either 'segment' or 'embed_cells'.")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported VirTues model_name {model_name!r}.")
    if device != "cuda":
        raise ValueError("Only device='cuda' is supported for sp.tl.virtues(...) in v0.1.")
    _validate_virtues_arguments(
        obj=obj,
        mode=mode,
        store_patch_embeddings=store_patch_embeddings,
        segmentation_key_used=segmentation_key_used,
        segmentation_key_added=segmentation_key_added,
        cell_type_mask_key_added=cell_type_mask_key_added,
        cell_type_label_key_added=cell_type_label_key_added,
        cell_embedding_key_added=cell_embedding_key_added,
        patch_embedding_key_added=patch_embedding_key_added,
        patch_coord_key_added=patch_coord_key_added,
        overwrite=overwrite,
    )
    _add_virtues_path(virtues_path) # TODO: document that this assumes virtues has not been imported yet, otherwise the local path will not override already cached sys.modules['virtues']

    try:
        import torch
    except ImportError as exc:
        raise _optional_import_error("torch", exc)

    if not torch.cuda.is_available():
        raise RuntimeError("sp.tl.virtues(...) requires CUDA for v0.1 model execution.")

    if batch_size is None:
        batch_size = 4 if mode == "segment" else 32

    mapping = validate_and_normalize_protein_mapping(
        obj, protein_mapping, marker_col, protein_id_col, model_name
    )
    image, preprocessing_attrs = preprocess_for_virtues(
        obj,
        mapping,
        image_key=image_key,
        quantile=quantile,
        mean=mean,
        std=std,
        gaussian_blur=gaussian_blur,
        gaussian_blur_sigma=gaussian_blur_sigma,
    )
    obj, marker_embeddings = get_or_create_marker_embeddings(
        obj,
        mapping,
        marker_embedding_key_added,
        overwrite=overwrite,
        device=device,
    )
    virtues_model, segmentation_model = load_virtues_model(
        marker_embeddings=marker_embeddings,
        model_name=model_name,
        cache_dir=cache_dir,
        virtues_path=virtues_path,
        device=device,
    )

    if mode == "segment":
        obj = run_virtues_segmentation(
            obj,
            image,
            mapping,
            segmentation_model,
            model_name=model_name,
            preprocessing_attrs=preprocessing_attrs,
            tile_size=tile_size,
            segment_overlap=segment_overlap,
            batch_size=batch_size,
            segmentation_key_used=segmentation_key_used,
            segmentation_key_added=segmentation_key_added,
            cell_type_mask_key_added=cell_type_mask_key_added,
            cell_type_label_key_added=cell_type_label_key_added,
            patch_embedding_key_added=patch_embedding_key_added,
            patch_coord_key_added=patch_coord_key_added,
            store_patch_embeddings=store_patch_embeddings,
            overwrite=overwrite,
            device=device,
        )
    elif mode == "embed_cells":
        obj = run_virtues_cell_embedding(
            obj,
            image,
            mapping,
            virtues_model,
            model_name=model_name,
            preprocessing_attrs=preprocessing_attrs,
            tile_size=tile_size,
            cell_token_stride=cell_token_stride,
            batch_size=batch_size,
            segmentation_key_used=segmentation_key_used,
            cell_embedding_key_added=cell_embedding_key_added,
            patch_embedding_key_added=patch_embedding_key_added,
            patch_coord_key_added=patch_coord_key_added,
            store_patch_embeddings=store_patch_embeddings,
            overwrite=overwrite,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported mode {mode!r}.")
    return obj
