from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ..constants import Dims, Layers
from .virtues_utils import (
    ESM_MODEL,
    VIRTUES_CELL_TYPE_MAPPING,
)


MARKER_METADATA_DIM = "virtues_marker_metadata"
VIRTUES_MARKERS_DIM = "virtues_markers"
VIRTUES_MARKER_EMBEDDING_DIM = "virtues_marker_embedding"
VIRTUES_PATCHES_DIM = "virtues_patches"
TOKEN_Y_DIM = "token_y"
TOKEN_X_DIM = "token_x"
VIRTUES_EMBEDDING_DIM = "virtues_embedding"
YX_DIM = "yx"


def _check_key_available(obj: xr.Dataset, key: str, overwrite: bool) -> None:
    if key in obj and not overwrite:
        raise ValueError(f"Layer {key} already exists. Use overwrite=True or choose another key.")


def _drop_if_overwrite(obj: xr.Dataset, key: str, overwrite: bool) -> xr.Dataset:
    _check_key_available(obj, key, overwrite)
    if key in obj and overwrite:
        return obj.drop_vars(key)
    return obj


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
        Dataset containing the original data, marker embeddings, and marker embedding metadata.

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
        order = pd.Index(mapping["marker_name"]).get_indexer(channels)
        stored_embeddings = embeddings[order]
        stored_mapping = mapping.iloc[order].reset_index(drop=True)
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


def _get_stored_virtues_marker_embeddings(
    obj: xr.Dataset,
    mapping: pd.DataFrame,
    marker_embedding_key: str,
    overwrite: bool = False,
) -> Optional[np.ndarray]:
    """Return matching stored priors in mapping order, or None when new priors are needed.

    overwrite=True bypasses stored priors so the shared workflow computes fresh priors. Otherwise,
    incompatible stored priors raise. Lookup never modifies the dataset, computes priors, or imports
    optional model dependencies.
    """
    if overwrite:
        return None
    if marker_embedding_key in obj:
        layer = obj[marker_embedding_key]
        metadata_matches = (
            layer.attrs.get("model_marker_order") == mapping["marker_name"].tolist()
            and layer.attrs.get("model_protein_id_order") == mapping["protein_id"].tolist()
            and layer.attrs.get("model_name") == mapping.attrs.get("model_name")
            and layer.attrs.get("esm_model") == mapping.attrs.get("esm_model")
        )
        if metadata_matches:
            embeddings = _marker_embeddings_from_layer(obj, marker_embedding_key, mapping)
            return embeddings
        raise ValueError(
            f"Marker embedding layer {marker_embedding_key!r} exists but metadata does not match the "
            "requested protein mapping. Use overwrite=True or choose another key."
        )

    return None


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
        Dataset containing the original data, patch embeddings, and patch top-left coordinates.

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
    cell_type_prediction: Optional[np.ndarray],
    segmentation_key_added: str,
    cell_type_prediction_key_added: Optional[str],
    attrs: dict,
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Add a canonical or custom VirTues segmentation and optional pixel-wise cell type predictions.

    Parameters
    ----------
    obj : xr.Dataset
        The spatialproteomics object.
    segmentation : np.ndarray
        VirTues instance segmentation with shape ``[y, x]``.
    cell_type_prediction : np.ndarray or None
        VirTues pixel-wise class predictions with shape ``[y, x]``, before assignment to cells.
    segmentation_key_added : str
        The key under which the VirTues segmentation will be stored.
    cell_type_prediction_key_added : str or None
        Store the optional predictions under this key; None skips prediction storage.
    attrs : dict
        Metadata to store on both output layers.
    overwrite : bool, optional
        If True, overwrite custom output layers. An existing canonical segmentation cannot be replaced.

    Returns
    -------
    xr.Dataset
        Dataset containing the original data, segmentation, and pixel-wise predictions only when requested.

    Notes
    -----
    Writing to "_segmentation" establishes canonical cell IDs and geometric observations and errors
    if that layer already exists, even with overwrite=True. A custom key stores only a mask and leaves
    canonical cells unchanged; without a canonical segmentation it does not establish cells.
    Cell type predictions require a distinct noncanonical key and a matching array shape. Their attrs store
    the VirTues cell type mapping. No cell type annotations are added automatically.
    """
    if segmentation.shape != (obj.sizes[Dims.Y], obj.sizes[Dims.X]):
        raise ValueError("VirTues segmentation shape must match image y/x shape.")
    if segmentation_key_added == Layers.SEGMENTATION and Layers.SEGMENTATION in obj:
        raise ValueError("Canonical segmentation '_segmentation' already exists. Choose a custom segmentation_key_added; overwrite=True cannot replace canonical cells.")
    _check_key_available(obj, segmentation_key_added, overwrite)
    if cell_type_prediction_key_added is not None:
        if cell_type_prediction_key_added in {Layers.SEGMENTATION, segmentation_key_added}:
            raise ValueError("Cell type prediction output must use a distinct noncanonical key.")
        if cell_type_prediction is None or cell_type_prediction.shape != segmentation.shape:
            raise ValueError("cell_type_prediction must be supplied with the same shape as segmentation when a prediction key is requested.")
        _check_key_available(obj, cell_type_prediction_key_added, overwrite)

    if segmentation_key_added == Layers.SEGMENTATION:
        obj = obj.pp.add_segmentation(segmentation.astype(np.int32), reindex=False, keep_labels=False)
    else:
        obj = _drop_if_overwrite(obj, segmentation_key_added, overwrite)
        obj = obj.pp.add_layer(segmentation.astype(np.int32), key_added=segmentation_key_added)
    obj[segmentation_key_added].attrs = dict(attrs)
    if cell_type_prediction_key_added is not None:
        obj = _drop_if_overwrite(obj, cell_type_prediction_key_added, overwrite)
        obj = obj.pp.add_layer(cell_type_prediction.astype(np.int16), key_added=cell_type_prediction_key_added)
        obj[cell_type_prediction_key_added].attrs = {**attrs, "cell_type_mapping": VIRTUES_CELL_TYPE_MAPPING}
    return obj


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
        Dataset containing the original data and VirTues cell embeddings aligned to ``cells``.

    Notes
    -----
    Embeddings are aligned to ``obj.coords['cells']``, the canonical ``_segmentation`` cell population. Cells without a returned
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


def _get_virtues_canonical_segmentation(obj: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract canonical segmentation and ordered IDs, rejecting inconsistent cell populations."""
    if Layers.SEGMENTATION not in obj:
        raise ValueError("VirTues requires an existing segmentation layer '_segmentation'.")
    if Dims.CELLS not in obj.coords or obj.coords[Dims.CELLS].dims != (Dims.CELLS,):
        raise ValueError("VirTues requires a one-dimensional cells coordinate for canonical segmentation.")
    if obj[Layers.SEGMENTATION].dims != (Dims.Y, Dims.X):
        raise ValueError("Canonical segmentation must have dimensions ['y', 'x'].")
    segmentation = obj[Layers.SEGMENTATION].values
    cells = obj.coords[Dims.CELLS].values
    if not np.array_equal(np.sort(cells), np.unique(segmentation[segmentation > 0])):
        raise ValueError("Canonical cell IDs must match the unique positive IDs in '_segmentation'.")
    return segmentation, cells


def _validate_virtues_arguments(
    obj: xr.Dataset,
    mode: str,
    image_key: str,
    store_patch_embeddings: bool,
    segmentation_key_added: str,
    cell_type_prediction_key_added: Optional[str],
    cell_embedding_key_added: str,
    marker_embedding_key_added: str,
    patch_embedding_key_added: str,
    patch_coord_key_added: str,
    overwrite: bool,
) -> None:
    if mode not in {"segment", "embed_cells"}:
        raise ValueError("mode must be either 'segment' or 'embed_cells'.")
    if image_key not in obj:
        raise KeyError(f"Image layer {image_key!r} not found.")
    expected_dims = (Dims.CHANNELS, Dims.Y, Dims.X)
    if obj[image_key].dims != expected_dims:
        raise ValueError(f"Image layer {image_key!r} must have dimensions {list(expected_dims)}; found {list(obj[image_key].dims)}.")
    if mode == "segment":
        if segmentation_key_added == Layers.SEGMENTATION and Layers.SEGMENTATION in obj:
            raise ValueError("Canonical segmentation '_segmentation' already exists. Choose a custom segmentation_key_added; overwrite=True cannot replace canonical cells.")
        output_keys = [segmentation_key_added]
        if cell_type_prediction_key_added is not None:
            output_keys.append(cell_type_prediction_key_added)
    else:
        if Layers.SEGMENTATION not in obj:
            raise ValueError("mode='embed_cells' requires an existing segmentation layer '_segmentation'.")
        if Dims.CELLS not in obj.coords:
            raise ValueError("mode='embed_cells' requires a cells coordinate.")
        output_keys = [cell_embedding_key_added]
    if store_patch_embeddings:
        output_keys.extend([patch_embedding_key_added, patch_coord_key_added])
    prior_keys = [marker_embedding_key_added, f"{marker_embedding_key_added}_metadata"]
    active_keys = output_keys + prior_keys
    if any(not isinstance(key, str) or not key for key in active_keys):
        raise ValueError("Output keys must be non-empty strings.")
    if len(active_keys) != len(set(active_keys)):
        raise ValueError("Requested output keys must be distinct.")
    if image_key in active_keys:
        raise ValueError("Output keys cannot replace the source image.")
    for key in active_keys:
        if key == Layers.SEGMENTATION and not (mode == "segment" and key == segmentation_key_added):
            raise ValueError("Only the instance-segmentation output may use canonical '_segmentation'.")
    for key in output_keys:
        _check_key_available(obj, key, overwrite)
    if marker_embedding_key_added not in obj:
        _check_key_available(obj, prior_keys[1], overwrite)
