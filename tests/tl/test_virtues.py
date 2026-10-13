import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import spatialproteomics  # noqa: F401
from spatialproteomics.constants import Dims, Layers
from spatialproteomics.tl import virtues as vt


def _image_dataset():
    image = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    image[0, 0, 0] = -5
    return xr.Dataset(
        {
            Layers.IMAGE: xr.DataArray(
                image,
                coords=[["A", "B", "C"], np.arange(4), np.arange(4)],
                dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            )
        }
    )


def _mapping(markers=("A", "B", "C"), proteins=("P1", "P2", "P3")):
    return pd.DataFrame({"marker_name": markers, "protein_id": proteins})


def _segmented_dataset():
    return _image_dataset().pp.add_segmentation(
        np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 0, 0],
                [3, 3, 0, 0],
            ],
            dtype=np.int32,
        )
    )


def _patch_dataset():
    ds = _image_dataset().pp.add_segmentation(
        np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.int32,
        )
    )
    patch_embeddings = xr.DataArray(
        np.array([[[[1.0], [3.0]], [[5.0], [7.0]]]], dtype=np.float16),
        dims=[
            vt.VIRTUES_PATCHES_DIM,
            vt.TOKEN_Y_DIM,
            vt.TOKEN_X_DIM,
            vt.VIRTUES_EMBEDDING_DIM,
        ],
        coords={
            vt.VIRTUES_PATCHES_DIM: [0],
            vt.TOKEN_Y_DIM: [0, 1],
            vt.TOKEN_X_DIM: [0, 1],
            vt.VIRTUES_EMBEDDING_DIM: [0],
        },
        attrs={
            "tile_size_px": 4,
            "crop_stride_px": 4,
            "crop_overlap_px": 0,
            "token_patch_size_px": 2,
            "token_stride_px": 2,
            "source": "segment_tissue",
        },
        name="_virtues_patch_embeddings",
    )
    patch_coords = xr.DataArray(
        np.array([[0, 0]], dtype=np.int32),
        dims=[vt.VIRTUES_PATCHES_DIM, vt.YX_DIM],
        coords={vt.VIRTUES_PATCHES_DIM: [0], vt.YX_DIM: ["y", "x"]},
        name="_virtues_patch_coords",
    )
    return xr.merge([ds, patch_embeddings, patch_coords])


def test_validate_and_normalize_protein_mapping():
    """
    Check that protein_mapping is normalized to the internal column names,
    subset to markers present in the image, and rejected when required
    information is missing or ambiguous.
    """
    ds = _image_dataset()
    mapping = pd.DataFrame({"marker": [" B ", "D", "A"], "uniprot": [" P2 ", "P4", "P1"]})

    resolved = vt.validate_and_normalize_protein_mapping(ds, mapping, "marker", "uniprot", "virtues-sp32")

    assert resolved["marker_name"].tolist() == ["B", "A"]
    assert resolved["protein_id"].tolist() == ["P2", "P1"]
    assert resolved["channel_index"].tolist() == [1, 0]
    assert resolved["marker_embedding_index"].tolist() == [0, 1]

    with pytest.raises(ValueError, match="protein_mapping must be supplied"):
        vt.validate_and_normalize_protein_mapping(ds, None, "marker_name", "protein_id", "virtues-sp32")

    with pytest.raises(ValueError, match="Protein IDs must be non-empty"):
        vt.validate_and_normalize_protein_mapping(
            ds, _mapping(("A",), (" ",)), "marker_name", "protein_id", "virtues-sp32"
        )

    with pytest.raises(ValueError, match="Marker names must be unique"):
        vt.validate_and_normalize_protein_mapping(
            ds, _mapping(("A", "A"), ("P1", "P2")), "marker_name", "protein_id", "virtues-sp32"
        )

    with pytest.raises(ValueError, match="No protein_mapping markers are present"):
        vt.validate_and_normalize_protein_mapping(
            ds, _mapping(("D",), ("P4",)), "marker_name", "protein_id", "virtues-sp32"
        )


def test_preprocess_for_virtues():
    """
    Check that preprocessing follows the VirTues order:
    select mapped markers, clip negative values to zero, apply log1p,
    align dataset-level mean/std by marker name, and validate common
    invalid inputs.
    """
    ds = _image_dataset()
    mapping = vt.validate_and_normalize_protein_mapping(
        ds, _mapping(("B", "A"), ("P2", "P1")), "marker_name", "protein_id", "virtues-sp32"
    )
    mean = pd.Series({"A": 10.0, "B": 20.0})
    std = pd.Series({"B": 5.0, "A": 2.0})

    image, attrs = vt.preprocess_for_virtues(
        ds,
        mapping,
        image_key=Layers.IMAGE,
        quantile=1.0,
        mean=mean,
        std=std,
        gaussian_blur=False,
    )

    expected_b = (np.log1p(np.clip(ds[Layers.IMAGE].sel(channels="B").values, 0, None)) - 20.0) / 5.0
    expected_a = (np.log1p(np.clip(ds[Layers.IMAGE].sel(channels="A").values, 0, None)) - 10.0) / 2.0
    np.testing.assert_allclose(image[0], expected_b.astype(np.float32))
    np.testing.assert_allclose(image[1], expected_a.astype(np.float32))
    assert attrs["mean_std_level"] == "dataset"
    assert attrs["markers_used"] == ["B", "A"]

    with pytest.raises(ValueError, match="mean and std must either both be supplied"):
        vt.preprocess_for_virtues(ds, mapping, Layers.IMAGE, mean=mean, std=None)

    with pytest.raises(TypeError, match="mean and std must be pandas Series"):
        vt.preprocess_for_virtues(ds, mapping, Layers.IMAGE, mean=pd.DataFrame(), std=pd.DataFrame())

    # check that transposed image with wrong dims raises error
    bad_dims = ds.copy()
    bad_dims[Layers.IMAGE] = bad_dims[Layers.IMAGE].transpose(Dims.Y, Dims.X, Dims.CHANNELS)
    bad_dims_mapping = vt.validate_and_normalize_protein_mapping(
        bad_dims, _mapping(("A",), ("P1",)), "marker_name", "protein_id", "virtues-sp32"
    )
    with pytest.raises(ValueError, match="must have dimensions"):
        vt.preprocess_for_virtues(bad_dims, bad_dims_mapping, Layers.IMAGE)

    # check that constant image with zero std raises error when using image-level mean/std fallback
    constant_ds = xr.Dataset(
        {
            Layers.IMAGE: xr.DataArray(
                np.ones((1, 3, 3), dtype=np.float32),
                coords=[["A"], np.arange(3), np.arange(3)],
                dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            )
        }
    )
    constant_mapping = vt.validate_and_normalize_protein_mapping(
        constant_ds, _mapping(("A",), ("P1",)), "marker_name", "protein_id", "virtues-sp32"
    )
    with pytest.raises(ValueError, match="standard deviations must be finite and > 0"):
        vt.preprocess_for_virtues(constant_ds, constant_mapping, Layers.IMAGE, gaussian_blur=False)


def test_add_marker_embeddings():
    """
    Check that full marker mappings are stored on the native channels
    dimension, while subset mappings are stored on the separate
    virtues_markers dimension.
    """
    ds = _image_dataset()
    mapping = vt.validate_and_normalize_protein_mapping(
        ds, _mapping(("B", "A", "C"), ("P2", "P1", "P3")), "marker_name", "protein_id", "virtues-sp32"
    )
    embeddings = np.array([[2, 20], [1, 10], [3, 30]], dtype=np.float32)

    out = vt.add_marker_embeddings(ds, embeddings, mapping, "_virtues_marker_embeddings")

    # full marker set used in embeddings: store along Dims.CHANNELS
    assert out["_virtues_marker_embeddings"].dims == (Dims.CHANNELS, vt.VIRTUES_MARKER_EMBEDDING_DIM)
    assert vt.VIRTUES_MARKERS_DIM not in out.coords
    np.testing.assert_array_equal(
        out["_virtues_marker_embeddings"].sel(channels=["A", "B", "C"]).values,
        [[1, 10], [2, 20], [3, 30]],
    )
    # check that even if model marker order differs, the attrs preserve the stored
    # marker order corresponding to the channels
    assert out["_virtues_marker_embeddings"].attrs["model_marker_order"] == ["B", "A", "C"]
    assert out["_virtues_marker_embeddings"].attrs["stored_marker_order"] == ["A", "B", "C"]

    # check that using a subset of markers works; this uses vt.VIRTUES_MARKERS_DIM
    subset_mapping = vt.validate_and_normalize_protein_mapping(
        ds, _mapping(("C", "A"), ("P3", "P1")), "marker_name", "protein_id", "virtues-sp32"
    )
    out = vt.add_marker_embeddings(ds, np.ones((2, 3), dtype=np.float32), subset_mapping, "_virtues_marker_embeddings")

    assert out["_virtues_marker_embeddings"].dims == (vt.VIRTUES_MARKERS_DIM, vt.VIRTUES_MARKER_EMBEDDING_DIM)
    assert vt.VIRTUES_MARKERS_DIM in out.coords
    assert out.coords[vt.VIRTUES_MARKERS_DIM].values.tolist() == ["C", "A"]


def test_get_or_create_marker_embeddings_reuses_matching_layer(monkeypatch):
    """
    Check that stored marker embeddings are reused when their metadata
    matches the requested protein mapping, without downloading FASTA
    sequences or recomputing embeddings.
    """
    # do not recompute marker embeddings if they already exist and match
    ds = _image_dataset()
    mapping = vt.validate_and_normalize_protein_mapping(ds, _mapping(), "marker_name", "protein_id", "virtues-sp32")
    ds = vt.add_marker_embeddings(ds, np.ones((3, 2), dtype=np.float32), mapping, "_virtues_marker_embeddings")

    class FakeTorch(types.SimpleNamespace):
        float32 = "float32"

        @staticmethod
        def as_tensor(values, dtype=None, device=None):
            return np.asarray(values, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setattr(
        vt,
        "_download_uniprot_fasta",
        lambda protein_id: (_ for _ in ()).throw(AssertionError("should not download FASTA")),
    )

    out, embeddings = vt.get_or_create_marker_embeddings(ds, mapping, "_virtues_marker_embeddings", device="cpu")

    assert out.identical(ds)
    np.testing.assert_array_equal(embeddings, np.ones((3, 2), dtype=np.float32))


def test_add_patch_embeddings():
    """
    Check that patch embeddings and their paired crop coordinates are stored
    with the expected dtype and coordinate metadata, and that existing output
    keys are not overwritten by default.
    """
    ds = _image_dataset()
    patch_embeddings = np.ones((1, 2, 2, 3), dtype=np.float32)
    patch_coords = np.array([[0, 0]], dtype=np.int32)

    out = vt.add_patch_embeddings(
        ds,
        patch_embeddings,
        patch_coords,
        "_virtues_patch_embeddings",
        "_virtues_patch_coords",
        attrs={"model_name": "virtues-sp32", "source": "segment_tissue", "tile_size_px": 16},
    )

    assert out["_virtues_patch_embeddings"].dtype == np.float16
    assert out["_virtues_patch_coords"].attrs["coordinate_convention"] == "top-left pixel (y, x)"
    assert out["_virtues_patch_coords"].attrs["paired_embedding_key"] == "_virtues_patch_embeddings"

    with pytest.raises(ValueError, match="already exists"):
        vt.add_patch_embeddings(
            out,
            patch_embeddings,
            patch_coords,
            "_virtues_patch_embeddings",
            "_virtues_patch_coords",
            attrs={"source": "segment_tissue"},
        )

    with pytest.raises(ValueError, match="finite"):
        vt.add_patch_embeddings(ds, patch_embeddings, np.array([[np.nan, 0]]), "_emb", "_coords", attrs={})


def test_add_virtues_segmentation_outputs():
    """
    Check that VirTues segmentation outputs are added under their own keys,
    without overwriting the canonical spatialproteomics segmentation layer.
    """
    ds = _image_dataset().pp.add_segmentation(np.ones((4, 4), dtype=np.int32))
    segmentation = np.full((4, 4), 2, dtype=np.int32)
    cell_type_mask = np.full((4, 4), 1, dtype=np.int16)

    out = vt.add_virtues_segmentation_outputs(
        ds,
        segmentation,
        cell_type_mask,
        segmentation_key_added="_virtues_segmentation",
        cell_type_mask_key_added="_virtues_cell_type_mask",
        attrs={"model_name": "virtues-sp32", "source": "segment_tissue"},
    )

    assert out[Layers.SEGMENTATION].identical(ds[Layers.SEGMENTATION])
    np.testing.assert_array_equal(out["_virtues_segmentation"].values, segmentation)
    np.testing.assert_array_equal(out["_virtues_cell_type_mask"].values, cell_type_mask)
    assert out["_virtues_segmentation"].dtype == np.int32
    assert out["_virtues_cell_type_mask"].dtype == np.int16
    assert out["_virtues_cell_type_mask"].attrs["cell_type_mapping"] == vt.VIRTUES_CELL_TYPE_MAPPING


def test_project_virtues_cell_types_to_existing_segmentation():
    """
    Check that a pixel-level VirTues cell type mask is projected onto the
    existing spatialproteomics cells by majority vote and stored as a label
    annotation layer.
    """
    cell_type_mask = np.array(
        [
            [0, 1, 2, 3],
            [1, 1, 3, 2],
            [0, 0, 0, 0],
            [4, 4, 0, 0],
        ],
        dtype=np.int16,
    )

    out = vt.project_virtues_cell_types_to_existing_segmentation(_segmented_dataset(), cell_type_mask)

    labels = out[Layers.LA_LAYERS].sel(la_features="_virtues_cell_type").values.tolist()
    assert labels == ["B cell", "CD4 T cell", "None"]


def test_add_virtues_embedding_outputs():
    """
    Check that add_virtues_embedding_outputs(...) aligns embeddings to the 
    dataset's own cells coordinate, instead of storing them in input order.
    
    Expected result: 
    cell 1 -> [10.0, 100.0]
    cell 2 -> [nan, nan]
    cell 3 -> [30.0, 300.0]
    """
    ds = _segmented_dataset() # has cells [1,2,3]

    out = vt.add_virtues_embedding_outputs(
        ds,
        cell_ids=np.array([3, 1]),
        cell_embeddings=np.array([[30.0, 300.0], [10.0, 100.0]], dtype=np.float32),
        key_added="_virtues_cell_embeddings",
        attrs={"model_name": "virtues-sp32", "source": "compute_cell_tokens"},
    )

    np.testing.assert_allclose(
        out["_virtues_cell_embeddings"].sel(cells=[1, 3]).values,
        [[10.0, 100.0], [30.0, 300.0]],
    )
    assert np.isnan(out["_virtues_cell_embeddings"].sel(cells=2).values).all()

    with pytest.raises(ValueError, match="without a cells coordinate"):
        vt.add_virtues_embedding_outputs(
            _image_dataset(),
            cell_ids=np.array([1]),
            cell_embeddings=np.array([[1.0]], dtype=np.float32),
            key_added="_virtues_cell_embeddings",
            attrs={},
        )


def test_virtues_aggregate_patch_embeddings():
    """
    Check that patch token embeddings are aggregated into cell embeddings
    using pixel overlap between token windows and the segmentation mask.
    Segmentation and patch embeddings are contrived to produce a simple, known result.
    Specifically: based on the segmentation, cell 1 receives token values 1.0 and 5.0,
    and cell 2 receives token values 3.0 and 7.0, so the expected means are 
    cell 1: (1 + 5) / 2 = 3.0; cell 2: (3 + 7) / 2 = 5.0.
    """
    out = vt.virtues_aggregate_patch_embeddings(_patch_dataset())

    np.testing.assert_allclose(out["_virtues_cell_embeddings"].sel(cells=[1, 2]).values[:, 0], [3.0, 5.0])
    assert out["_virtues_cell_embeddings"].attrs["source"] == "aggregate:segment_tissue"


def test_tool_accessor_virtues_methods_delegate_cpu_only_paths():
    """
    Check the public tl accessor methods: model execution should 
    reject CPU (non-CUDA devices), while patch aggregation should 
    delegate to the implementation function that does not require 
    VirTues/Torch imports. 
    """
    with pytest.raises(ValueError, match="Only device='cuda'"):
        _image_dataset().tl.virtues(protein_mapping=_mapping(), device="cpu")

    out = _patch_dataset().tl.virtues_aggregate_patch_embeddings()

    assert "_virtues_cell_embeddings" in out


def test_virtues_embed_cells_requires_segmentation_before_optional_imports():
    """
    Check that the embed_cells mode validates the required segmentation layer
    before optional VirTues/Torch imports would be needed.
    """
    with pytest.raises(ValueError, match="requires an existing segmentation layer"):
        vt.virtues(_image_dataset(), mode="embed_cells", protein_mapping=_mapping(), device="cuda")
