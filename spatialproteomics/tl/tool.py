import copy as cp
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ..base_logger import logger
from ..constants import Dims, Features, Layers, Props, SDLayers
from ..sd.utils import (
    _get_channels_spatialdata,
    _process_adata,
    _process_image,
    _process_segmentation,
)
from .utils import (
    _astir,
    _cellpose,
    _compute_transformation,
    _convert_masks_to_data_array,
    _get_channels,
    _mesmer,
    _spotiflow,
    _stardist,
)


# === SPATIALDATA ACCESSOR ===
def cellpose(
    sdata,
    channel: Optional[str] = None,
    image_key: str = SDLayers.IMAGE,
    key_added: str = SDLayers.SEGMENTATION,
    data_key: Optional[str] = None,
    copy: bool = False,
    **kwargs,
):
    """
    This function runs the cellpose segmentation algorithm on the provided image data.
    It extracts the image data from the spatialdata object, applies the cellpose algorithm,
    and adds the segmentation masks to the spatialdata object.
    The segmentation masks are stored in the labels attribute of the spatialdata object.
    The function also handles multiple channels by iterating over the channels and applying the segmentation algorithm to each channel separately.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
        channel (Optional[str]): The channel(s) to be used for segmentation. If None, all channels will be used.
        image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
        key_added (str, optional): The key under which the segmentation masks will be stored in the labels attribute of the spatialdata object. Defaults to segmentation.
        data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the cellpose algorithm.
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    if copy:
        sdata = cp.deepcopy(sdata)

    channels = _get_channels_spatialdata(channel)

    # assert that the format is correct and extract the image
    image = _process_image(sdata, channels=channels, image_key=image_key, key_added=key_added, data_key=data_key)

    # run cellpose
    segmentation_masks, _ = _cellpose(image, **kwargs)

    # get transformations
    transformation = get_transformation(sdata[image_key])

    # add the segmentation masks to the spatial data object
    if segmentation_masks.shape[0] > 1:
        for i, channel in enumerate(channels):
            sdata.labels[f"{key_added}_{channel}"] = spatialdata.models.Labels2DModel.parse(
                segmentation_masks[i], transformations=None, dims=("y", "x")
            )
            set_transformation(sdata.labels[f"{key_added}_{channel}"], transformation)
    else:
        sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
            segmentation_masks[0], transformations=None, dims=("y", "x")
        )
        set_transformation(sdata.labels[key_added], transformation)

    if copy:
        return sdata


def stardist(
    sdata,
    channel: Optional[str] = None,
    image_key: str = SDLayers.IMAGE,
    key_added: str = SDLayers.SEGMENTATION,
    data_key: Optional[str] = None,
    copy: bool = False,
    **kwargs,
):
    """
    This function runs the stardist segmentation algorithm on the provided image data.
    It extracts the image data from the spatialdata object, applies the stardist algorithm,
    and adds the segmentation masks to the spatialdata object.
    The segmentation masks are stored in the labels attribute of the spatialdata object.
    The function also handles multiple channels by iterating over the channels and applying the segmentation algorithm to each channel separately.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
        channel (Optional[str]): The channel(s) to be used for segmentation. If None, all channels will be used.
        image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
        key_added (str, optional): The key under which the segmentation masks will be stored in the labels attribute of the spatialdata object. Defaults to segmentation.
        data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the stardist algorithm.
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    if copy:
        sdata = cp.deepcopy(sdata)

    channels = _get_channels_spatialdata(channel)

    # assert that the format is correct and extract the image
    image = _process_image(sdata, channels=channels, image_key=image_key, key_added=key_added, data_key=data_key)

    # run stardist
    segmentation_masks = _stardist(image, **kwargs)

    # get transformations
    transformation = get_transformation(sdata[image_key])

    # add the segmentation masks to the spatial data object
    if segmentation_masks.shape[0] > 1:
        for i, channel in enumerate(channels):
            sdata.labels[f"{key_added}_{channel}"] = spatialdata.models.Labels2DModel.parse(
                segmentation_masks[i], transformations=None, dims=("y", "x")
            )
            set_transformation(sdata.labels[f"{key_added}_{channel}"], transformation)
    else:
        sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
            segmentation_masks[0], transformations=None, dims=("y", "x")
        )
        set_transformation(sdata.labels[key_added], transformation)

    if copy:
        return sdata


def mesmer(
    sdata,
    channel: Optional[str] = None,
    image_key: str = SDLayers.IMAGE,
    key_added: str = SDLayers.SEGMENTATION,
    data_key: Optional[str] = None,
    copy: bool = False,
    **kwargs,
):
    """
    This function runs the mesmer segmentation algorithm on the provided image data.
    It extracts the image data from the spatialdata object, applies the mesmer algorithm,
    and adds the segmentation masks to the spatialdata object.
    The segmentation masks are stored in the labels attribute of the spatialdata object.
    The first channel is assumed to be nuclear and the second one membraneous.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
        channel (Optional[str]): The channel(s) to be used for segmentation.
        image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
        key_added (str, optional): The key under which the segmentation masks will be stored in the labels attribute of the spatialdata object. Defaults to segmentation.
        data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the mesmer algorithm.
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    if copy:
        sdata = cp.deepcopy(sdata)

    channels = _get_channels_spatialdata(channel)

    assert (
        len(channels) == 2
    ), "Mesmer only supports two channel segmentation. Please ensure that the first channel is nuclear and the second one is membraneous."

    # assert that the format is correct and extract the image
    image = _process_image(sdata, channels=channels, image_key=image_key, key_added=key_added, data_key=data_key)

    # run mesmer
    segmentation_masks = _mesmer(image, **kwargs)

    # get transformations
    transformation = get_transformation(sdata[image_key])

    # add the segmentation masks to the spatial data object
    sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
        segmentation_masks[0].squeeze(), transformations=None, dims=("y", "x")
    )
    set_transformation(sdata.labels[key_added], transformation)

    if copy:
        return sdata


def astir(
    sdata,
    marker_dict: dict,
    table_key=SDLayers.TABLE,
    threshold: float = 0,
    seed: int = 42,
    learning_rate: float = 0.001,
    batch_size: float = 64,
    n_init: int = 5,
    n_init_epochs: int = 5,
    max_epochs: int = 500,
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
    copy: bool = False,
    **kwargs,
):
    """
    This function applies the ASTIR algorithm to predict cell types based on the expression matrix.
    It extracts the expression matrix from the spatialdata object, applies the ASTIR algorithm,
    and adds the predicted cell types to the spatialdata object.
    The predicted cell types are stored in the obs attribute of the AnnData object in the tables attribute of the spatialdata object.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
        marker_dict (dict): A dictionary containing the marker genes for each cell type.
        table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
        threshold (float, optional): The threshold value to be used for the ASTIR algorithm. Defaults to 0.
        seed (int, optional): The random seed to be used for the ASTIR algorithm. Defaults to 42.
        learning_rate (float, optional): The learning rate to be used for the ASTIR algorithm. Defaults to 0.001.
        batch_size (float, optional): The batch size to be used for the ASTIR algorithm. Defaults to 64.
        n_init (int, optional): The number of initializations to be used for the ASTIR algorithm. Defaults to 5.
        n_init_epochs (int, optional): The number of initial epochs to be used for the ASTIR algorithm. Defaults to 5.
        max_epochs (int, optional): The maximum number of epochs to be used for the ASTIR algorithm. Defaults to 500.
        cell_id_col (str, optional): The name of the column containing the cell IDs in the expression matrix. Defaults to "cell_id".
        cell_type_col (str, optional): The name of the column containing the cell types in the expression matrix. Defaults to "cell_type".
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the ASTIR algorithm.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    adata = _process_adata(sdata, table_key=table_key)
    expression_df = adata.to_df()

    assigned_cell_types = _astir(
        expression_df=expression_df,
        marker_dict=marker_dict,
        threshold=threshold,
        seed=seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_init=n_init,
        n_init_epochs=n_init_epochs,
        max_epochs=max_epochs,
        cell_id_col=cell_id_col,
        cell_type_col=cell_type_col,
        **kwargs,
    )

    # merging the resulting dataframe to the adata object
    df = pd.DataFrame(adata.obs)
    df = df.merge(assigned_cell_types, left_on="id", right_on=cell_id_col, how="left")
    adata.obs = df.drop(columns=cell_id_col)

    if copy:
        return sdata


def spotiflow(
    sdata,
    channel: Optional[str] = None,
    image_key: str = SDLayers.IMAGE,
    segmentation_key: str = SDLayers.SEGMENTATION,
    table_key: str = SDLayers.TABLE,
    key_added: str = "spotiflow",
    key_added_suffix: str = "_spots",
    pretrained_name: str = "general",
    model_kwargs: Optional[dict] = {"verbose": False},
    predict_kwargs: Optional[dict] = {"verbose": False},
    copy: bool = False,
):
    """
    This function applies the Spotiflow algorithm to detect spots in the provided image data.
    It extracts the image data from the spatialdata object, applies the Spotiflow algorithm,
    and adds the detected spots to the spatialdata object.
    The detected spots are stored in the points attribute of the spatialdata object.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
        channel (Optional[str]): The channel(s) to be used for spot detection. If None, all channels will be used.
        image_key (str, optional): The key for the image data in the spatialdata object. Defaults to "image".
        segmentation_key (str, optional): The key for the segmentation mask in the spatialdata object. Defaults to "segmentation".
        table_key (str, optional): The key for the table in the spatialdata object. Defaults to "table".
        key_added (str, optional): The key under which the detected spots will be stored in the points attribute of the spatialdata object. Defaults to "spotiflow".
        key_added_suffix (str, optional): The suffix to be added to the channel names in the detected spots. Defaults to "_spots".
        pretrained_name (str, optional): The name of the pretrained model to be used for spot detection. Defaults to "general".
        model_kwargs (Optional[dict], optional): Additional keyword arguments to be passed to the Spotiflow model. Defaults to {'verbose': False}.
        predict_kwargs (Optional[dict], optional): Additional keyword arguments to be passed to the Spotiflow prediction method. Defaults to {'verbose': False}.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.

    Returns:
        spatialdata.SpatialData: The spatialdata object with the detected spots added to the points attribute.
    """
    import spatialdata as sd
    from spatialdata.transformations import get_transformation, set_transformation

    assert image_key in sdata, f"Image key {image_key} not found in the object."
    assert segmentation_key in sdata, f"Segmentation key {segmentation_key} not found in the object."
    assert table_key in sdata, f"Table key {table_key} not found in the object."

    if copy:
        sdata = cp.deepcopy(sdata)

    channels = _get_channels_spatialdata(channel)
    img = _process_image(sdata, channels=channel, image_key=image_key, key_added=None)
    adata = _process_adata(sdata, table_key=table_key)

    points_df = _spotiflow(
        img,
        channel_names=channels,
        pretrained_name=pretrained_name,
        model_kwargs=model_kwargs,
        predict_kwargs=predict_kwargs,
    )

    seg = _process_segmentation(sdata, segmentation_key=segmentation_key)
    yy = points_df["y"].round().astype(int).clip(0, seg.shape[0] - 1)
    xx = points_df["x"].round().astype(int).clip(0, seg.shape[1] - 1)
    points_df["cell_id"] = seg[yy, xx]
    points_df = points_df[points_df["cell_id"] > 0]  # drop background detections

    # --- counts per cell, per channel -------------------------------------
    counts = points_df.groupby(["cell_id", "channel"]).size().unstack(fill_value=0)
    counts.columns = [f"{ch}{key_added_suffix}" for ch in counts.columns]

    # --- counts per cell, per channel -------------------------------------
    counts = points_df.groupby(["cell_id", "channel"]).size().unstack(fill_value=0)
    counts.columns = [f"{ch}{key_added_suffix}" for ch in counts.columns]
    new_cols = counts.columns.tolist()  # remember which columns are new

    # get transformations
    transformation = get_transformation(sdata[image_key])

    # write points to spatialdata
    sdata.points[key_added] = sd.models.PointsModel.parse(points_df, coordinates={"x": "x", "y": "y"})
    set_transformation(sdata.points[key_added], transformation)

    # merge counts to existing obs
    obs = adata.obs
    obs = obs.merge(counts, left_on="id", right_index=True, how="left")
    obs[new_cols] = obs[new_cols].fillna(0).astype(int)  # only fill the newly merged columns

    sdata.tables[table_key].obs = obs

    if copy:
        return sdata


# === SPATIALPROTEOMICS ACCESSOR ===
@xr.register_dataset_accessor("tl")
class ToolAccessor:
    """The tool accessor enables the application of external tools such as StarDist or Astir."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def cellpose(
        self,
        channel: Optional[str] = None,
        key_added: str = Layers.SEGMENTATION,
        diameter: float = None,
        channel_settings: list = None,
        num_iterations: int = 2000,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        batch_size: int = 8,
        gpu: bool = True,
        model_type: str = "cyto3",  # cellpose < 4.0
        pretrained_model: str = "cpsam",  # cellpose 4.0
        postprocess_func: Callable = lambda x: x,
        return_diameters: bool = False,
        **kwargs,
    ):
        """
        Segment cells using Cellpose. Adds a layer to the spatialproteomics object
        with dimension (X, Y) or (C, X, Y) dependent on whether channel argument
        is specified or not.

        Parameters
        ----------
        channel : str, optional
            Channel to use for segmentation. If None, all channels are used for independent segmentation.
        key_added : str, optional
            Key to assign to the segmentation results.
        diameter : float, optional
            Expected cell diameter in pixels.
        channel_settings : List[int], optional
            Channels for Cellpose to use for segmentation. If [0, 0], independent segmentation is performed on all channels. If it is anything else (e. g. [1, 2]), joint segmentation is attempted.
        num_iterations : int, optional
            Maximum number of iterations for segmentation.
        cellprob_threshold : float, optional
            Threshold for cell probability.
        flow_threshold : float, optional
            Threshold for flow.
        batch_size : int, optional
            Batch size for segmentation.
        gpu : bool, optional
            Whether to use GPU for segmentation.
        model_type : str, optional
            Type of Cellpose model to use.
        pretrained_model : str, optional
            Pretrained model to use for Cellpose (4+).
        postprocess_func : Callable, optional
            Function to apply to the segmentation masks after prediction.
        return_diameters : bool, optional
            Whether to return the cell diameters.

        Returns
        -------
        xr.Dataset
            Dataset containing original data and segmentation mask.

        Notes
        -----
        This method requires the 'cellpose' package to be installed.
        """
        channels = _get_channels(self._obj, key_added, channel)

        all_masks, diams = _cellpose(
            self._obj.pp[channels]._image.values,
            diameter=diameter,
            channel_settings=channel_settings,
            num_iterations=num_iterations,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            batch_size=batch_size,
            gpu=gpu,
            model_type=model_type,
            pretrained_model=pretrained_model,
            postprocess_func=postprocess_func,
            **kwargs,
        )

        if all_masks.shape[0] == 1:
            all_masks = all_masks[0].squeeze()
        # if we segment on all of the channels, we need to add the channel dimension
        else:
            all_masks = np.stack(all_masks, 0)

        # if no segmentation exists yet, and no key_added was specified, we make this one the default
        if key_added == Layers.SEGMENTATION:
            if return_diameters:
                return self._obj.pp.add_segmentation(all_masks), diams
            return self._obj.pp.add_segmentation(all_masks)

        da = _convert_masks_to_data_array(self._obj, all_masks, key_added)

        if return_diameters:
            return xr.merge([self._obj, da], join="outer", compat="no_conflicts"), diams

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def stardist(
        self,
        channel: Optional[str] = None,
        key_added: str = Layers.SEGMENTATION,
        scale: float = 3,
        n_tiles: int = 12,
        normalize: bool = True,
        predict_big: bool = False,
        postprocess_func: Callable = lambda x: x,
        **kwargs,
    ) -> xr.Dataset:
        """
        Apply StarDist algorithm to perform instance segmentation on the nuclear image.

        Parameters
        ----------
        channel : str, optional
            Channel to use for segmentation. If None, all channels are used.
        key_added : str, optional
            Key to write the segmentation results to.
        scale : float, optional
            Scaling factor for the StarDist model (default is 3).
        n_tiles : int, optional
            Number of tiles to split the image into for prediction (default is 12).
        normalize : bool, optional
            Flag indicating whether to normalize the nuclear image (default is True).
        nuclear_channel : str, optional
            Name of the nuclear channel in the image (default is "DAPI").
        predict_big : bool, optional
            Flag indicating whether to use the 'predict_instances_big' method for large images (default is False).
        postprocess_func : Callable, optional
            Function to apply to the segmentation masks after prediction (default is lambda x: x).
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the StarDist prediction method.

        Returns
        -------
        obj : xr.Dataset
            Xarray dataset containing the segmentation mask and centroids.

        Raises
        ------
        ValueError
            If the object already contains a segmentation mask.

        """
        channels = _get_channels(self._obj, key_added, channel)

        all_masks = _stardist(
            self._obj.pp[channels]._image.values,
            scale=scale,
            n_tiles=n_tiles,
            normalize=normalize,
            predict_big=predict_big,
            postprocess_func=postprocess_func,
            **kwargs,
        )

        if all_masks.shape[0] == 1:
            all_masks = all_masks[0].squeeze()
        # if we segment on all of the channels, we need to add the channel dimension
        else:
            all_masks = np.stack(all_masks, 0)

        # if no segmentation exists yet, and no key_added was specified, we make this one the default
        if key_added == Layers.SEGMENTATION:
            return self._obj.pp.add_segmentation(all_masks)

        da = _convert_masks_to_data_array(self._obj, all_masks, key_added)

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def mesmer(
        self,
        key_added: str = Layers.SEGMENTATION,
        channel: Optional[List] = None,
        postprocess_func: Callable = lambda x: x,
        **kwargs,
    ):
        """
        Segment cells using Mesmer. Adds a layer to the spatialproteomics object
        with dimension (C, X, Y). Assumes C is two and has the order (nuclear, membrane).

        Parameters
        ----------
        key_added : str, optional
            Key to assign to the segmentation results.
        channel : List, optional
            Channel to use for segmentation. If None, all channels are used.
        postprocess_func : Callable, optional
            Function to apply to the segmentation masks after prediction.

        Returns
        -------
        xr.Dataset
            Dataset containing original data and segmentation mask.

        Notes
        -----
        This method requires the 'mesmer' package to be installed.
        """
        channels = _get_channels(self._obj, key_added, channel)

        assert (
            len(channels) == 2
        ), "Mesmer only supports two channels for segmentation. If two channels are provided, the first channel is assumed to be the nuclear channel and the second channel is assumed to be the membrane channel. You can set the channels using the 'channel' argument."

        all_masks = _mesmer(self._obj.pp[channels]._image.values, postprocess_func=postprocess_func, **kwargs)

        if all_masks.shape[0] == 1:
            all_masks = all_masks[0].squeeze()
        # if we segment on all of the channels, we need to add the channel dimension
        else:
            all_masks = np.stack(all_masks, 0)

        # if no segmentation exists yet, and no key_added was specified, we make this one the default
        if key_added == Layers.SEGMENTATION:
            return self._obj.pp.add_segmentation(all_masks)

        da = _convert_masks_to_data_array(self._obj, all_masks, key_added)

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def astir(
        self,
        marker_dict: dict,
        key: str = Layers.INTENSITY,
        threshold: float = 0,
        seed: int = 42,
        learning_rate: float = 0.001,
        batch_size: float = 64,
        n_init: int = 5,
        n_init_epochs: int = 5,
        max_epochs: int = 500,
        cell_id_col: str = "cell_id",
        cell_type_col: str = "cell_type",
        **kwargs,
    ):
        """
        This method predicts cell types from an expression matrix using the Astir algorithm.

        Parameters
        ----------
        marker_dict : dict
            Dictionary mapping markers to cell types. Can also include cell states. Example: {"cell_type": {'B': ['PAX5'], 'T': ['CD3'], 'Myeloid': ['CD11b']}}
        key : str, optional
            Layer to use as expression matrix.
        threshold : float, optional
            Certainty threshold for astir to assign a cell type. Defaults to 0.
        seed : int, optional
            Random seed. Defaults to 42.
        learning_rate : float, optional
            Learning rate. Defaults to 0.001.
        batch_size : float, optional
            Batch size. Defaults to 64.
        n_init : int, optional
            Number of initializations. Defaults to 5.
        n_init_epochs : int, optional
            Number of epochs for each initialization. Defaults to 5.
        max_epochs : int, optional
            Maximum number of epochs. Defaults to 500.
        cell_id_col : str, optional
            Column name for cell IDs. Defaults to "cell_id".
        cell_type_col : str, optional
            Column name for cell types. Defaults to "cell_type".

        Raises
        ------
        ValueError
            If no expression matrix was present or the image is not of type uint8.

        Returns
        -------
        DataArray
            A DataArray with the assigned cell types.
        """
        # check if there is an expression matrix
        if key not in self._obj:
            raise ValueError(
                f"No expression matrix with key {key} found in the object. Make sure to call pp.quantify first."
            )

        # raise an error if the image is of anything other than uint8
        if self._obj[Layers.IMAGE].dtype != "uint8":
            logger.warning(
                "The image is not of type uint8, which is required for astir to work properly. Use pp.convert_to_8bit() to convert the image to uint8. If you applied operations such as filtering, you may ignore this warning."
            )

        # warn the user if the input dict has the wrong format
        if "cell_type" not in marker_dict.keys():
            logger.warning(
                "Did not find 'cell_type' key in the marker_dict. Your dictionary should have the following structure: {'cell_type': {'B': ['PAX5'], 'T': ['CD3'], 'Myeloid': ['CD11b']}}."
            )

        # converting the xarray to a pandas dataframe to keep track of channel names and indices after running astir
        expression_df = self._obj.pp.get_layer_as_df(key)

        # running astir
        assigned_cell_types = _astir(
            expression_df=expression_df,
            marker_dict=marker_dict,
            threshold=threshold,
            seed=seed,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_init=n_init,
            n_init_epochs=n_init_epochs,
            max_epochs=max_epochs,
            cell_id_col=cell_id_col,
            cell_type_col=cell_type_col,
        )

        # adding the labels to the xarray object
        return self._obj.la.add_labels_from_dataframe(
            assigned_cell_types, cell_col=cell_id_col, label_col=cell_type_col
        )

    def spotiflow(
        self,
        channel: Optional[str] = None,
        image_key: str = Layers.IMAGE,
        segmentation_key: str = Layers.SEGMENTATION,
        key_added_suffix: str = "_spots",
        pretrained_name: str = "general",
        return_spots: bool = False,
        model_kwargs: Optional[dict] = {"verbose": False},
        predict_kwargs: Optional[dict] = {"verbose": False},
    ):
        """
        This method detects spots in the image using the Spotiflow algorithm and counts the number of spots per cell, per channel.

        Parameters
        ----------
        channel : str, optional
            Channel to use for spot detection. If None, all channels are used.
        image_key : str, optional
            Key to the image layer in the spatialproteomics object. Default is 'image'.
        segmentation_key : str, optional
            Key to the segmentation layer in the spatialproteomics object. Default is 'segmentation'.
        key_added_suffix : str, optional
            Suffix to add to the channel names in the resulting counts. Default is '_spots'.
        pretrained_name : str, optional
            Name of the pretrained model to use for spot detection. Default is 'general'.
        return_spots : bool, optional
            Whether to return the detected spots as a DataFrame. Default is False.
        model_kwargs : dict, optional
            Additional keyword arguments to pass to the Spotiflow model. Default is {'verbose': False}.
        predict_kwargs : dict, optional
            Additional keyword arguments to pass to the Spotiflow prediction method. Default is {'verbose': False}.

        Returns
        -------
        xr.Dataset
            Dataset containing the counts of spots per cell, per channel. If return_spots is True, also returns a DataFrame of the detected spots with their coordinates and assigned cell IDs.
        """
        assert image_key in self._obj, f"Image key {image_key} not found in the object."
        assert segmentation_key in self._obj, f"Segmentation key {segmentation_key} not found in the object."
        channels = _get_channels(self._obj, key_added=None, channel=channel)
        img = self._obj.pp[channels][image_key].values
        points_df = _spotiflow(
            img,
            channel_names=channels,
            pretrained_name=pretrained_name,
            model_kwargs=model_kwargs,
            predict_kwargs=predict_kwargs,
        )

        # assign each point to a cell via the segmentation mask
        seg = self._obj[segmentation_key].values
        yy = points_df["y"].round().astype(int).clip(0, seg.shape[0] - 1)
        xx = points_df["x"].round().astype(int).clip(0, seg.shape[1] - 1)
        points_df["cell_id"] = seg[yy, xx]
        points_df = points_df[points_df["cell_id"] > 0]  # drop background detections

        # count points per cell, per channel
        counts = (
            points_df.groupby(["cell_id", "channel"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=self._obj.cells.values, columns=channels, fill_value=0)
        )
        counts.columns = [f"{ch}{key_added_suffix}" for ch in counts.columns]

        if return_spots:
            return self._obj.pp.add_obs_from_dataframe(counts.reset_index(drop=True)), points_df

        return self._obj.pp.add_obs_from_dataframe(counts.reset_index(drop=True))

    def virtues(
        self,
        mode: str = "segment",
        protein_mapping: pd.DataFrame = None,
        marker_col: str = "marker_name",
        protein_id_col: str = "protein_id",
        image_key: str = Layers.IMAGE,
        model_name: str = "virtues-sp32",
        cache_dir=None,
        virtues_path=None,
        quantile: float = 0.99,
        mean=None,
        std=None,
        gaussian_blur: bool = True,
        gaussian_blur_sigma: float = 1.0,
        store_patch_embeddings: bool = False,
        tile_size: int = 128,
        segment_overlap: int = 32,
        cell_token_stride: int = 42,
        batch_size=None,
        segmentation_key_added: str = Layers.SEGMENTATION,
        cell_type_prediction_key_added: Optional[str] = None,
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
        mode : {"segment", "embed_cells"}, optional
            The VirTues workflow to run. ``"segment"`` stores a segmentation and optional pixel-wise cell type predictions.
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
            Overlap parameter in pixels for mode="segment"; defaults to 32. VirTues requests
            a total crop overlap of 2 * segment_overlap for images whose longest side is at most
            1500 pixels, and 2 * (segment_overlap + 20) for larger images, including a detection
            margin. In the latter case, segment_overlap also controls the instance-stitching offset.
            For tile_size=128 and segment_overlap=32, nominal crop strides are 64 and 24 pixels,
            respectively. More overlap increases the number of crops and computation.
        cell_token_stride : int, optional
            Crop stride used for cell embedding extraction.
        batch_size : int, optional
            Batch size for segmentation or chunk size for cell embedding extraction.
        segmentation_key_added : str, optional
            Key for the VirTues segmentation output; defaults to canonical "_segmentation".
        cell_type_prediction_key_added : str or None, optional
            Store pixel-wise class predictions under this key; None skips prediction storage.
            Predictions suppress "None" and "Unknown" before argmax and are not assigned to VirTues cells
            or set to background outside them. Assign predictions to canonical cells explicitly with
            project_virtues_cell_types_to_existing_segmentation().
        cell_embedding_key_added : str, optional
            Key for the VirTues cell embedding output.
        marker_embedding_key_added : str, optional
            Key for marker embeddings.
        patch_embedding_key_added : str, optional
            Key for optional patch embeddings.
        patch_coord_key_added : str, optional
            Key for optional patch coordinates.
        overwrite : bool, optional
            If True, overwrite custom output layers. An existing canonical segmentation cannot be replaced.
        device : str, optional
            Device used for model execution. Only ``"cuda"`` is supported in v0.1.

        Returns
        -------
        xr.Dataset
            Dataset containing the original data and the requested VirTues outputs. In ``mode="segment"``,
            this includes the segmentation and pixel-wise cell type predictions only when requested. In ``mode="embed_cells"``, this
            includes cell-level embeddings.

        Notes
        -----
        UniProt IDs must be supplied explicitly in ``protein_mapping``. Optional VirTues dependencies
        are imported only when this method is called.

        In mode="segment", segmentation_key_added defaults to "_segmentation". Writing to this key
        establishes canonical cell IDs and geometric observations. If "_segmentation" already exists,
        the call raises an error even when overwrite=True; use a different output key for an alternative mask.

        A custom segmentation key stores only an image-shaped mask. It does not change canonical
        segmentation, cell IDs, or observations, and does not establish cells when no canonical segmentation
        exists. An existing custom output can be replaced with overwrite=True.

        Pixel-wise cell type predictions are stored only when cell_type_prediction_key_added is supplied. Segmentation never
        automatically adds cell type annotations. Cell embedding extraction always uses canonical cells.
        """
        from .virtues_utils import _virtues, validate_and_normalize_protein_mapping
        from .virtues_backends import (
            _get_stored_virtues_marker_embeddings,
            _get_virtues_canonical_segmentation,
            _validate_virtues_arguments,
            add_marker_embeddings,
            add_virtues_segmentation_outputs,
            add_virtues_embedding_outputs,
            add_patch_embeddings,
        )

        obj = self._obj
        _validate_virtues_arguments(
            obj, 
            mode, 
            image_key, 
            store_patch_embeddings, 
            segmentation_key_added, 
            cell_type_prediction_key_added,
            cell_embedding_key_added, 
            marker_embedding_key_added, 
            patch_embedding_key_added, 
            patch_coord_key_added, 
            overwrite,
        )
        channels = obj.coords[Dims.CHANNELS].values.astype(str).tolist()
        mapping = validate_and_normalize_protein_mapping(
            channels, 
            protein_mapping, 
            marker_col, 
            protein_id_col, 
            model_name
        )
        marker_embeddings = _get_stored_virtues_marker_embeddings(
            obj, 
            mapping, 
            marker_embedding_key_added, 
            overwrite=overwrite
        )
        segmentation = _get_virtues_canonical_segmentation(obj)[0] if mode == "embed_cells" else None

        image = obj[image_key].sel({Dims.CHANNELS: mapping["marker_name"].tolist()}).values

        result = _virtues(
            image, 
            mapping, 
            mode=mode, 
            segmentation=segmentation, 
            marker_embeddings=marker_embeddings,
            model_name=model_name, 
            cache_dir=cache_dir, 
            virtues_path=virtues_path, 
            quantile=quantile,
            mean=mean, 
            std=std, 
            gaussian_blur=gaussian_blur, 
            gaussian_blur_sigma=gaussian_blur_sigma,
            tile_size=tile_size, 
            segment_overlap=segment_overlap, 
            cell_token_stride=cell_token_stride,
            batch_size=batch_size, 
            return_cell_type_prediction=(mode == "segment" and cell_type_prediction_key_added is not None),
            return_patch_embeddings=store_patch_embeddings, 
            device=device,
        )
        outputs = result["outputs"]

        if result["new_marker_embeddings"] is not None:
            obj = add_marker_embeddings(
                obj, 
                result["new_marker_embeddings"], 
                mapping, 
                marker_embedding_key_added, 
                overwrite=overwrite
            )

        if mode == "segment":
            attrs = {
                **outputs["segmentation_metadata"], 
                "image_key": image_key
            }
            obj = add_virtues_segmentation_outputs(
                obj, 
                outputs["segmentation"], 
                outputs["cell_type_prediction"], 
                segmentation_key_added,
                cell_type_prediction_key_added, 
                attrs, 
                overwrite=overwrite,
            )
        else:
            attrs = {
                **outputs["cell_embedding_metadata"], 
                "image_key": image_key, 
                "segmentation_key_used": Layers.SEGMENTATION
            }
            obj = add_virtues_embedding_outputs(
                obj, 
                outputs["cell_ids"], 
                outputs["cell_embeddings"], 
                cell_embedding_key_added, 
                attrs, 
                overwrite=overwrite
            )

        if store_patch_embeddings:
            attrs = {
                **outputs["patch_metadata"], 
                "image_key": image_key
            }
            obj = add_patch_embeddings(
                obj, 
                outputs["patch_embeddings"], 
                outputs["patch_coords_yx"], 
                patch_embedding_key_added,
                patch_coord_key_added, 
                attrs, 
                overwrite=overwrite,
            )
        return obj

    def virtues_aggregate_patch_embeddings(
        self,
        patch_embedding_key: str = "_virtues_patch_embeddings",
        patch_coord_key: str = "_virtues_patch_coords",
        key_added: str = "_virtues_cell_embeddings",
        overwrite: bool = False,
    ) -> xr.Dataset:
        """
        Aggregate stored VirTues patch embeddings into cell embeddings. Follows the logic of the VirTues tutorials, 
        where patch embeddings are assigned to segmented cells based on a weighted average 
        with weights determined according to the number of pixels of the patch that overlap with the cell.

        Parameters
        ----------
        patch_embedding_key : str, optional
            The key of the stored patch embeddings.
        patch_coord_key : str, optional
            The key of the stored patch top-left coordinates.
        key_added : str, optional
            The key under which aggregated embeddings of canonical "_segmentation" cells will be stored.
        overwrite : bool, optional
            If True, overwrite an existing cell embedding layer.

        Returns
        -------
        xr.Dataset
            Dataset containing the original data and cell embeddings aggregated from the stored VirTues
            patch embeddings.

        Notes
        -----
        Aggregation always targets canonical "_segmentation" cells; alternate masks do not define cells.
        Each token contributes to every overlapping cell, weighted by the number of overlapping pixels.
        """
        from .virtues_utils import TOKEN_PATCH_SIZE_PX, _virtues_aggregate_patch_embeddings
        from .virtues_backends import (
            VIRTUES_PATCHES_DIM, TOKEN_Y_DIM, TOKEN_X_DIM, VIRTUES_EMBEDDING_DIM, YX_DIM,
            _check_key_available,
            _get_virtues_canonical_segmentation,
            add_virtues_embedding_outputs,
        )

        obj = self._obj

        for key in [patch_embedding_key, patch_coord_key]:
            if key not in obj:
                raise KeyError(f"Layer {key!r} not found.")
        _check_key_available(obj, key_added, overwrite)
        if key_added == Layers.SEGMENTATION:
            raise ValueError("Cell embeddings cannot replace canonical segmentation.")

        segmentation, cells = _get_virtues_canonical_segmentation(obj)

        # get patch embeddings and coordinates and validate their dimensions and attributes
        patch_da = obj[patch_embedding_key]
        coords_da = obj[patch_coord_key]
        if patch_da.dims != (VIRTUES_PATCHES_DIM, TOKEN_Y_DIM, TOKEN_X_DIM, VIRTUES_EMBEDDING_DIM):
            raise ValueError("Patch embeddings must use crop, token_y, token_x, and embedding dimensions.")
        if coords_da.dims != (VIRTUES_PATCHES_DIM, YX_DIM) or coords_da.coords[YX_DIM].values.tolist() != ["y", "x"]:
            raise ValueError("Patch coordinates must use [crop, yx] dimensions in top-left (y, x) order.")
        if coords_da.attrs.get("paired_embedding_key", patch_embedding_key) != patch_embedding_key:
            raise ValueError("Patch coordinates are paired with a different embedding key.")

        required_attrs = [
            "tile_size_px",
            "token_patch_size_px",
            "token_stride_px",
        ]
        missing_attrs = [attr for attr in required_attrs if attr not in patch_da.attrs]
        if missing_attrs:
            raise ValueError(f"Patch embeddings are missing required attrs: {missing_attrs}.")

        if patch_da.attrs["token_patch_size_px"] != TOKEN_PATCH_SIZE_PX or patch_da.attrs["token_stride_px"] != TOKEN_PATCH_SIZE_PX:
            raise ValueError("Stored token geometry must match the fixed 8-pixel VirTues token size and stride.")

        values = _virtues_aggregate_patch_embeddings(
            patch_da.values, 
            coords_da.values, 
            segmentation, 
            cells, 
            patch_da.attrs["tile_size_px"],
        )
        attrs = {
            "source": f"aggregate:{patch_da.attrs.get('source', 'unknown')}",
            "patch_embedding_key": patch_embedding_key,
            "patch_coord_key": patch_coord_key,
            "segmentation_key": Layers.SEGMENTATION,
        }
        return add_virtues_embedding_outputs(
            obj, 
            cells, 
            values, 
            key_added, 
            attrs, 
            overwrite=overwrite
        )

    def project_virtues_cell_types_to_existing_segmentation(
        self,
        cell_type_prediction: np.ndarray,
        cell_type_label_key_added: str = "_virtues_cell_type",
        overwrite: bool = False,
    ) -> xr.Dataset:
        """
        Project VirTues cell type predictions onto an existing segmentation.

        Parameters
        ----------
        cell_type_prediction : np.ndarray
            Pixel-wise VirTues class predictions before assignment to an instance segmentation.
        cell_type_label_key_added : str, optional
            The column added to ``_la_layers``.
        overwrite : bool, optional
            If True, overwrite an existing label layer column with the same name.

        Returns
        -------
        xr.Dataset
            Dataset containing the original data and a ``_la_layers`` column with projected VirTues cell types.

        Notes
        -----
        Projection is explicit and always targets canonical "_segmentation" cells. To annotate another
        segmentation population, first make that segmentation canonical in a separate object.
        For each existing cell, the assigned VirTues cell type is the most frequent value of
        ``cell_type_prediction`` under that cell's segmentation mask. Use the pixel-wise predictions rather
        than a mask already assigned to VirTues cells. All supplied class IDs participate, including 0
        if present. Ties follow ``np.bincount(...).argmax()``
        and therefore choose the lowest tied integer cell type ID.
        """
        from .virtues_backends import _get_virtues_canonical_segmentation
        from .virtues_utils import _project_virtues_cell_types_to_segmentation

        obj = self._obj
        segmentation, cells = _get_virtues_canonical_segmentation(obj)
        if Layers.LA_LAYERS in obj:
            existing = obj[Layers.LA_LAYERS].to_pandas()
            if cell_type_label_key_added in existing.columns and not overwrite:
                raise ValueError(f"Label layer column {cell_type_label_key_added} already exists.")
        else:
            existing = pd.DataFrame(index=cells)
        cell_types = _project_virtues_cell_types_to_segmentation(cell_type_prediction, segmentation, cells)
        existing[cell_type_label_key_added] = pd.Series(cell_types, index=cells)
        if Layers.LA_LAYERS in obj:
            obj = obj.drop_vars(Layers.LA_LAYERS)
        return obj.pp.add_layer_from_dataframe(existing, key_added=Layers.LA_LAYERS)

    def convert_to_anndata(
        self,
        expression_matrix_key: str = Layers.INTENSITY,
        obs_key: str = Layers.OBS,
        additional_layers: Optional[dict] = None,
        additional_obsm: Optional[dict] = None,
        additional_uns: Optional[dict] = None,
    ):
        """
        Convert the spatialproteomics object to an anndata.AnnData object. The resulting AnnData object does not store the original image or segmentation mask.

        Parameters
        ----------
        expression_matrix_key : str, optional
            The key of the expression matrix in the spatialproteomics object. Default is '_intensity'.
        obs_key : str, optional
            The key of the observation data in the spatialproteomics object. Default is '_obs'.
        additional_layers : dict, optional
            Additional layers to include in the anndata.AnnData object. The keys are the names of the layers and the values are the corresponding keys in the spatialproteomics object.
        additional_obsm : dict, optional
            Additional obsm data to include in the anndata.AnnData object. The keys are the names of the obsm data and the values are the corresponding keys in the spatialproteomics object.
        additional_uns : dict, optional
            Additional uns data to include in the anndata.AnnData object. The keys are the names of the uns data and the values are the corresponding keys in the spatialproteomics object.

        Returns
        -------
        adata : anndata.AnnData
            The converted anndata.AnnData object.

        Raises
        ------
        AssertionError
            If additional layers, obsm entries, or uns entries are not found in the spatialproteomics object, or if additional layers or obsm entries have incompatible shapes.

        Notes
        -----
        - The expression matrix is extracted from the spatialproteomics object using the provided expression matrix key.
        - If additional layers are specified, they are extracted from the spatialproteomics object and added to the anndata.AnnData object.
        - If obs_key is present in the spatialproteomics object, it is used to create the obs DataFrame of the anndata.AnnData object.
        - If additional_uns is specified, the corresponding uns data is extracted from the spatialproteomics object and added to the anndata.AnnData object.
        """
        import anndata

        # if there is no expression matrix, we return an empty anndata object
        if expression_matrix_key not in self._obj:
            return anndata.AnnData()

        expression_matrix = self._obj[expression_matrix_key].values
        adata = anndata.AnnData(expression_matrix)
        if additional_layers:
            for key, layer in additional_layers.items():
                # checking that the additional layer is present in the object
                assert layer in self._obj, f"Layer {layer} not found in the object."
                # checking that layer has the same shape as the expression matrix
                assert self._obj[layer].shape == expression_matrix.shape, (
                    f"Layer {layer} has shape {self._obj[layer].shape}, but AnnData layers must match "
                    f"the shape of the expression matrix {expression_matrix.shape}."
                )
                adata.layers[key] = self._obj[layer].values
        adata.var_names = self._obj.coords[Dims.CHANNELS].values

        if obs_key in self._obj:
            adata.obs = self._obj.pp.get_layer_as_df(obs_key, idx_to_str=True)

            # converting the labels into categorical
            if Dims.LABELS in self._obj.dims:
                adata.obs[Features.LABELS] = adata.obs[Features.LABELS].astype("category")

            # if we have labels and colors for them, we add them to the anndata object
            if Dims.LABELS in self._obj.dims and Layers.LA_PROPERTIES in self._obj:
                properties = self._obj.pp.get_layer_as_df(Layers.LA_PROPERTIES)
                if Props.COLOR in properties.columns:
                    # putting it into the anndata object
                    adata.uns[f"{Features.LABELS}_colors"] = properties[Props.COLOR].values
                # if the labels are in there, we want to use the order of them for the categories
                if Props.NAME in properties.columns:
                    adata.obs[Features.LABELS] = adata.obs[Features.LABELS].cat.set_categories(
                        properties[Props.NAME].values, ordered=True
                    )

            # to be compatible with squidpy out of the box, a spatial key is added to obsm if possible
            if Features.X in adata.obs and Features.Y in adata.obs:
                adata.obs[Features.X] = adata.obs[Features.X].astype(float)
                adata.obs[Features.Y] = adata.obs[Features.Y].astype(float)
                adata.obsm["spatial"] = np.array(adata.obs[[Features.X, Features.Y]])

        if additional_obsm:
            for key, layer in additional_obsm.items():
                # checking that the additional obsm source layer is present in the object
                assert layer in self._obj, f"Layer {layer} not found in the object."
                # checking that the additional obsm layer has the same number of observations as the expression matrix
                assert self._obj[layer].shape[0] == expression_matrix.shape[0], (
                    f"Layer {layer} has {self._obj[layer].shape[0]} observations, but the expression "
                    f"matrix has {expression_matrix.shape[0]} observations. AnnData obsm entries must match "
                    "the number of observations in the expression matrix."
                )
                adata.obsm[key] = self._obj[layer].values

        if additional_uns:
            for key, layer in additional_uns.items():
                # checking that the additional uns source layer is present in the object
                assert layer in self._obj, f"Layer {layer} not found in the object."
                adata.uns[key] = self._obj.pp.get_layer_as_df(layer)

        return adata

    def convert_to_spatialdata(
        self, image_key: str = Layers.IMAGE, segmentation_key: str = Layers.SEGMENTATION, **kwargs
    ):
        """
        Convert the spatialproteomics object to a spatialdata object.

        Parameters:
            image_key (str): The key of the image data in the object. Defaults to Layers.IMAGE.
            segmentation_key (str): The key of the segmentation data in the object. Defaults to Layers.SEGMENTATION.
            **kwargs: Additional keyword arguments to be passed to the convert_to_anndata method.

        Returns:
            spatial_data_object (spatialdata.SpatialData): The converted spatialdata object.
        """
        import spatialdata
        from spatialdata.transformations import set_transformation

        store_segmentation = False
        store_adata = False

        markers = self._obj.coords[Dims.CHANNELS].values
        image = spatialdata.models.Image2DModel.parse(
            self._obj[image_key].values, transformations=None, dims=("c", "x", "y"), c_coords=markers
        )
        if Dims.CELLS in self._obj.coords:
            cells = self._obj.coords[Dims.CELLS].values
            segmentation = spatialdata.models.Labels2DModel.parse(
                self._obj[segmentation_key].values, transformations=None, dims=("x", "y")
            )
            store_segmentation = True

        adata = self._obj.tl.convert_to_anndata(**kwargs)

        if adata.X is not None:
            # the anndata object within the spatialdata object requires some additional slots, which are created here
            adata.uns["spatialdata_attrs"] = {
                "region": SDLayers.SEGMENTATION,
                "region_key": "region",
                "instance_key": "id",
            }

            if adata.obs is not None:
                adata.obs["id"] = cells
                adata.obs["region"] = pd.Series([SDLayers.SEGMENTATION] * adata.n_obs, index=adata.obs.index).astype(
                    pd.api.types.CategoricalDtype(categories=[SDLayers.SEGMENTATION])
                )
            else:
                obs_df = pd.DataFrame(
                    {
                        "id": cells,
                        "region": pd.Series([SDLayers.SEGMENTATION] * len(cells)).astype(
                            pd.api.types.CategoricalDtype(categories=[SDLayers.SEGMENTATION])
                        ),
                    }
                )
                adata.obs = obs_df
            # anndata insists that the obs_names are strings, and will throw a warning if they are not
            # to be consistent with their examples, we add the "Cell_" prefix here
            adata.obs_names = [f"Cell_{x}" for x in cells]

            # transforming the index to string
            adata.obs.index = [str(x) for x in adata.obs.index]
            store_adata = True

        # Your known crop origin in the global coordinate space
        transformation = _compute_transformation(self._obj.coords["x"].values, self._obj.coords["y"].values)

        # storing only the info that is present in the object (image/segmentation/anndata)
        # we can only have an anndata if we also have a segmentation mask
        if store_segmentation:
            if store_adata:
                spatial_data_object = spatialdata.SpatialData(
                    images={SDLayers.IMAGE: image},
                    labels={SDLayers.SEGMENTATION: segmentation},
                    tables={SDLayers.TABLE: adata},
                )
            else:
                spatial_data_object = spatialdata.SpatialData(
                    images={SDLayers.IMAGE: image}, labels={SDLayers.SEGMENTATION: segmentation}
                )

            set_transformation(
                spatial_data_object.labels[SDLayers.SEGMENTATION], transformation, to_coordinate_system="global"
            )
        else:
            spatial_data_object = spatialdata.SpatialData(images={SDLayers.IMAGE: image})

        set_transformation(spatial_data_object.images[SDLayers.IMAGE], transformation, to_coordinate_system="global")

        return spatial_data_object
