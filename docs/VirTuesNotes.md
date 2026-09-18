# Applying the VirTues foundation model 

## Proposed notebook structure 

- brief intro paragraphing citing the [VirTues paper](https://www.nature.com/articles/s41586-026-10884-y#)
- explanation of how it works: 
    - VirTues uses an internal marker embedding, to take into account that some markers might be biologically more related than others, etc. 
    - this means it needs to know which marker channels are used in a given image, and assign a numerical embedding matrix to them that captures their biological relationships somehow 
    - how they do this: map each marker to a Uniprot protein ID, then use a protein lanauge model (ESM) to embed the respective protein's sequence, obtained from the `fasta` file
    - the original VirTues code stores these `fasta` files and ESM embeddings. Here it is computed on the fly and the final embedding matrix is added to the `sp` object directly. 
    - (Side note: this avoids downloading things without the user's explicit permission and creating temporary directories, but means that we cannot directly use the higher-level `virtues` wrappers for this functionality. I pulled out the lower-level functions and used them in our own `sp`-specific wrappers.)
    - images are preprocessed as follows: 
        - a pre-specified quantile `q` (`virtues` default: 0.99) for each marker channels over the entire image is computed, and used for clipping the image to the `[0, q]` range.
        - a `log1p` transformation is applied 
        - the image is standardised, i.e., for each marker channel and each image, the mean of that marker across the entire dataset is subtracted, and the values are divided by the standard deviation of that marker across the entire dataset 
        - this means that we need datast-level marker means and standard deviations. These have to be calculated in advance based on all quantile-clipped images and passed to `virtues`. As a fallback, the implementation uses the image-level mean and standard deviation after clipping. However, this can lead to markers with zero mean or std (which should then be ignored anyway, but at the moment the function will simply raise an error, which is probably not the most user-friendly behaviour - see TODOs below)
        - finally, optionally, a Gaussian blur with prespecified parameters is applied
    - the specified `virtues` model is loaded from the [huggingface checkpoint](https://huggingface.co/bunnelab/virtues) (default: the `virtues-sp32 model trained on data from diverse technologies incl. CODEX, see [here](https://github.com/bunnelab/virtues/tree/main#models) for an overview of their available models)
    - this model is then used to embed the image, tiled into crops which are then more finely chopped into patches (size: internal spatial token resolution of `virtues`, 8x8 pixel)
    - these patch embeddings can then be fed into a pre-trained U-Net to obtain a segmentation and a broad cell type annotation, or convolved with a segmentation mask to obtain cell-level embeddings 
- loading one example image (the one from the other `sp` tutorials, probably), visualising some raw marker channels as in the the cell type prediction tutorial, for instance 
- load the the marker-protein ID mapping and show how it has to look like
- load cohort-level mean and std and explain about this 
- run `sp.tl.virtues(mode= 'segment')` 
- show plots of the `virtues` segmentation 
- show their cell type dictionary and the computed cell type annotations, maybe with a brief explanation of how it works (pixel-wise label mask, aggregated by taking the softmax cell type assignment of each pixel, then aggregating over all pixel of a cell according to the segmentation mask and taking the most frequently occuring one)
- show how their cell type annotation can also be projected onto another already existing segmentation, using `project_virtues_cell_types_to_existing_segmentation`
- show how to compute cell embeddings based on that, if the patch embeddings are stored from the segment run (`store_patch_embeddings=True`), using `virtues_aggregate_patch_embeddings`
- show, a second workflow, starting again from the same image, and running the cell embedding mode directly, which requires an already existing segmentation mask


## TODO 

- In preprocessing: Check and handle markers with zero mean/std when using image-level stats as fallback. At the moment there is probably just going to be an unspecific error message. I think these markers should just not used for this `virtues` run. But a warning should be printed, and we need to take care to mask/subset properly such that the markers don't get lost from the image altogether, but are just ignored for the `virtues` run. 
- check whether `embedding` vs. `embeddings` is used consistently in function naming
- add proper tests 
- add docstrings
- tests for VirTues fork
- add version that is compatible with `SpatialData` 

## Notes

- When applying it on multiple images which all share the same marker channels, it would be inefficient to re-compute the marker embedding for each image. The current implementation looks for a `marker_embedding` layer in the `sp` object, so a possible workflow would be to use just the `get_or_create_marker_embeddings` function once in advance, store the mapping in an array, and then add that as layer to each image via `add_marker_embeddings` as the first step of the loop over all images. 
- user-defined `marker_col` and `protein_id_col` become normalised to `'marker_name'` and `'protein_id'`
- current simplification: marker-protein mapping has to be supplied as `pandas.DataFrame` explicitly
- all the metadata attributes (what information to save from a `virtues` run, tile size, batch size, model name, etc.) are just choices I made -- not sure what is really best here, please review and adjust where you consider necessary

## Sync local VirTues checkout with my GitHub fork

Fork: `https://github.com/maren-ha/virtues`

```bash
cd /path/to/virtues

# Check current state
git status
git remote -v

# Add my fork if needed
git remote add myfork https://github.com/maren-ha/virtues.git

# Fetch fork branches
git fetch myfork

# Check out my spatialproteomics branch from the fork
git switch -c spatialproteomics-patch-embeddings myfork/spatialproteomics-patch-embeddings
# If the branch already exists locally:
# git switch spatialproteomics-patch-embeddings
# git pull --ff-only myfork spatialproteomics-patch-embeddings

# Push local commits back to my fork
git push myfork spatialproteomics-patch-embeddings

# If I amended/rebased the branch:
# git push --force-with-lease myfork spatialproteomics-patch-embeddings
```