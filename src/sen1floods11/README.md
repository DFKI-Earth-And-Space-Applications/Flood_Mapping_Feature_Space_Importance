# Sen1Floods11

Sen1Floods11 is a dataset that has been proposed to train flood mapping algorithms on Sentinel-1 and Sentinel-2 data,
which has been collected across all major biomes. It features 4,384 weakly labeled and 446 hand labeled 512X512 images
(chips). You can find their paper on [ieeexplore](https://ieeexplore.ieee.org/document/9150760).

## Structure of this Repo

In this folder a simple API for accessing the Sen1Floods11 dataset is provided in `dataset.py`. and `convert_datadings_seperately.py` is a script used to convert the data
into the much more efficient datadings format.

## Downloading the dataset

The following will assume that you have [gsutil](https://cloud.google.com/storage/docs/gsutil_install) installed.

First of all, note that the 1.1 version of the dataset as described in the
[Sen1Floods11-Repo](https://github.com/cloudtostreet/Sen1Floods11) (as well as their sample code) is incomplete - it
does not contain the Sentinel-2 weakly labeled train data. If you don't need to use this, you can simply run
`gsutil -m rsync -r gs://sen1floods11 [<Your target folder>]` to download everything in one go (which will be around
~14GB, however there is a lot of unnecessary (if you don't need information such as the exact geo-coordinates) small
catalogue files). To install the individual parts (or add the Sentinel-2 weakly-labeled data) check the following
commands (you need create your target folders beforehand as otherwise you'll
get `"[<Your folder>] does not name a directory, bucket, or bucket subdir"` error):

### Hand-Labeled data

**Train-Val-Test split
information:** `gsutil -m rsync -r gs://sen1floods11/v1.1/splits/flood_handlabeled [<Your split target folder>]` (This
contains the csv files used for identifying which files belong to which of the pre-defined splits of the dataset)

**Labels (
2.43Mb):** `gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/ [<Your label target folder>]`

**Sentinel-1 images (
695Mb):** `gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/ [<Your S1Hand target folder>]`

**Sentinel-2 images (
971Mb):** `gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/ [<Your S2Hand target folder>]`

There is also a folder `S1OtsuLabelHand` here which is not used at all by the pre-defined splits (it contains Otsu-
Thresholding based labels for the hand labeled data). Therefore, this folder and the `JRCWaterHand` folder from the v1.1
dataset are not listed here as they are not (less) relevant. If you need the `S1OtsuLabelHand` folder, it should be
processable using the provided methods.

### Weakly-Labeled data

**Sentinel-1 threshold labels (26.1Mb):**
`gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/[<Your S1WeakLabel target folder>]`

**Sentinel-1 images (6.81Gb):** `gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1Weak/
[<Your S1Weak target folder>]`

**Sentinel-2 threshold labels (39.7Mb):**
`gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S2IndexLabelWeak/[<Your S2WeakLabel target folder>]`

**Sentinel-2 images (23.9Gb):** `gsutil -m rsync -r gs://cnn_chips/S2_NoQC/ [<Your S2Weak target folder>]`

### Metadata

Singe files such as the general metadata cannot be transferred via gsutil (so you either have to sync the complete
dataset or download it via the [browser](https://console.cloud.google.com/)). However, metadata for the individual chips
is found in the catalogue `gsutil -m rsync -r gs://sen1floods11/v1.1/catalog
[<Your catalogue target folder>]`.

## Using the sen1floods11 API

This section gives an overview of how to access the dataset in code. For convenience this section contains a
"quickstart guide" - for more information please refer to the in-code python documentation.

### Required packages

- `rasterio` or `tifffile` (Notice that rasterio is only available up to python 3.7, but it is the package used in the
  original paper)
- `numpy`
- `pytorch`

### Loading the data

You can construct a pytorch dataset using the `read_labels_only`, `read_s1_images` and `read_s2_images`. As the names
suggest these read either only the labels or some s1 or s2 images (for which reading labels is optional - just keep the
default `None` for the `label_folder` argument). If you want to use a specific train-test split (such as the ones
provided for the hand-labeled data), just specify the path to the csv file using the `split_csv_file` argument
(Notice however, that the Sen1Floods11 dataset only provides splits for the sentinel-1 hand labeled data and not the
sentinel-2 data). The folders provided to the `label_folder` and `data_folder` arguments should point to the same
folders that you specified as target folders when downloading the individual datasubsets.

The returned datasets return the data as pytorch tensors with shape `[C, H, W]` . If multiple datasubsets are combined
(such as labels and corresponding data) the corresponding tensors are returned in a tuple. If you want to manually merge
some datasubsets, you can use the `MergingDatasetWrapper` which will create these non-nested tuples (see in-code-docs
for a more detailed description of what I mean with non-nested). Notice also that all dataset access the data files
on-query (and will thus only detect errors on access), if you want to pre-load the data, just wrap the results in
an `InMemoryDataset` which will load the *complete* dataset during construction.

The returned data is of type float32 for both the Sentinel-1 and Sentinel-2 images and of type int8 for the masks. Note
that uint8 is not used as in the sample code provided by the paper, this is because it seems more straight forward to
have the no-information masks provided in the hand-labeled data as -1 instead of 255.

As the Sen1Floods11 dataset does not provide the splits for the Sentinel-2 hand labeled data, you can create these based
on the Sentinel-1 splits by calling the `create_s2_split_files` function or invoking `dataset.py` with a path to the
folder containing the split files as an argument.

## Converting to Datadings

In order to use this data for the experiments in the neighbouring folders, you will have to create datadings-msgpack files first (or otherwise consistenly add no-datadings arguments
to all data-using calls, but this is not recommended). For this run `convert_datadings_seperately.py` with the folder you downloaded your Sen1Floods11 files to and another folder 
depicting where the datadings files should be stored. The latter will then be your `data_folder` when running experiments.

# For convenience: Here the data info from their repo

Except for the bit and resolution information (that was fetched from the
[Sentinel-2 User Guide](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/resolutions/radiometric))
everything below is copied from the [Sen1Floods11-Repo](https://github.com/cloudtostreet/Sen1Floods11). Notice though
that the Sentinel-2 images are converted to float32 when loading the dataset.

Each file follows the naming scheme EVENT_CHIPID_LAYER.tif (e.g. `Bolivia_103757_S2Hand.tif`). Chip IDs are unique, and
not shared between events. Events are named by country and further information on each event (including dates) can be
found in the event metadata below. Each layer has a separate GeoTIFF, and can contain multiple bands in a stacked
GeoTIFF. All images are projected to WGS 84 (`EPSG:4326`) at 10 m ground resolution.

| Layer | Description                                                                                                                                              | Values                                                  | Format                                           | Bands                                                                                                                                                                                                                                                                           |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| QC    | Hand labeled chips containing ground truth                                                                                                               | -1: No Data / Not Valid <br> 0: Not Water <br> 1: Water | GeoTIFF <br> 512 x 512 <br> 1 band <br> Int16    | 0: QC                                                                                                                                                                                                                                                                           |
| S1    | Raw Sentinel-1 imagery. <br> IW mode, GRD product <br> See [here](https://developers.google.com/earth-engine/sentinel1) for information on preprocessing | Unit: dB                                                | GeoTIFF <br> 512 x 512 <br> 2 bands <br> Float32 | 0: VV <br> 1: VH                                                                                                                                                                                                                                                                |
| S2    | Raw Sentinel-2 MSI Level-1C imagery <br> Contains all spectral bands (1 - 12) <br> Does not contain QA mask                                              | Unit: TOA reflectance <br> (scaled by 10000)            | GeoTIFF <br> 512 x 512 <br> 13 bands <br> UInt16 (12bits used) | 0: B1 (Coastal, 60m) <br> 1: B2 (Blue, 10m) <br> 2: B3 (Green, 10m) <br> 3: B4 (Red, 10m) <br> 4: B5 (RedEdge-1, 20m) <br> 5: B6 (RedEdge-2, 20m) <br> 6: B7 (RedEdge-3, 20m) <br> 7: B8 (NIR, 10m) <br> 8: B8A (Narrow NIR, 20m) <br> 9: B9 (Water Vapor, 60m) <br> 10: B10 (Cirrus, 60m) <br> 11: B11 (SWIR-1, 20m) <br> 12: B12 (SWIR-2, 20m) |
