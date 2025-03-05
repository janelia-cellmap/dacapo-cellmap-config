from pathlib import Path
import yaml
import os
import numpy as np
from dacapo.experiments.datasplits import (
    TrainValidateDataSplitConfig,
    DummyDataSplitConfig,
)
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    BinarizeArrayConfig,
    IntensitiesArrayConfig,
    OnesArrayConfig,
    CropArrayConfig,
)
import logging
from funlib.geometry import Coordinate

logger = logging.getLogger(__name__)


def split_path(path):
    elm = path.split(".zarr")
    container = elm[0] + ".zarr"
    dataset = elm[1].lstrip("/").rstrip("/")
    return Path(container), dataset

def get_validation_crops(crops):
    crops_copy = crops.copy()
    smallest_val = None
    smallest = None
    for c in crops:
        try:
            total = np.prod(c.gt_config.array().shape)
            # print(c.name,total)
            if smallest is None or total < smallest_val:
                smallest = c
                smallest_val = total
        except Exception as e:
            raise Exception(f"Error with {c}\n{e}")
    crops_copy.remove(smallest)
    return crops_copy, [smallest]

def check_resolution(datasplit, resolution=(8,8,8)):
    groups = [datasplit.train_configs, datasplit.validate_configs]
    for group in groups:
        for d in group:
            raw_config = d.raw_config
            if raw_config.array().voxel_size != resolution:
                raise ValueError(f"Resolution of {raw_config} does not match expected {resolution}")
            # if raw_config.array().offset % raw_config.array().voxel_size != Coordinate((0,) * raw_config.array().voxel_size.dims):
            #     print(
            #         f"{raw_config.name} {raw_config.array().offset} - {raw_config.array().voxel_size} has an offset that is not a multiple of the voxel size, \n ")
            gt_config = d.gt_config
            if gt_config.array().voxel_size != resolution:
                raise ValueError(f"Resolution of {gt_config} does not match expected {resolution}")
            # if gt_config.array().offset % gt_config.array().voxel_size != Coordinate((0,) * gt_config.array().voxel_size.dims):
            #     print(
            #         f"{gt_config.name} {gt_config.array().offset} - {gt_config.array().voxel_size} has an offset that is not a multiple of the voxel size, \n ")
    print("All datasets have the expected resolution.")

def generate_datasplit(yaml_file, groupping,name, resolution):
    data =  yaml.load(open(yaml_file), Loader=yaml.FullLoader)["datasets"]

    all_crops = []
    for dataset_name,dataset_info in data.items():

        raw_path = dataset_info["raw_path"]
        raw_min = dataset_info.get("raw_min",0)
        raw_max = dataset_info.get("raw_max",255)
        raw_container, raw_dataset = split_path(raw_path)
        # print(f"raw_container: {raw_container}, raw_dataset: {raw_dataset}")
        crop_raw = ZarrArrayConfig(
            name=f"{dataset_name}_raw",
            file_name=raw_container,
            dataset=raw_dataset,
            ome_metadata=True,
            snap_to_grid=resolution,
        )
        intensity_raw = IntensitiesArrayConfig(
            name=f"{dataset_name}_intensity_raw",
            source_array_config=crop_raw,
            # clip=True,
            min=raw_min,
            max=raw_max,
        )
        current_crops = [] 

        for crop in dataset_info["crops"]:
            crop_name = crop["name"]
            crop_path = crop["gt_path"]
            if not os.path.exists(crop_path):
                raise Exception(f"{crop_path} does not exist")
            crop_container, crop_dataset = split_path(crop_path)
            # print(f"crop_container: {crop_container}, crop_dataset: {crop_dataset}")
            crop_gt = ZarrArrayConfig(
                name=f"{crop_name}_gt",
                file_name=crop_container,
                dataset=crop_dataset,
                ome_metadata=True,
                snap_to_grid=resolution,
            )
            crop_distances = BinarizeArrayConfig(
                f"{crop_name}_distances",
                source_array_config=crop_gt,
                groupings=groupping,
            )

            # cropped_raw = CropArrayConfig(
            #     name=f"{crop_name}_raw",
            #     source_array_config=intensity_raw,
            #     roi=crop_gt.array().roi,
            # )

            # mask_array = OnesArrayConfig(
            #     name=f"{crop_name}_mask",
            #     source_array_config=crop_gt,
            # )
            crop_data = RawGTDatasetConfig(
                name=f"{dataset_name}_{crop_name}_dataset",
                raw_config=intensity_raw,
                gt_config=crop_distances,
                # mask_config=mask_array,
                
            )
            current_crops.append(crop_data)
        all_crops.append(current_crops)
    train = []
    val = []
    for crops in all_crops:
        train_crops, val_crops = get_validation_crops(crops)
        train.extend(train_crops)
        val.extend(val_crops)

    datasplit = TrainValidateDataSplitConfig(
            name=name,
            train_configs=train,
            validate_configs=val,
        )
    check_resolution(datasplit, resolution)
    return datasplit








#%%


# %%
