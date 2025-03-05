# Used to export dacapo model

You can use the interactive script to export the model.:
```bash
dcc_dacapo
```

The exported folder will contain the following files:

```
<run_name>/
├── model.omnx
├── model.pt
├── model.ts
├── README.md
└── metadata.json
```

The `metadata.json` file contains the following model metadata structure:

```json
{
    "model_name": "model_name",
    "model_type": "UNet",
    "framework": "Dacapo",
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 2,
    "iteration": 1000,
    "input_voxel_size": [8, 8, 8],
    "output_voxel_size": [8, 8, 8],
    "channels_names": ["CT", "PET"],
    "input_shape": [96, 96, 96],
    "output_shape": [96, 96, 96],
    "author": "author",
    "description": "description",
    "version": "1.0.0"
}
```

| Attribute          | Type                | Description                                      | Example                       |
|--------------------|---------------------|--------------------------------------------------|-------------------------------|
| model_name         | Optional[str]       |                                                  |                               |
| model_type         | Optional[str]       | UNet or DenseNet121                              |                               |
| framework          | Optional[str]       | Dacapo or PyTorch                                 |                               |
| spatial_dims       | Optional[int]       | 2 or 3                                           |                               |
| in_channels        | Optional[int]       |                                                  |                               |
| out_channels       | Optional[int]       |                                                  |                               |
| iteration          | Optional[int]       |                                                  |                               |
| input_voxel_size   | Optional[List[int]] | Comma-separated values                           | 8,8,8                         |
| output_voxel_size  | Optional[List[int]] | Comma-separated values                           | 8,8,8                         |
| channels_names     | Optional[List[str]] | Comma-separated values                           | 'CT, PET'                     |
| input_shape        | Optional[List[int]] | Comma-separated values                           | 96,96,96                  |
| output_shape       | Optional[List[int]] | Comma-separated values                           | 96,96,96                  |
| author             | Optional[str]       |                                                  |                               |
| description        | Optional[str]       |                                                  |                               |
| version            | Optional[str]       |                                                  | 1.0.0                         |


# Saved models
## jrc_mus_liver 
- 8nm mito :
    - v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1
    - iteration: 345000

- 8nm peroxisome :
    - v22_peroxisome_funetuning_best_v20_1e4_finetuned_distances_8nm_peroxisome_jrc_mus-livers_peroxisome_8nm_attention-upsample-unet_default_one_label_finetuning_0
    - iteration: 45000


## Note: i had to checkout dacapo to version Feb 15 10:49:53 2024

commit 5371dedd3a008e438b601a227c4166273aab34bf
```bash
commit 5371dedd3a008e438b601a227c4166273aab34bf (HEAD)
Author: Marwan Zouinkhi <zouinkhi.marwan@gmail.com>
Date:   Thu Feb 15 10:49:53 2024 -0500

    docstrings losses
```

dacapo.yaml
```yaml
mongo_db_name: dacapo_cellmap_v3_zouinkhim
runs_base_dir: "/nrs/cellmap/zouinkhim/crop_num_experiment_v2"
mongo_db_host: mongodb://cellmapAdmin:LUwWXkSY8N3AqCcw@cellmap-mongo:27017
type: "mongo"
```
