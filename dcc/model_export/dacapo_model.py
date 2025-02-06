from dacapo.experiments.run import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from .generate_metadata import ModelMetadata, export_metadata, get_export_folder
from .export_model import export_torch_model
import os
from funlib.geometry import Coordinate


def export_dacapo_model():
    run_name = input("Enter the run name: ")
    iteration = int(input("Enter the iteration number: "))
    get_dacapo_infos(run_name, iteration)


def get_dacapo_infos(run_name: str, iteration: int):

    config_store = create_config_store()
    run = Run(config_store.retrieve_run_config(run_name))
    if iteration > 0:
        weights_store = create_weights_store()
        weights = weights_store.retrieve_weights(run_name, iteration)
        run.model.load_state_dict(weights.model)

    input_scale = Coordinate(8,8,8)
    output_scale = run.model.scale(input_scale)


    metadata = ModelMetadata(
        model_name=run_name,
        model_type=run.model.architecture.__class__.__name__,
        framework="dacapo/torch",
        in_channels=run.model.num_in_channels,
        out_channels=run.model.num_out_channels,
        channels_names=run.task.channels,
        input_shape=run.model.input_shape,
        output_shape=run.model.output_shape,
        iteration=iteration,
        input_voxel_size=input_scale,
        output_voxel_size=output_scale,
    )
    input_shape = (1, run.model.num_in_channels, *run.model.input_shape)

    export_metadata(metadata)
    export_torch_model(run.model, input_shape, os.path.join(get_export_folder(), run_name))
