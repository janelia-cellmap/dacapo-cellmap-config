import json
import click
from typing import List, Optional
from pydantic import BaseModel, Field
import os
import dcc.model_export.config as c


def get_export_folder():
    if c.DCC_EXPORT_FOLDER is not None:
        return c.DCC_EXPORT_FOLDER
    print("DCC_EXPORT_FOLDER is not set in the config, checking environment variables...")
    folder = os.getenv("DCC_EXPORT_FOLDER")
    if not folder:
        folder = input(
            "Didn't found DCC_EXPORT_FOLDER, Please enter the export folder path: "
        )
        os.environ["DCC_EXPORT_FOLDER"] = folder
    return folder


class ModelMetadata(BaseModel):
    model_name: Optional[str] = Field(None, description="Name of the model")
    model_type: Optional[str] = Field(None, description="Type of the model, e.g., UNet or DenseNet121")
    framework: Optional[str] = Field(None, description="Framework used, e.g., MONAI or PyTorch")
    spatial_dims: Optional[int] = Field(None, description="Number of spatial dimensions, e.g., 2 or 3")
    in_channels: Optional[int] = Field(None, description="Number of input channels")
    out_channels: Optional[int] = Field(None, description="Number of output channels")
    iteration: Optional[int] = Field(None, description="Iteration number")
    input_voxel_size: Optional[List[int]] = Field(None, description="Input voxel size as comma-separated values, e.g., 8,8,8")
    output_voxel_size: Optional[List[int]] = Field(None, description="Output voxel size as comma-separated values, e.g., 8,8,8")
    channels_names: Optional[List[str]] = Field(None, description="Names of the channels as comma-separated values, e.g., 'CT, PET'")
    input_shape: Optional[List[int]] = Field(None, description="Input shape as comma-separated values, e.g., 1,1,96,96,96")
    output_shape: Optional[List[int]] = Field(None, description="Output shape as comma-separated values, e.g., 1,2,96,96,96")
    inference_input_shape: Optional[List[int]] = Field(None, description="Inference input shape as comma-separated values, e.g., 1,1,96,96,96")
    inference_output_shape: Optional[List[int]] = Field(None, description="Inference output shape as comma-separated values, e.g., 1,2,96,96,96")
    author: Optional[str] = Field(None, description="Author of the model")
    description: Optional[str] = Field(None, description="Description of the model")
    version: Optional[str] = Field("1.0.0", description="Version of the model")


def generate_readme(metadata: ModelMetadata):
    readme_content = f"""
    # {metadata.model_name} Model
    iteration: {metadata.iteration}

    ## Description
    {metadata.description}

    ## Model Details
    - **Model Type:** {metadata.model_type}
    - **Framework:** {metadata.framework}
    - **Spatial Dimensions:** {metadata.spatial_dims}
    - **Input Channels:** {metadata.in_channels}
    - **Output Channels:** {metadata.out_channels}
    - **Channel Names:** {', '.join(metadata.channels_names)}
    - **Input Shape:** {', '.join(map(str, metadata.input_shape))}
    - **Output Shape:** {', '.join(map(str, metadata.output_shape))}

    ## Author
    {metadata.author}

    ## Version
    {metadata.version}
    """
    return readme_content


def export_metadata(metadata: ModelMetadata, overwrite: bool = False):
    
    export_folder = get_export_folder()
    result_folder = os.path.join(export_folder, metadata.model_name)
    if os.path.exists(result_folder) and not overwrite:
        result = click.confirm(
            f"Folder {result_folder} already exists. Do you want to overwrite it?",
        )
        if not result:
            return
    metadata = prompt_for_missing_fields(metadata)
    os.makedirs(result_folder, exist_ok=True)
    output_file = os.path.join(result_folder, "metadata.json")
    with open(output_file, "w") as f:
        json.dump(metadata.dict(), f, indent=4)
    click.echo(f"Metadata saved to {output_file}")
    readme = generate_readme(metadata)
    readme_file = os.path.join(result_folder, "README.md")
    with open(readme_file, "w") as f:
        f.write(readme)
    click.echo(f"README saved to {readme_file}")


def prompt_for_missing_fields(metadata: ModelMetadata):
    for field_name, field in metadata.__fields__.items():
        value = getattr(metadata, field_name)
        if value is None:
            try:
                prompt_text = (
                field.description or f"Enter {field_name.replace('_', ' ')}"
                )

                if field_name in ["channels_names", "input_shape", "output_shape"]:
                    user_input = click.prompt(prompt_text, type=str)
                    if field_name == "channels_names":
                        value = [item.strip() for item in user_input.split(",")]
                    else:
                        value = [int(item) for item in user_input.split(",")]
                elif field.type_ == int:
                    value = click.prompt(prompt_text, type=int)
                else:
                    value = click.prompt(prompt_text, type=str)
                setattr(metadata, field_name, value)
            except Exception as e:
                raise Exception(f"Field {field_name} is missing a description. {e}")

    return metadata


@click.command()
@click.option("--model_name", prompt="Enter model name", type=str)
@click.option("--model_type", prompt="Enter model type (UNet/DenseNet121)", type=str)
@click.option(
    "--framework", prompt="Enter framework (MONAI/PyTorch)", type=str, default="Pytorch"
)
@click.option(
    "--spatial_dims", prompt="Enter spatial dimensions (2 or 3)", type=int, default=3
)
@click.option(
    "--in_channels", prompt="Enter number of input channels", type=int, default=1
)
@click.option(
    "--out_channels", prompt="Enter number of output channels", type=int, default=2
)
@click.option(
    "--channels_names",
    prompt="Enter channel names as comma-separated values (e.g., 'CT, PET')",
    type=str,
)
@click.option(
    "--input_shape",
    prompt="Enter input shape as comma-separated values (e.g., 1,1,96,96,96)",
    type=str,
)
@click.option(
    "--output_shape",
    prompt="Enter output shape as comma-separated values (e.g., 1,2,96,96,96)",
    type=str,
)
@click.option(
    "--author",
    prompt="Enter model author",
    type=str,
    default="",
)
@click.option(
    "--description",
    prompt="Enter model description",
    type=str,
    default="",
)
@click.option(
    "--version",
    prompt="Enter model version",
    type=str,
    default="1.0.0",
)
def generate_metadata(
    model_name,
    model_type,
    framework,
    spatial_dims,
    in_channels,
    out_channels,
    channels_names,
    input_shape,
    output_shape,
    author,
    description,
    version,
):
    input_shape = tuple(map(int, input_shape.split(",")))  # Convert input to tuple
    output_shape = tuple(map(int, output_shape.split(",")))  # Convert output to tuple
    channels_names = channels_names.split(",")

    # Generate metadata
    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "framework": framework,
        "spatial_dims": spatial_dims,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "channels_names": channels_names,
        "author": author,
        "description": description,
        "version": version,
    }

    # Save metadata to JSON file
    output_file = f"{model_name}_metadata.json"
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=4)

    click.echo(f"Metadata saved to {output_file}")


if __name__ == "__main__":
    generate_metadata()
