#%%
import torch
from fly_organelles.model import StandardUnet
import numpy as np
from dcc.model_export.generate_metadata import ModelMetadata, export_metadata, get_export_folder
from dcc.model_export.export_model import export_torch_model
import dcc.model_export.config as c
c.DCC_EXPORT_FOLDER = "/groups/cellmap/cellmap/zouinkhim/models/saalfeldlab/fly"
import os
os.chdir(c.DCC_EXPORT_FOLDER)

#%%
models = {"run07":{700000:"/nrs/saalfeld/heinrichl/fly_organelles/run07/model_checkpoint_700000",
                   432000:"/nrs/saalfeld/heinrichl/fly_organelles/run07/model_checkpoint_432000"},
          "run08":{438000:"/nrs/saalfeld/heinrichl/fly_organelles/run08/model_checkpoint_438000"}}


#%%
# pip install fly-organelles

input_voxel_size = np.array((8, 8, 8))
output_voxel_size = np.array((8, 8, 8))
input_shape = np.array((178, 178, 178)) 
output_shape = np.array((56, 56, 56))
inference_input_shape = input_shape
infernece_output_shape = output_shape
author = "Larissa Heinrich"
description = "Fly organelles segmentation model"

classes_names = ["all_mem", "organelle", "mito", "er", "nucleus", "pm", "vs", "ld"]
block_shape = np.array((56, 56, 56,8))
output_channels = 8
#%%

#%%
def load_eval_model(num_labels, checkpoint_path):
    model_backbone = StandardUnet(num_labels)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)    
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model_backbone.load_state_dict(checkpoint["model_state_dict"])
    model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
    model.to(device)
    model.eval()
    return model

# %%
# print("model loaded",model)
# %%
for run_name, iterations in models.items():
    for iteration, checkpoint_path in iterations.items():
        name = f"fly_organelles_{run_name}_{iteration}"
        print(f"Exporting model {name} from {checkpoint_path}")
        model = load_eval_model(output_channels, checkpoint_path)
        # print("model loaded",model)
        description = f"Fly organelles segmentation model {run_name} iteration {iteration}"
        


        metadata = ModelMetadata(
            model_name=name,
            iteration=iteration,
            model_type=model.__class__.__name__,
            framework="torch",
            in_channels=1,
            spatial_dims=3,
            out_channels=output_channels,
            channels_names=classes_names,
            inference_input_shape=inference_input_shape,
            inference_output_shape=infernece_output_shape,
            input_shape=input_shape,
            output_shape=output_shape,
            input_voxel_size=input_voxel_size,
            output_voxel_size=output_voxel_size,
            author=author,
            description=description,
        )
        input_shape = (1, 1, *inference_input_shape)

        export_metadata(metadata)
        export_torch_model(model, input_shape, os.path.join(get_export_folder(), name))




# %%

# %%
# check
import onnxruntime as ort

import numpy as np

# Path to your ONNX model
k = list(marwan_models.keys())[0]
onnx_file = os.path.join(c.DCC_EXPORT_FOLDER,k,"model.onnx")
print(f"Loading model from {onnx_file}")
#%%

# Create an inference session
session = ort.InferenceSession(onnx_file)

# Get the name of the first input of the model
input_name = session.get_inputs()[0].name
print("Input name  :", input_name)
print("Input shape :", session.get_inputs()[0].shape)

# Get the name of the first output of the model
output_name = session.get_outputs()[0].name
print("Output name :", output_name)
print("Output shape:", session.get_outputs()[0].shape)

# Prepare input data as a NumPy array (make sure shapes & dtypes match your model)
dummy_input = np.random.randn(1, 1,288,288,288).astype(np.float32)

# Run inference
result = session.run([output_name], {input_name: dummy_input})  # returns a list of outputs
print("Inference result shape:", np.array(result).shape)

# %%
import torch

model_path = os.path.join(c.DCC_EXPORT_FOLDER,k,"model.ts")
# model_path = "model.ts"
model = torch.jit.load(model_path)

# 2. Switch the model to evaluation mode
model.eval()

# 3. Prepare some test input data
#    (Make sure its shape and dtype match your model's expectations)
dummy_input = np.random.randn(1, 1,288,288,288).astype(np.float32)
tensor_input = torch.from_numpy(dummy_input)
# 4. Run inference
output = model(tensor_input)

# 5. Inspect the output
print("Output shape:", output.shape)
print("Output data:", output)

# %%
