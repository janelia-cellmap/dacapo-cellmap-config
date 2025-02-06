import torch.onnx
import torch
import os

omnx_version = 17


def export_torch_model(model, input_shape, folder_result):
    model.eval()
    pt_file = os.path.join(folder_result, "model.pt")
    onnx_file = os.path.join(folder_result, "model.onnx")

    dummy_input = torch.rand(input_shape)
    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.save(pt_file)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=omnx_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
