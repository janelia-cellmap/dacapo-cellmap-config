import torch.onnx
import torch
import os

omnx_version = 17


def export_torch_model(model, input_shape, folder_result):
    model.eval()
    print(f"Exporting model to {folder_result}")
    pt_file = os.path.join(folder_result, "model.pt")
    onnx_file = os.path.join(folder_result, "model.onnx")
    ts_file = os.path.join(folder_result, "model.ts")

    # Export to TorchScript
    torch.save(model, pt_file)
    print(f"Model saved to {pt_file}")

    dummy_input = torch.rand(input_shape)
    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.save(ts_file)
    print(f"Model saved to {ts_file}")

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
    print(f"Model saved to {onnx_file}")
