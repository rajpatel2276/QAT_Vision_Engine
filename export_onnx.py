import torch
from qat_core import QATNet

def export_to_onnx():
    print("--- Initializing ONNX Compiler ---")
    
    # 1. Initialize the architecture
    model = QATNet(num_classes=10)
    
    # 2. Load the hardened INT8 weights from your training session
    weight_path = "qat_net_weights.pt"
    try:
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        print(f"[SUCCESS] Loaded trained weights from {weight_path}")
    except FileNotFoundError:
        print(f"[ERROR] Could not find {weight_path}. Did you run train.py first?")
        return

    # 3. Set the model to evaluation mode (shuts off training behaviors like Dropout/BatchNorm updates)
    model.eval()

    # 4. Create a dummy tensor that perfectly matches a CIFAR-10 image (Batch=1, Channels=3, H=32, W=32)
    # The compiler needs this to physically trace the mathematical operations.
    dummy_input = torch.randn(1, 3, 32, 32)

    # 5. Compile and Export
    onnx_file_path = "qat_net_deployable.onnx"
    print(f"Tracing execution graph and compiling to {onnx_file_path}...")
    
    torch.onnx.export(
        model,                      # The instantiated model
        dummy_input,                # The tracing input
        onnx_file_path,             # Where to save the file
        export_params=True,         # Store the trained weights inside the file
        opset_version=11,           # Standard ONNX version for edge compatibility
        do_constant_folding=True,   # Optimizes the graph by pre-calculating constant math
        input_names=['input_image'],   # Name the input node for the C++ developer
        output_names=['class_logits']  # Name the output node for the C++ developer
    )
    
    print(f"\n[SUCCESS] Bare-metal compilation complete. The model is ready for edge deployment.")

if __name__ == "__main__":
    export_to_onnx()