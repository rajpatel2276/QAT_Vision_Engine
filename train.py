import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Import the architecture directly from your custom package
from qat_core import QATNet, QATConv2d

if __name__ == "__main__":
    print("--- Initializing QAT Production Training Pipeline ---")

    device = torch.device("cpu")
    print(f"Target Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Downloading/Loading CIFAR-10 Dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    model = QATNet(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1 
    
    print("\nCommencing Training (Simulating 8-bit Integer Math)...")
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1:5d}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("\n[SUCCESS] QAT Training Complete.")
    export_path = "qat_net_weights.pt"
    torch.save(model.state_dict(), export_path)
    print(f"Deployable Weights extracted and saved to: {export_path}")
    
    print("\n--- Starting QAT Engine Verification ---")
    dummy_image = torch.randn(1, 1, 4, 4)
    qat_layer = QATConv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
    
    with torch.no_grad():
        qat_layer.weight.copy_(torch.tensor([[[[0.15, -0.35], 
                                               [0.78, 0.02]]]]))
    
    output = qat_layer(dummy_image)
    loss = output.mean()
    loss.backward()

    weight_gradient = qat_layer.weight.grad
    if weight_gradient is not None and torch.any(weight_gradient != 0):
        print("[SUCCESS] Verification Complete! Gradients bypassed the rounding layer.")
    else:
        print("[FAILURE] Gradient Death Detected.")