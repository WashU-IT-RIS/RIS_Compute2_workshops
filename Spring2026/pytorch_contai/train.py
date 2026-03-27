import torch
import torch.nn as nn
import torch.optim as optim

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Input: 1x28x28 (MNIST style)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), # Padding=2 keeps it 28x28
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10) # 10 output classes
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        logits = self.classifier(x)
        return logits


def main():
    info = get_device_memory_info(0)
    print("PyTorch version:", torch.__version__)
    x = torch.rand(5, 3)
    print(x)
    if torch.cuda.is_available():
      print("CUDA is available!")
      
      # Get the number of available GPUs
      num_gpus = torch.cuda.device_count()
      print(f"Number of GPUs available: {num_gpus}")
      
      # Get the name of each GPU
      for i in range(num_gpus):
          print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
      
      # Get the current device index
      current_device = torch.cuda.current_device()
      print(f"Current CUDA device index: {current_device}")
      print(f"Current CUDA device name: {torch.cuda.get_device_name(current_device)}")
    else:
      print("CUDA is not available. Running on CPU.")
    if info:
        print(f"Device: {info['device']}")
        print(f"Total Memory: {info['total_MB']:.2f} MB")
        print(f"Reserved Memory: {info['reserved_MB']:.2f} MB")
        print(f"Allocated Memory: {info['allocated_MB']:.2f} MB")
        print(f"Free Memory (within reserved): {info['free_MB']:.2f} MB")
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    
    # 2. Initialize Model and move to GPU
    model = LeNet5().to(device)
    
    # 3. Create Random Data (Batch Size, Channels, Height, Width)
    # 64 images of 1 channel, 28x28 pixels
    random_input = torch.randn(64, 1, 28, 28).to(device)
    random_labels = torch.randint(0, 10, (64,)).to(device)
    
    # 4. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. A Single Forward/Backward Pass
    model.train()
    optimizer.zero_grad()
    
    outputs = model(random_input)
    loss = criterion(outputs, random_labels)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print("Successfully ran one training step on GPU!")
if __name__ == "__main__":
    main()
