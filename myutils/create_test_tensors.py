import torch

# Generate uniform random noise tensor of size [B=2, T=7, C=3, H=64, W=64]
noise_tensor = torch.rand([2, 7, 3, 64, 64])

# Save the generated tensor to a file
tensor_path = 'test_input_tensor2_7_3_64_64.pt'
torch.save(noise_tensor, tensor_path)
