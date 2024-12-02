import torch

if __name__ == '__main__':
    # Define the shape of the tensor
    tensor_shape = (2, 3, 3, 64, 64)

    # Generate a random tensor with the specified shape
    # Values are scaled to be between 0 and 100, as implied by the division by 100 in the script
    input_tensor = torch.rand(tensor_shape) * 100

    # Save the tensor to a .pt file
    tensor_filepath = "/workspace/mmve/test_input_tensor2_3_3_64_64.pt"
    torch.save(input_tensor, tensor_filepath)

    print(f"Tensor saved to: {tensor_filepath}")
