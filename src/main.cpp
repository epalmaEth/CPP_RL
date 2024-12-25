#include <torch/torch.h>
#include <iostream>

int main() {
    // Create a random 2x2 tensor
    torch::Tensor tensor = torch::rand({2, 2});

    // Print the tensor on CPU
    std::cout << "Tensor on CPU:\n" << tensor << std::endl;

    // Move the tensor to CUDA if available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Running on GPU." << std::endl;
        torch::Tensor tensor_cuda = tensor.to(torch::kCUDA);
        std::cout << "Tensor on CUDA:\n" << tensor_cuda << std::endl;
    } else {
        std::cout << "CUDA is not available. Running on CPU." << std::endl;
    }

    return 0;
}
