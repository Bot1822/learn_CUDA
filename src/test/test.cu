
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

// 测试卷积操作
void test_conv2d() {
    // 读取输入特征图
    std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                                15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                                20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
    // 转换为3通道的输入特征图
    std::vector<float> input_3_channel(input.size() * 3);
    for (int i = 0; i < input.size(); ++i) {
        input_3_channel[i] = input[i];
        input_3_channel[i + input.size()] = input[i];
        input_3_channel[i + input.size() * 2] = input[i];
    }
    // 读取卷积核
    std::vector<float> kernel = {0.0f, 1.0f, 2.0f,
                   3.0f, 4.0f, 5.0f,
                   6.0f, 7.0f, 8.0f};
    // 转换为3*6通道的卷积核
    std::vector<float> kernel_3_6_channel(kernel.size() * 3 * 6);
    for (int i = 0; i < kernel.size(); ++i) {
        for (int j = 0; j < 3 * 6; ++j) {
            kernel_3_6_channel[i + j * kernel.size()] = kernel[i];
        }
    }
    auto& weight = kernel_3_6_channel;
    // 读取偏置
    std::vector<float> bias = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    // 计算输出特征图的尺寸
    int output_width = (5 + 2 * 1 - 3) / 1 + 1;
    int output_height = (5 + 2 * 1 - 3) / 1 + 1;

    // 申请输出特征图的空间
    std::vector<float> output(output_width * output_height * 6);

    // 将输入特征图、卷积核和偏置拷贝到显存中
    float* input_gpu;
    float* output_gpu;
    float* weight_gpu;
    float* bias_gpu;
    cudaMalloc(&input_gpu, input_3_channel.size() * sizeof(float));
    cudaMalloc(&output_gpu, output.size() * sizeof(float));
    cudaMalloc(&weight_gpu, weight.size() * sizeof(float));
    cudaMalloc(&bias_gpu, bias.size() * sizeof(float));
    cudaMemcpy(input_gpu, input_3_channel.data(), input_3_channel.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_gpu, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 计算线程块的尺寸，block_dim大小为32*32，每个block中有1024个线程
    dim3 block_dim(32, 32);
    // 计算线程块的数量，grid_dim大小为2*2，一共有4个线程块
    dim3 grid_dim((output_width + block_dim.x - 1) / block_dim.x, (output_height + block_dim.y - 1) / block_dim.y);

    // 调用卷积操作
    conv2d<<<grid_dim, block_dim>>>(input_gpu, output_gpu, weight_gpu, bias_gpu, 5, 5, 3, 6, 3, 1, 1);

    // 将输出特征图拷贝回主机端
    cudaMemcpy(output.data(), output_gpu, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印输出特征图
    for (int c = 0; c < 6; ++c) {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                std::cout << output[x + y * output_width + c * output_width * output_height] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

}

// 测试池化操作
void test_max_pooling2d() {
    // 读取输入特征图
    std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                                15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                                20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
    // 转换为3通道的输入特征图
    std::vector<float> input_3_channel(input.size() * 3);
    for (int i = 0; i < input.size(); ++i) {
        input_3_channel[i] = input[i];
        input_3_channel[i + input.size()] = input[i];
        input_3_channel[i + input.size() * 2] = input[i];
    }

    // 计算输出特征图的尺寸
    int output_width = (5 - 2) / 2 + 1;
    int output_height = (5 - 2) / 2 + 1;

    // 申请输出特征图的空间
    std::vector<float> output(output_width * output_height * 3);

    // 将输入特征图拷贝到显存中
    float* input_gpu;
    float* output_gpu;
    cudaMalloc(&input_gpu, input_3_channel.size() * sizeof(float));
    cudaMalloc(&output_gpu, output.size() * sizeof(float));
    cudaMemcpy(input_gpu, input_3_channel.data(), input_3_channel.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 计算线程块的尺寸，block_dim大小为32*32，每个block中有1024个线程
    dim3 block_dim(32, 32);
    // 计算线程块的数量，grid_dim大小为2*2，一共有4个线程块
    dim3 grid_dim((output_width + block_dim.x - 1) / block_dim.x, (output_height + block_dim.y - 1) / block_dim.y);

    // 调用池化操作
    max_pooling2d<<<grid_dim, block_dim>>>(input_gpu, output_gpu, 2, 5, 5, 3, 3);

    // 将输出特征图拷贝回主机端
    cudaMemcpy(output.data(), output_gpu, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印输出特征图
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                std::cout << output[x + y * output_width + c * output_width * output_height] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// 测试ReLU激活函数
void test_relu() {
    // 读取输入特征图
    std::vector<float> input = {-1.0f, 1.0f, -2.0f, 2.0f, -3.0f,
                                3.0f, -4.0f, 4.0f, -5.0f, 5.0f};
    // 申请输出特征图的空间
    std::vector<float> output(input.size());

    // 将输入特征图拷贝到显存中
    float* input_gpu;
    float* output_gpu;
    cudaMalloc(&input_gpu, input.size() * sizeof(float));
    cudaMalloc(&output_gpu, output.size() * sizeof(float));
    cudaMemcpy(input_gpu, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 计算线程块的尺寸，block_dim大小为32*32，每个block中有1024个线程
    dim3 block_dim(32, 32);
    // 计算线程块的数量，grid_dim大小为2*2，一共有4个线程块
    dim3 grid_dim((input.size() + block_dim.x - 1) / block_dim.x);

    // 调用ReLU激活函数
    relu<<<grid_dim, block_dim>>>(input_gpu, output_gpu, input.size());

    // 将输出特征图拷贝回主机端
    cudaMemcpy(output.data(), output_gpu, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印输出特征图
    for (int i = 0; i < output.size(); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
}

// 测试全连接层
void test_fully_connected() {
    // 读取输入特征图
    std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    // 读取全连接层参数
    std::vector<float> weight = {0.0f, 1.0f, 2.0f,
                                 3.0f, 4.0f, 5.0f,
                                 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f,
                                 12.0f, 13.0f, 14.0f};
    std::vector<float> bias = {0.0f, 0.1f, 0.2f};

    // 申请输出特征图的空间
    std::vector<float> output(bias.size());

    // 将输入特征图、全连接层参数和偏置拷贝到显存中
    float* input_gpu;
    float* output_gpu;
    float* weight_gpu;
    float* bias_gpu;
    cudaMalloc(&input_gpu, input.size() * sizeof(float));
    cudaMalloc(&output_gpu, output.size() * sizeof(float));
    cudaMalloc(&weight_gpu, weight.size() * sizeof(float));
    cudaMalloc(&bias_gpu, bias.size() * sizeof(float));
    cudaMemcpy(input_gpu, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_gpu, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 计算线程块的尺寸，block_dim大小为32*32，每个block中有1024个线程
    dim3 block_dim(32, 32);
    // 计算线程块的数量，grid_dim大小为2*2，一共有4个线程块
    dim3 grid_dim((bias.size() + block_dim.x - 1) / block_dim.x);

    // 调用全连接层
    fully_connected<<<grid_dim, block_dim>>>(input_gpu, output_gpu, weight_gpu, bias_gpu, input.size(), bias.size());

    // 将输出特征图拷贝回主机端
    cudaMemcpy(output.data(), output_gpu, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印输出特征图
    for (int i = 0; i < output.size(); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
}
