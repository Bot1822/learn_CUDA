// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70
// nvcc test.cu -o train -Xcompiler "-O3 -std=c++14" -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -rdc=true
// nvcc inferCNN.cu -o inferCNN -Xcompiler "-O3 -std=c++14" -gencode arch=compute_89,code=sm_89 -rdc=true
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

#define CUDA_CALL(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// // 日志文件
// std::ofstream log_file;

class DataLoader {
private:
    const int batchSize;
    std::vector<std::vector<float>> data;
    std::vector<int> labels;
    int currentIndex;

public:
    DataLoader(int batchSize, std::vector<std::vector<float>> data, std::vector<int> labels) 
        : batchSize(batchSize), data(data), labels(labels), currentIndex(0) {
    }

    // 获取下一批次的数据
    std::vector<std::vector<float>> getNextBatch() {
        if (currentIndex >= data.size()) {
            std::cerr << "All batches have been loaded." << std::endl;
            return {};
        }

        std::vector<std::vector<float>> batch;
        for (int i = 0; i < batchSize && currentIndex < data.size(); ++i, ++currentIndex) {
            batch.push_back(data[currentIndex]);
        }
        return batch;
    }

    // 重置批次索引
    void reset() {
        currentIndex = 0;
    }
};

// ReLU激活函数
__global__ void relu(float* input, float* output, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        output[index] = fmaxf(input[index], 0.0f);
    }
}

// batch版本的ReLU激活函数
__global__ void relu_batch(float* input, float* output, int batch_size, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockIdx.y;
    if (index < size && b < batch_size) {
        output[b * size + index] = fmaxf(input[b * size + index], 0.0f);
    }
}

// 利用CUDA实现卷积操作
__global__ void conv2d(float* input, float* output, float* weight, float* bias, int width, int height,
                       int input_channel, int output_channel, int kernel_size, int stride, int padding) 
{
    // 计算输出特征图的坐标
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    // 计算输出特征图的尺寸
    int output_width = (width + 2 * padding - kernel_size) / stride + 1;
    int output_height = (height + 2 * padding - kernel_size) / stride + 1;

    // 确保线程索引不超出输出特征图的尺寸
    if (x < output_width && y < output_height) {
        // 遍历输出特征图的每个通道
        for (int oc = 0; oc < output_channel; oc++) {
            float value = bias[oc];  //偏置

            // 对输入特征图的每个通道进行操作
            for (int ic = 0; ic < input_channel; ic++) {
                // 遍历卷积核的每个元素
                for (int kx = 0; kx < kernel_size; kx++) {
                    for (int ky = 0; ky < kernel_size; ky++) {
                        // 计算输入特征图上对应的x和y坐标
                        int ix = x * stride - padding + kx;
                        int iy = y * stride - padding + ky;

                        // 如果ix和iy在输入特征图的边界内，则进行计算
                        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                            value += input[iy * width + ix + ic * width * height]   //中括号里面的是偏移量
                                   * weight[ky * kernel_size + kx + ic * kernel_size * kernel_size + oc * kernel_size * kernel_size * input_channel];
                        }
                    }
                }
            }

            // 将计算得到的value赋值给输出特征图的对应位置
            output[y * output_width + x + oc * output_width * output_height] = value;
        }
    }   
}

// batch版本的卷积操作
__global__ void conv2d_batch(float* input, float* output, float* weight, float* bias, int batch_size, int width, int height,
                       int input_channel, int output_channel, int kernel_size, int stride, int padding) 
{
    int output_width = (width - kernel_size + 2 * padding) / stride + 1;
    int output_height = (height - kernel_size + 2 * padding) / stride + 1;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int b = blockIdx.z;

    if (x < output_width && y < output_height && b < batch_size) {
        for (int oc = 0; oc < output_channel; oc++) {
            float value = bias[oc];

            for (int ic = 0; ic < input_channel; ic++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    for (int ky = 0; ky < kernel_size; ky++) {
                        int ix = x * stride - padding + kx;
                        int iy = y * stride - padding + ky;

                        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                            value += input[b * input_channel * width * height + iy * width + ix + ic * width * height]
                                   * weight[ky * kernel_size + kx + ic * kernel_size * kernel_size + oc * kernel_size * kernel_size * input_channel];
                        }
                    }
                }
            }

            output[b * output_channel * output_width * output_height + y * output_width + x + oc * output_width * output_height] = value;
        }
    }
}

// batch版本的卷积操作结合ReLU激活函数
__global__ void conv2d_batch_relu(float* input, float* output, float* weight, float* bias, int batch_size, int width, int height,
                       int input_channel, int output_channel, int kernel_size, int stride, int padding) 
{
    int output_width = (width - kernel_size + 2 * padding) / stride + 1;
    int output_height = (height - kernel_size + 2 * padding) / stride + 1;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int b = blockIdx.z;

    if (x < output_width && y < output_height && b < batch_size) {
        for (int oc = 0; oc < output_channel; oc++) {
            float value = bias[oc];

            for (int ic = 0; ic < input_channel; ic++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    for (int ky = 0; ky < kernel_size; ky++) {
                        int ix = x * stride - padding + kx;
                        int iy = y * stride - padding + ky;

                        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                            value += input[b * input_channel * width * height + iy * width + ix + ic * width * height]
                                   * weight[ky * kernel_size + kx + ic * kernel_size * kernel_size + oc * kernel_size * kernel_size * input_channel];
                        }
                    }
                }
            }

            output[b * output_channel * output_width * output_height + y * output_width + x + oc * output_width * output_height] = fmaxf(value, 0.0f);
        }
    }
}

// 池化操作
__global__ void max_pooling2d(float* input, float* output, int pool_size, int width, int height, int channel, int stride) {
    // 计算输出特征图的坐标
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // 计算输出特征图的尺寸
    int output_width = (width - pool_size) / stride + 1;
    int output_height = (height - pool_size) / stride + 1;

    // 确保线程索引不超出输出特征图的尺寸
    if (x < output_width && y < output_height) {
        // 遍历输出特征图的每个通道
        for (int c = 0; c < channel; c++) {
            float max_value = -FLT_MAX;
            
            // 遍历池化核的每个元素
            for (int px = 0; px < pool_size; px++) {
                for (int py = 0; py < pool_size; py++) {
                    // 计算输入特征图上对应的x和y坐标
                    int ix = x * stride + px;
                    int iy = y * stride + py;

                    // 如果ix和iy在输入特征图的边界内，则进行计算
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        max_value = fmaxf(max_value, input[iy * width + ix + c * width * height]);
                    }
                }
            }

            // 将计算得到的value赋值给输出特征图的对应位置
            output[y * output_width + x + c * output_width * output_height] = max_value;
        }
    }
}

// batch版本的池化操作
__global__ void max_pooling2d_batch(float* input, float* output, int batch_size, int pool_size, int width, int height, int channel, int stride) {
    int output_width = (width - pool_size) / stride + 1;
    int output_height = (height - pool_size) / stride + 1;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int b = blockIdx.z;

    if (x < output_width && y < output_height && b < batch_size) {
        for (int c = 0; c < channel; c++) {
            float max_value = -FLT_MAX;

            for (int px = 0; px < pool_size; px++) {
                for (int py = 0; py < pool_size; py++) {
                    int ix = x * stride + px;
                    int iy = y * stride + py;

                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        max_value = fmaxf(max_value, input[b * channel * width * height + iy * width + ix + c * width * height]);
                    }
                }
            }

            output[b * channel * output_width * output_height + y * output_width + x + c * output_width * output_height] = max_value;
        }
    }
}

// 全连接层
__global__ void fully_connected(float* input, float* output, float* weight, float* bias, int input_size, int output_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weight[index * input_size + i];
        }
        output[index] = sum + bias[index];
    }
}

// batch版本的全连接层
__global__ void fully_connected_batch(float* input, float* output, float* weight, float* bias, int batch_size, int input_size, int output_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockIdx.y;
    if (index < output_size && b < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[b * input_size + i] * weight[index * input_size + i];
        }
        output[b * output_size + index] = sum + bias[index];
    }
}

// batch版本的全连接层结合ReLU激活函数
__global__ void fully_connected_batch_relu(float* input, float* output, float* weight, float* bias, int batch_size, int input_size, int output_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockIdx.y;
    if (index < output_size && b < batch_size) {
        float sum = 0.0f;
        // Pytorch主序列为行
        for (int i = 0; i < input_size; ++i) {
            sum += input[b * input_size + i] * weight[index * input_size + i];
        }
        output[b * output_size + index] = fmaxf(sum + bias[index], 0.0f);
    }
}

// 读取MNIST数据集
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_images = ((num_images & 0xff000000) >> 24) | ((num_images & 0x00ff0000) >> 8) |
                 ((num_images & 0x0000ff00) << 8) | ((num_images & 0x000000ff) << 24);
    num_rows = ((num_rows & 0xff000000) >> 24) | ((num_rows & 0x00ff0000) >> 8) |
                ((num_rows & 0x0000ff00) << 8) | ((num_rows & 0x000000ff) << 24);
    num_cols = ((num_cols & 0xff000000) >> 24) | ((num_cols & 0x00ff0000) >> 8) |
                ((num_cols & 0x0000ff00) << 8) | ((num_cols & 0x000000ff) << 24);

    int image_size = num_rows * num_cols;
    std::vector<std::vector<float>> images(num_images, std::vector<float>(image_size));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0f;
            images[i][j] = (images[i][j] - 0.5f) / 0.5f;
        }
    }

    return images;
}

// 读取MNIST label数据集
std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_items = ((num_items & 0xff000000) >> 24) | ((num_items & 0x00ff0000) >> 8) |
                ((num_items & 0x0000ff00) << 8) | ((num_items & 0x000000ff) << 24);

    std::vector<int> labels(num_items);
    for (int i = 0; i < num_items; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

// 读取模型参数
std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    std::vector<float> params;
    float param;
    while (file >> param) {
        params.push_back(param);
    }
    return params;
}

// 范例kernel函数，无实际作用
__global__ void add_arrays(int* a, int* b, int* c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}


/*
    假设输入图像用单精度浮点数存储，每个像素点占用一个float，假设一次读入x张图片，那么输入特征图的大小为x * 1 * 28 * 28
    网络结构：
    conv1: 1 * 28 * 28 -> 6 * 24 * 24
    max_pooling2d: 6 * 24 * 24 -> 6 * 12 * 12
    conv2: 6 * 12 * 12 -> 16 * 8 * 8
    max_pooling2d: 16 * 8 * 8 -> 16 * 4 * 4
    fc1: 16 * 4 * 4 -> 120
    fc2: 120 -> 84
    fc3: 84 -> 10
*/ 
int main(int argc, char* argv[]) {
    // log_file.open("log.txt", std::ios::out | std::ios::trunc);
    // log_file << "Start infering..." << std::endl;
	std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
	// cout << dir;
	
    // 读取测试集，对于想实现CUDA C/C++训练的同学，参考训练集文件名为train-images-idx3-ubyte和train-labels-idx1-ubyte
    auto images = read_mnist_images(dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte");

    // // 输出一张images到log
    // log_file << "images.size():" << images.size() << std::endl;
    // log_file << "images[0].size():" << images[0].size() << std::endl;
    // log_file << "images[0]:" << std::endl;
    // for (int i = 0; i < 28; i++) {
    //     for (int j = 0; j < 28; j++) {
    //         log_file << std::setw(4) << images[0][i * 28 + j] << " ";
    //     }
    //     log_file << std::endl;
    // }
    // log_file << std::endl;

    // 读取测试集标签
    auto labels = read_mnist_labels(dir + "/../../data/FashionMNIST/raw/t10k-labels-idx1-ubyte");

    // // 输出一张label到log
    // log_file << "labels.size():" << labels.size() << std::endl;
    // log_file << "labels[0]:" << labels[0] << std::endl << std::endl;

    // 读取模型参数
    auto conv1_weight = read_param(dir + "/conv1.weight.txt");
    auto conv1_bias = read_param(dir + "/conv1.bias.txt");
    auto conv2_weight = read_param(dir + "/conv2.weight.txt");
    auto conv2_bias = read_param(dir + "/conv2.bias.txt");
    auto fc1_weight = read_param(dir + "/fc1.weight.txt");
    auto fc1_bias = read_param(dir + "/fc1.bias.txt");
    auto fc2_weight = read_param(dir + "/fc2.weight.txt");
    auto fc2_bias = read_param(dir + "/fc2.bias.txt");
    auto fc3_weight = read_param(dir + "/fc3.weight.txt");
    auto fc3_bias = read_param(dir + "/fc3.bias.txt");

    // log_file << "conv1_weight.size():" << conv1_weight.size() << std::endl;
    // log_file << "conv1_bias.size():" << conv1_bias.size() << std::endl;
    // log_file << "conv2_weight.size():" << conv2_weight.size() << std::endl;
    // log_file << "conv2_bias.size():" << conv2_bias.size() << std::endl;
    // log_file << "fc1_weight.size():" << fc1_weight.size() << std::endl;
    // log_file << "fc1_bias.size():" << fc1_bias.size() << std::endl;
    // log_file << "fc2_weight.size():" << fc2_weight.size() << std::endl;
    // log_file << "fc2_bias.size():" << fc2_bias.size() << std::endl;
    // log_file << "fc3_weight.size():" << fc3_weight.size() << std::endl;
    // log_file << "fc3_bias.size():" << fc3_bias.size() << std::endl << std::endl;

    // // 输出conv1_weight到log
    // log_file << "conv1_weight:" << std::endl;
    // // 6 * 1 * 5 * 5
    // for (int i = 0; i < 6; i++) {
    //     for (int j = 0; j < 25; j++) {
    //         log_file << std::setw(4) << conv1_weight[i * 25 + j] << " ";
    //     }
    //     log_file << std::endl;
    // }
    // log_file << std::endl;

    int batch_size = 10000;
    int num_batches = images.size() / batch_size;

    DataLoader data_loader(batch_size, images, labels);
    std::vector<int> predictions;

    // 将模型参数拷贝到显存中
    float* conv1_weight_gpu;
    float* conv1_bias_gpu;
    float* conv2_weight_gpu;
    float* conv2_bias_gpu;
    float* fc1_weight_gpu;
    float* fc1_bias_gpu;
    float* fc2_weight_gpu;
    float* fc2_bias_gpu;
    float* fc3_weight_gpu;
    float* fc3_bias_gpu;
    CUDA_CALL(cudaMalloc(&conv1_weight_gpu, conv1_weight.size() * sizeof(float)));
    CUDA_CALL(cudaMalloc(&conv1_bias_gpu, conv1_bias.size() * sizeof(float)));
    CUDA_CALL(cudaMalloc(&conv2_weight_gpu, conv2_weight.size() * sizeof(float)));
    CUDA_CALL(cudaMalloc(&conv2_bias_gpu, conv2_bias.size() * sizeof(float)));
    CUDA_CALL(cudaMalloc(&fc1_weight_gpu, fc1_weight.size() * sizeof(float)));
    CUDA_CALL(cudaMalloc(&fc1_bias_gpu, fc1_bias.size() * sizeof(float)));
    CUDA_CALL(cudaMalloc(&fc2_weight_gpu, fc2_weight.size() * sizeof(float)));
    CUDA_CALL(cudaMalloc(&fc2_bias_gpu, fc2_bias.size() * sizeof(float)));
    CUDA_CALL(cudaMalloc(&fc3_weight_gpu, fc3_weight.size() * sizeof(float)));
    CUDA_CALL(cudaMalloc(&fc3_bias_gpu, fc3_bias.size() * sizeof(float)));
    CUDA_CALL(cudaMemcpy(conv1_weight_gpu, conv1_weight.data(), conv1_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(conv1_bias_gpu, conv1_bias.data(), conv1_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(conv2_weight_gpu, conv2_weight.data(), conv2_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(conv2_bias_gpu, conv2_bias.data(), conv2_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(fc1_weight_gpu, fc1_weight.data(), fc1_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(fc1_bias_gpu, fc1_bias.data(), fc1_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(fc2_weight_gpu, fc2_weight.data(), fc2_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(fc2_bias_gpu, fc2_bias.data(), fc2_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(fc3_weight_gpu, fc3_weight.data(), fc3_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(fc3_bias_gpu, fc3_bias.data(), fc3_bias.size() * sizeof(float), cudaMemcpyHostToDevice));

    // // 测试拷贝是否成功
    // std::vector<float> conv1_weight_test;
    // conv1_weight_test.resize(conv1_weight.size());
    // cudaMemcpy(conv1_weight_test.data(), conv1_weight_gpu, conv1_weight.size() * sizeof(float), cudaMemcpyDeviceToHost);
    // log_file << "conv1_weight_test:" << std::endl;
    // for (int i = 0; i < 6; i++) {
    //     for (int j = 0; j < 25; j++) {
    //         log_file << std::setw(4) << conv1_weight_test[i * 25 + j] << " ";
    //     }
    //     log_file << std::endl;
    // }
    // log_file << std::endl;

    // 进行推理
    /*
        网络结构：
        conv1: 1 * 28 * 28 -> 6 * 24 * 24
        max_pooling2d: 6 * 24 * 24 -> 6 * 12 * 12
        conv2: 6 * 12 * 12 -> 16 * 8 * 8
        max_pooling2d: 16 * 8 * 8 -> 16 * 4 * 4
        fc1: 16 * 4 * 4 -> 120
        fc2: 120 -> 84
        fc3: 84 -> 10
    */

    // 开始计时，使用chrono计时
    auto start = std::chrono::high_resolution_clock::now();
    for (int batch_id = 0; batch_id < num_batches; batch_id++) {
        // log_file << "infering batch " << batch_id << "..." << std::endl;
        // 读取一个batch的数据
        auto batch = data_loader.getNextBatch();
        if (batch.empty()) {
            // log_file << "batch is empty!" << std::endl;
            return;
        }
        // log_file << "batch.size():" << batch.size() << std::endl;

        // 转化为一维
        std::vector<float> batch_1d;
        batch_1d.resize(batch.size() * 28 * 28);
        for (int i = 0; i < batch.size(); i++) {
            for (int j = 0; j < 28 * 28; j++) {
                batch_1d[i * 28 * 28 + j] = batch[i][j];
            }
        }
        // log_file << "batch_1d.size():" << batch_1d.size() << std::endl;
        // log_file << "batch_1d:" << std::endl;
        // for (int b = 0; b < batch.size(); b++) {
        //     log_file << "batch " << b << ":" << std::endl;
        //     for (int i = 0; i < 28; i++) {
        //         for (int j = 0; j < 28; j++) {
        //             log_file << std::setw(4) << batch_1d[b * 28 * 28 + i * 28 + j] << " ";
        //         }
        //         log_file << std::endl;
        //     }
        // }

        float* input_gpu, *conv1_output_gpu, *conv2_output_gpu, *fc1_output_gpu, *fc2_output_gpu, *fc3_output_gpu;
        float* max_pooling1_output_gpu, *max_pooling2_output_gpu;

        // 第一层卷积
        // 将输入特征图拷贝到显存中
        CUDA_CALL(cudaMalloc(&input_gpu, batch.size() * 1 * 28 * 28 * sizeof(float)));
        CUDA_CALL(cudaMemcpy(input_gpu, batch_1d.data(), batch.size() * 1 * 28 * 28 * sizeof(float), cudaMemcpyHostToDevice));

        // // 测试拷贝是否成功
        // float* input_test = new float[batch.size() * 28 * 28];
        // cudaMemcpy(input_test, input_gpu, batch.size() * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost);
        // log_file << "input_test:" << std::endl;
        // for (int b = 0; b < batch.size(); b++) {
        //     log_file << "batch " << b << ":" << std::endl;
        //     for (int i = 0; i < 28; i++) {
        //         for (int j = 0; j < 28; j++) {
        //             log_file << std::setw(4) << input_test[b * 28 * 28 + i * 28 + j] << " ";
        //         }
        //         log_file << std::endl;
        //     }
        // }
        // log_file << std::endl;
        // delete[] input_test;

        // 定义block和grid的大小
        dim3 block_dim(16, 16);
        dim3 grid_dim(2, 2, batch.size());

        CUDA_CALL(cudaMalloc(&conv1_output_gpu, batch.size() * 6 * 24 * 24 * sizeof(float)));
        // 调用卷积+ReLU激活函数的kernel函数
        conv2d_batch_relu<<<grid_dim, block_dim>>>(input_gpu, conv1_output_gpu, conv1_weight_gpu, conv1_bias_gpu, batch.size(), 28, 28, 1, 6, 5, 1, 0);
        CUDA_CALL(cudaDeviceSynchronize());

        // // 测试卷积结果
        // std::vector<float> conv1_output_test;
        // conv1_output_test.resize(batch.size() * 6 * 24 * 24);
        // cudaMemcpy(conv1_output_test.data(), conv1_output_gpu, batch.size() * 6 * 24 * 24 * sizeof(float), cudaMemcpyDeviceToHost);
        // log_file << "conv1_output_test:" << std::endl;
        // for (int b = 0; b < batch.size(); b++) {
        //     log_file << "batch " << b << ":" << std::endl;
        //     for (int i = 0; i < 6; i++) {
        //         log_file << "channel " << i << ":" << std::endl;
        //         for (int j = 0; j < 24; j++) {
        //             for (int k = 0; k < 24; k++) {
        //                 log_file << std::setw(4) << conv1_output_test[b * 6 * 24 * 24 + i * 24 * 24 + j * 24 + k] << " ";
        //             }
        //             log_file << std::endl;
        //         }
        //         log_file << std::endl;
        //     }
        // }


        CUDA_CALL(cudaFree(input_gpu));

        // 第一层池化
        cudaMalloc(&max_pooling1_output_gpu, batch.size() * 6 * 12 * 12 * sizeof(float));
        // 调用池化函数，！！！！！！ blockdim可以优化
        max_pooling2d_batch<<<grid_dim, block_dim>>>(conv1_output_gpu, max_pooling1_output_gpu, batch.size(), 2, 24, 24, 6, 2);
        cudaDeviceSynchronize();

        // // 测试池化结果
        // std::vector<float> max_pooling1_output_test;
        // max_pooling1_output_test.resize(batch.size() * 6 * 12 * 12);
        // cudaMemcpy(max_pooling1_output_test.data(), max_pooling1_output_gpu, batch.size() * 6 * 12 * 12 * sizeof(float), cudaMemcpyDeviceToHost);
        // log_file << "max_pooling1_output_test:" << std::endl;
        // for (int b = 0; b < batch.size(); b++) {
        //     log_file << "batch " << b << ":" << std::endl;
        //     for (int i = 0; i < 6; i++) {
        //         log_file << "channel " << i << ":" << std::endl;
        //         for (int j = 0; j < 12; j++) {
        //             for (int k = 0; k < 12; k++) {
        //                 log_file << std::setw(4) << max_pooling1_output_test[b * 6 * 12 * 12 + i * 12 * 12 + j * 12 + k] << " ";
        //             }
        //             log_file << std::endl;
        //         }
        //         log_file << std::endl;
        //     }
        // }

        cudaFree(conv1_output_gpu);

        // 第二层卷积
        cudaMalloc(&conv2_output_gpu, batch.size() * 16 * 8 * 8 * sizeof(float));
        // 调用卷积+ReLU激活函数的kernel函数
        conv2d_batch_relu<<<grid_dim, block_dim>>>(max_pooling1_output_gpu, conv2_output_gpu, conv2_weight_gpu, conv2_bias_gpu, batch.size(), 12, 12, 6, 16, 5, 1, 0);
        cudaDeviceSynchronize();
        cudaFree(max_pooling1_output_gpu);

        // 第二层池化
        CUDA_CALL(cudaMalloc(&max_pooling2_output_gpu, batch.size() * 16 * 4 * 4 * sizeof(float)));
        // 调用池化函数，！！！！！！ blockdim可以优化
        max_pooling2d_batch<<<grid_dim, block_dim>>>(conv2_output_gpu, max_pooling2_output_gpu, batch.size(), 2, 8, 8, 16, 2);
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaFree(conv2_output_gpu));

        // // 测试第二层池化结果
        // std::vector<float> max_pooling2_output_test;
        // max_pooling2_output_test.resize(batch.size() * 16 * 4 * 4);
        // cudaMemcpy(max_pooling2_output_test.data(), max_pooling2_output_gpu, batch.size() * 16 * 4 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        // log_file << "max_pooling2_output_test:" << std::endl;
        // for (int b = 0; b < batch.size(); b++) {
        //     log_file << "batch " << b << ":" << std::endl;
        //     for (int i = 0; i < 16; i++) {
        //         log_file << "channel " << i << ":" << std::endl;
        //         for (int j = 0; j < 4; j++) {
        //             for (int k = 0; k < 4; k++) {
        //                 log_file << std::setw(4) << max_pooling2_output_test[b * 16 * 4 * 4 + i * 4 * 4 + j * 4 + k] << " ";
        //             }
        //             log_file << std::endl;
        //         }
        //         log_file << std::endl;
        //     }
        // }

        dim3 block_dim_fc1(64);
        dim3 grid_dim_fc1(2, batch.size());
        // 第一层全连接
        cudaMalloc(&fc1_output_gpu, batch.size() * 120 * sizeof(float));
        // 调用全连接+ReLU激活函数的kernel函数
        fully_connected_batch_relu<<<grid_dim_fc1, block_dim_fc1>>>(max_pooling2_output_gpu, fc1_output_gpu, fc1_weight_gpu, fc1_bias_gpu, batch.size(), 16 * 4 * 4, 120);
        cudaDeviceSynchronize();

        // // 测试第一层全连接结果
        // std::vector<float> fc1_output_test;
        // fc1_output_test.resize(batch.size() * 120);
        // cudaMemcpy(fc1_output_test.data(), fc1_output_gpu, batch.size() * 120 * sizeof(float), cudaMemcpyDeviceToHost);
        // log_file << "fc1_output_test:" << std::endl;
        // for (int b = 0; b < batch.size(); b++) {
        //     log_file << "batch " << b << ":" << std::endl;
        //     for (int i = 0; i < 120; i++) {
        //         log_file << std::setw(4) << fc1_output_test[b * 120 + i] << " ";
        //     }
        //     log_file << std::endl;
        // }

        cudaFree(max_pooling2_output_gpu);

        dim3 block_dim_fc2(64);
        dim3 grid_dim_fc2(2, batch.size());
        // 第二层全连接
        cudaMalloc(&fc2_output_gpu, batch.size() * 84 * sizeof(float));
        // 调用全连接+ReLU激活函数的kernel函数
        fully_connected_batch_relu<<<grid_dim_fc2, block_dim_fc2>>>(fc1_output_gpu, fc2_output_gpu, fc2_weight_gpu, fc2_bias_gpu, batch.size(), 120, 84);
        cudaDeviceSynchronize();
        cudaFree(fc1_output_gpu);

        dim3 block_dim_fc3(16);
        dim3 grid_dim_fc3(1, batch.size());
        // 第三层全连接
        cudaMalloc(&fc3_output_gpu, batch.size() * 10 * sizeof(float));
        // 调用全连接的kernel函数
        fully_connected_batch<<<grid_dim_fc3, block_dim_fc3>>>(fc2_output_gpu, fc3_output_gpu, fc3_weight_gpu, fc3_bias_gpu, batch.size(), 84, 10);
        cudaDeviceSynchronize();
        cudaFree(fc2_output_gpu);

        // 将输出特征图拷贝到主机端
        float* fc3_output = new float[batch.size() * 10];
        cudaMemcpy(fc3_output, fc3_output_gpu, batch.size() * 10 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(fc3_output_gpu);

        // 存储每个batch的预测结果
        for (int i = 0; i < batch.size(); i++) {
            float max_value = -FLT_MAX;
            int max_index = -1;
            for (int j = 0; j < 10; j++) {
                if (fc3_output[i * 10 + j] > max_value) {
                    max_value = fc3_output[i * 10 + j];
                    max_index = j;
                }
            }
            // std::cout << "prediction: " << max_index << ", label: " << labels[batch_id * batch.size() + i] << std::endl;
            predictions.push_back(max_index);
        }

        delete[] fc3_output;
    };

    data_loader.reset();
    
	
	// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
	cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 计算准确率
    int correct = 0;
    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << (float)correct / predictions.size() << std::endl;

    return 0;
}