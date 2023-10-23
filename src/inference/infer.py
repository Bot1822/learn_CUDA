import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os

log_file = open('log_py.txt', 'w')

# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

script_dir = os.path.dirname(__file__)  # 获取脚本所在的目录

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载数据集
trainset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, '../../data'), download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, '../../data'), download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

# 创建模型
model = LeNet()
model = model.to('cuda')

# 训练时读出格式为：for name, param in model.named_parameters():
    # np.savetxt(os.path.join(script_dir, f'./{name}.txt'), param.detach().cpu().numpy().flatten())

# 读取模型参数
# 网络结构：conv1：6个5*5的卷积核，conv2：16个5*5的卷积核，fc1：120个神经元，fc2：84个神经元，fc3：10个神经元
# 读取模型参数
for name, param in model.named_parameters():
    param.data = torch.from_numpy(np.loadtxt(os.path.join(script_dir, f'./{name}.txt'))).float()

# 参数转为float类型
for name, param in model.named_parameters():
    param.data = param.data.float()
    
# 将模型参数转为正确的维度
model.conv1.weight.data = model.conv1.weight.data.reshape(6, 1, 5, 5)
model.conv2.weight.data = model.conv2.weight.data.reshape(16, 6, 5, 5)
model.fc1.weight.data = model.fc1.weight.data.reshape(120, 16 * 4 * 4)
model.fc2.weight.data = model.fc2.weight.data.reshape(84, 120)
model.fc3.weight.data = model.fc3.weight.data.reshape(10, 84)

# 转到GPU上
model = model.to('cuda')
    
# 输出模型参数到log文件
for name, param in model.named_parameters():
    log_file.write(f'{name}:\n')
    log_file.write(f'{param}\n')
    
# # 在测试集上前向传播一层，只进行第一层卷积，查看输出
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images, labels = images.to('cuda'), labels.to('cuda')
#         output = model.conv1(images)
#         log_file.write(f'output shape: {output.shape}\n')
#         log_file.write(f'output: {output}\n')
#         break
    
# 
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        output = F.relu(model.conv1(images))
        output = model.pool(output)
        output = F.relu(model.conv2(output))
        output = model.pool(output)
        log_file.write(f'output shape: {output.shape}\n')
        log_file.write(f'output: {output}\n')
        output = output.view(-1, 16 * 4 * 4)
        output = F.relu(model.fc1(output))
        log_file.write(f'output shape: {output.shape}\n')
        log_file.write(f'output: {output}\n')
        break

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(correct)
print(correct/total)  
