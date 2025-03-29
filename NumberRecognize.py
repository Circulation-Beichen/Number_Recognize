import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import os
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from PIL import Image

# 设置中文字体
def set_chinese_font():
    system = platform.system()
    font_found = False
    
    if system == 'Windows':
        # Windows系统字体路径
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
        for font_name in font_list:
            try:
                font_path = f"C:\\Windows\\Fonts\\{font_name}.ttf"
                if os.path.exists(font_path):
                    font_prop = FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = font_prop.get_name()
                    font_found = True
                    print(f"使用Windows字体: {font_name}")
                    break
            except Exception as e:
                print(f"尝试加载字体{font_name}失败: {e}")
                continue
    
    elif system == 'Linux':
        # Linux系统字体路径
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc'
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN']
                font_found = True
                print(f"使用Linux字体: {font_path}")
                break
    
    elif system == 'Darwin':  # macOS
        # macOS系统字体路径
        font_list = ['PingFang.ttc', 'STHeiti Light.ttc', 'Heiti SC Light.ttc']
        for font_name in font_list:
            try:
                font_path = f"/System/Library/Fonts/{font_name}"
                if os.path.exists(font_path):
                    plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti Light', 'Heiti SC Light']
                    font_found = True
                    print(f"使用macOS字体: {font_name}")
                    break
            except:
                continue
    
    if not font_found:
        # 如果找不到系统字体，使用matplotlib内置字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("使用matplotlib内置字体")

# 在开始时设置字体
set_chinese_font()

# 定义更高级的ResNet模型
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DigitResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitResNet, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(64, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 加载MNIST数据集并训练模型
def train_model(model_path='digit_model.pth'):
    # 检查是否已有训练好的模型
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DigitResNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    
    print("训练新模型...")
    
    # 设置数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载MNIST数据集
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitResNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'测试准确率: {accuracy:.2f}%')
    
    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")
    
    model.eval()
    return model

# 预处理图像
def preprocess_image(image_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 显示原始图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.title('原始图像')
    plt.imshow(img, cmap='gray')
    
    # 自适应阈值二值化 - 处理不同光照条件
    # 如果图像是白底黑字，需要反转颜色
    is_inverted = np.mean(img) > 128  # 判断是否为白底黑字
    
    if is_inverted:
        img = cv2.bitwise_not(img)  # 反转颜色使其变为黑底白字
    
    # 去噪
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 自适应阈值二值化
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2)
    
    plt.subplot(1, 4, 2)
    plt.title('二值化后')
    plt.imshow(img, cmap='gray')
    
    # 形态学操作，去除噪点和加强特征
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算去除小噪点
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算填充小孔
    
    plt.subplot(1, 4, 3)
    plt.title('形态学处理')
    plt.imshow(img, cmap='gray')
    
    # 寻找轮廓，裁剪到只包含数字的区域
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大的轮廓（假设最大的是数字）
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # 确保裁剪区域不会太小
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        
        # 裁剪
        img = img[y:y+h, x:x+w]
    
    # 调整大小为28x28，保持纵横比
    # 首先创建一个空白正方形图像
    side_length = max(img.shape)
    square_img = np.zeros((side_length, side_length), dtype=np.uint8)
    
    # 将原图像居中放置在正方形中
    offset_h = (side_length - img.shape[0]) // 2
    offset_w = (side_length - img.shape[1]) // 2
    square_img[offset_h:offset_h+img.shape[0], offset_w:offset_w+img.shape[1]] = img
    
    # 调整到28x28像素
    img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 显示处理后的图像
    plt.subplot(1, 4, 4)
    plt.title('最终图像(28x28)')
    plt.imshow(img, cmap='gray')
    plt.tight_layout()
    plt.show()
    
    # 转换为PyTorch张量并标准化
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])(Image.fromarray(img))
    
    return img_tensor.unsqueeze(0)  # 添加batch维度

# 识别数字
def recognize_digit(model, img_tensor):
    # 确保模型处于评估模式
    model.eval()
    
    # 将图像移动到与模型相同的设备上
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        digit = torch.argmax(probabilities).item()
        confidence = probabilities[digit].item() * 100
    
    # 创建一个更美观的结果显示
    plt.figure(figsize=(12, 6))
    
    # 左侧显示识别结果
    plt.subplot(1, 2, 1)
    plt.text(0.5, 0.5, str(digit), fontsize=120, ha='center', va='center')
    plt.title(f'识别结果: {digit}', fontsize=16)
    plt.text(0.5, 0.25, f'置信度: {confidence:.2f}%', fontsize=16, ha='center')
    plt.axis('off')
    
    # 右侧显示预测概率分布
    plt.subplot(1, 2, 2)
    
    # 创建彩色条形图，突出显示最高概率
    probabilities_np = probabilities.cpu().numpy() * 100
    colors = ['lightgray'] * 10
    colors[digit] = 'royalblue'
    
    bars = plt.bar(range(10), probabilities_np, color=colors)
    
    # 在每个条形上方添加概率百分比
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 5:  # 只显示大于5%的概率值
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(range(10), fontsize=12)
    plt.xlabel('数字', fontsize=14)
    plt.ylabel('概率 (%)', fontsize=14)
    plt.title('预测概率分布', fontsize=16)
    plt.ylim(0, 110)  # 留出空间显示文本
    
    plt.tight_layout()
    plt.show()
    
    print(f'识别结果: {digit}，置信度: {confidence:.2f}%')
    
    return digit, confidence

def main():
    # 定义模型文件路径
    model_path = 'digit_model.pth'
    
    print("加载/训练ResNet模型...")
    model = train_model(model_path)
    
    print("处理测试图像...")
    try:
        test_img = preprocess_image("Number.jpg")
        
        print("识别数字...")
        digit, confidence = recognize_digit(model, test_img)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
