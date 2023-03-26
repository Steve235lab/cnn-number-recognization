import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from number_recognition_model import NumberRecognitionModel

# 加载训练好的模型参数
model = NumberRecognitionModel().to("mps")
model.load_state_dict(torch.load("number_recognition_model.pt"))
model.eval()

# 读取测试图片并进行预处理
image_path = "./test_img/manual_roi_2.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图形式读取图片
image = cv2.resize(image, (28, 28))  # 调整图片大小为28x28，与MNIST数据集相同
image = Image.fromarray(image)
image = transforms.ToTensor()(image)  # 转换为张量
image = image.unsqueeze(0).to("mps")  # 增加批次维度并将图片传递到设备

# 进行预测
with torch.no_grad():
    output = model(image)
    _, prediction = torch.max(output.data, 1)
    print(f"Predicted number: {prediction.item()}")
