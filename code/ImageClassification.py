import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

from torchvision import transforms
from tqdm import tqdm

import numpy as np
from PIL import Image

from Resnet import resnet34, resnet101

import matplotlib.pyplot as plt

# 数据加载
class CifarDataset(Dataset):
    def __init__(self, img_path, info_path, is_test = None):
        self.image, self.label = self.LoadImage(img_path, info_path, is_test)

    def __getitem__(self, index):
        # 添加getitem函数的相关内容
        image = self.image[index]
        label = self.label[index]
        return image, label

    def __len__(self):
        # 添加len函数的相关内容
        return len(self.image)

    def LoadImage(self, img_path, info_path, is_test = None):
        img_list = []
        label_list = []
        with open(info_path, 'r') as f:
            for line in f:
                words = line.strip('\n').split(' ')
                img = Image.open(os.path.join(img_path, words[0]))
                # print(words)
                # print(img)
                r'''
                <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x206E131B100>
                ['img_00492.png', '0\n']
                使用strip去掉'\n'
                '''
                img_np = np.array(img).transpose((2, 0, 1))
                # print(img_np.dtype)
                # (32, 32, 3) BGR排列
                # uint8
                image_np = img_np.astype(np.float32) / 255.0
                img_list.append(image_np)
                # PyTorch通常会使用 torch.float32（在大多数GPU上）或 torch.float64（在CPU上）
                if is_test == None:
                    label = int(words[1])
                    label_np = np.array(label)
                    # .astype(np.float32)
                    label_list.append(label_np)
                else:
                    # 测试集标签先用随机数填充
                    label_list.append(np.random.randint(0, 10))

        images = torch.from_numpy(np.array(img_list))
        labels = torch.from_numpy(np.array(label_list))
        print(images.shape)
        print(labels.shape)
        # torch.Size([40000, 32, 32, 3])  val and test: torch.Size([10000, 32, 32, 3])
        # torch.Size([40000])
        return images, labels

def collate_fn(batch):
    '''
    :param samples:
    :return:
    collate_fn是Dataloader里面的参数，可以用来自定义处理一个batch，
    在nlp中可以设置每个batch的padding不一样长度。
    '''
    # print(batch.shape) # 'list' object has no attribute 'shape'
    # 数据增强，数据量不变，但是每个epoch进行下列的操作，有些操作是随机进行的
    augmentation_transform = transforms.Compose([
        # transforms.RandomResizedCrop(32),  # 随机裁剪并缩放 这里的数值size指resize到设定好的size
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    images, labels = zip(*batch) # 'tuple' object has no attribute 'shape' 输出tuple
    # zip接收多个迭代器以元组为元素合并为一个列表，zip(*)是解压，将以元组为元素的列表拆成多个列表

    # PyTorch 中的张量默认采用 N×D×H×W 的顺序
    # PIL 默认(h,w,c)

    # print(images[0].shape)
    # print(augmentation_transform(images[0]).shape)

    # 应用数据增强变换到每个图像
    # augmented_images = [augmentation_transform(transforms.ToPILImage()(image)) for image in images]
    # image_tensor = transforms.ToTensor()(augmented_images)
    augmented_images = [augmentation_transform(image) for image in images]

    images_aug = torch.stack(augmented_images, 0)
    labels_tensor = torch.tensor(labels, dtype=torch.float32) # 此时labels是元组，转为tensor
    # print(images_aug)

    return images_aug, labels_tensor

# 构建模型
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # 定义模型的网络结构
        self.fc1 = nn.Linear(32 * 32 * 3, 32 * 8 * 3)
        self.fc2 = nn.Linear(32 * 8 * 3, 8 * 8 * 3)
        self.fc3 = nn.Linear(8 * 8 * 3, 4 * 4 * 3)
        self.fc4 = nn.Linear(4 * 4 * 3, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        # print(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        # 直接输出原始得分,无需relu，也无需softmax,因为CrossEntropyLoss内部会计算softmax
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义模型的网络结构 input(3,32,32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool1= nn.MaxPool2d(kernel_size=2, stride=2)  # (32, 16, 16)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (64, 8, 8)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 8 * 8 * 8)
        self.fc2 = nn.Linear(8 * 8 * 8, 8 * 8)
        self.fc3 = nn.Linear(8 * 8, 10)

    def forward(self, x):
        # 定义模型前向传播的内容
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义 train 函数
def train(save_path, epoch_num = 10, val_num = 1, batch_size = 16):
    # 参数设置
    model.train()
    max_val_acc = 0
    test_acc = []
    loss_pic = []

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        tqdm_iterator = tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{epoch_num}')
        total_loss = []
        total_true = []
        for data in tqdm_iterator:
            data_batch, target_batch = data[0].to(device), data[1].to(device)
            # print(data_batch, target_batch)

            optimizer.zero_grad()
            # Forward
            prob_batch = model(data_batch)
            # print(prob_batch)
            # print(target_batch)
            # crossentropy封装了onehot
            # Backward
            # crossentropy要求output为floattorch，label为longtorch
            loss = criterion(prob_batch, target_batch.long())
            loss.backward()
            # Update
            optimizer.step()

            # 实时显示正确率和loss
            predict_batch = torch.tensor([torch.argmax(_) for _ in prob_batch]).to(device)
            # print()
            total_true.append(torch.sum(predict_batch == target_batch).item())
            total_loss.append(loss.item())

            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss),
                                      acc=sum(total_true) / (batch_size * len(total_true)))
        tqdm_iterator.close() # 记得关闭tqdm
        loss_pic.append(sum(total_loss) / len(total_loss))

        # 模型训练n轮之后进行验证
        if epoch % val_num == 0:
            valid_acc = validation()
            print(f"epoch: {epoch+1}, valid accuracy: {valid_acc * 100:.2f}%")
            test_acc.append(valid_acc)
            if valid_acc >= max_val_acc:
                max_val_acc = valid_acc
                torch.save(model.state_dict(), save_path)
    # x = np.linspace(1, epoch_num+1)
    x = np.arange(1, epoch_num+1, 1)
    acc_np = np.array(test_acc)
    los_np = np.array(loss_pic)

    plt.figure(1)
    plt.subplot(1, 2, 1)  # 图一包含1行2列子图，当前画在第一行第一列图上
    plt.plot(x, acc_np, label="accuracy")
    plt.title('validation accuracy')
    plt.xlabel('epoch')

    plt.figure(1)
    plt.subplot(1, 2, 2)  # 当前画在第一行第2列图上
    plt.plot(x, los_np, label='loss')
    plt.title('training loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

# 定义 validation 函数
def validation():
    correct = 0
    total = 0
    with torch.no_grad():
        tqdm_iterator = tqdm(val_loader, dynamic_ncols=True, desc=f'test')
        for data in tqdm_iterator:
            data_batch, target_batch = data[0].to(device), data[1].to(device)
            # 在这一部分撰写验证的内容
            prob_batch = model(data_batch)
            predict_batch = torch.tensor([torch.argmax(_) for _ in prob_batch]).to(device)
            correct += torch.sum(predict_batch == target_batch).item()
            total += len(target_batch)
        tqdm_iterator.close()
    valid_acc = correct / total
    return valid_acc

# 定义 test 函数
def test(output_path):
    model.eval()
    with torch.no_grad():
        with open(output_path, 'w') as f:
            tqdm_iterator = tqdm(val_loader, dynamic_ncols=True, desc=f'test')
            for data in tqdm_iterator:
                data_batch = data[0].to(device)
                # 在这一部分撰写验证的内容
                prob_batch = model(data_batch)
                predict_batch = torch.tensor([torch.argmax(_) for _ in prob_batch])

                for element in predict_batch:
                    f.write(str(element.item()) + '\n')

            tqdm_iterator.close()

if __name__ == "__main__":
    # 误写为dateset，为戒
    cur_path = os.path.dirname(__file__)
    path = 'Dataset'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCH = 20
    VAL_GAP = 1
    LEARNING_RATE = 0.0005  # Adam默认值0.001，这个不能超过0.01

    mlp_save_path = 'mlp_model.pth'
    cnn_save_path = 'cnn_model.pth'
    res34_save_path = 'res34_model_aug.pth'
    res101_save_path = 'res101_model.pth'
    cur_path = os.path.dirname(__file__)
    img_path = os.path.join(cur_path, path, 'image')

    train_path = os.path.join(cur_path, path, 'trainset.txt')
    valid_path = os.path.join(cur_path, path, 'validset.txt')
    test_path = os.path.join(cur_path, path, 'testset.txt')

    # 构建数据集
    train_set = CifarDataset(img_path, train_path, is_test=None)
    val_set = CifarDataset(img_path, valid_path, is_test=None)
    test_set = CifarDataset(img_path, test_path, is_test=True)

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=None)
    test_loader = DataLoader(dataset=test_set, shuffle=None)
    # , collate_fn=collate_fn
    # 测试看collate效果
    # iter_loader = iter(train_loader)
    # batch1 = next(iter_loader)
    # print(batch1)
    # collate_fn在创建dataloader的时候没操作，在遍历时候操作，属于在线增强


    # # ------------------------------------------------------------------以下为mlp使用的训练测试代码
    # 初始化模型对象
    model = MLPNet().to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 保存与载入
    # model_state_dict = torch.load(mlp_save_path)
    # model.load_state_dict(model_state_dict)
    # # 模型训练
    train(save_path=mlp_save_path, epoch_num=EPOCH, val_num=VAL_GAP, batch_size = BATCH_SIZE)
    #
    output_path = os.path.join(cur_path, 'predict_labels_MLP_1120211392.txt')
    # 对模型进行测试，并生成预测结果
    test(output_path)
    # # ------------------------------------------------------------------以下为SimpleCNN使用的训练测试代码
    # # 初始化模型对象
    model = SimpleCNN().to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # # 保存与载入
    # model_state_dict = torch.load(cnn_save_path)
    # model.load_state_dict(model_state_dict)
    # # 模型训练
    train(save_path=cnn_save_path, epoch_num=EPOCH, val_num=VAL_GAP, batch_size=BATCH_SIZE)
    output_path = os.path.join(cur_path, 'predict_labels_CNN_1120211392.txt')
    test(output_path)

    # # ------------------------------------------------------------------以下为ResNet34用的训练测试代码
    # 加载增强版的数据
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=None, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_set, shuffle=None, collate_fn=collate_fn)
    # # 初始化模型对象
    model = resnet34(num_classes=10).to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # # 保存与载入
    # model_state_dict = torch.load(res34_save_path)
    # model.load_state_dict(model_state_dict)
    # # 模型训练
    train(save_path=res34_save_path, epoch_num=EPOCH, val_num=VAL_GAP, batch_size=BATCH_SIZE)
    output_path = os.path.join(cur_path, 'predict_labels_ResNet_1120211392.txt')
    # 对模型进行测试，并生成预测结果，记得改output_path
    test(output_path)

    # # ------------------------------------------------------------------以下为ResNet101用的训练测试代码
    # # # 初始化模型对象
    model = resnet101(num_classes=10).to(device)
    # # 保存与载入
    # model_state_dict = torch.load(res101_save_path)
    # model.load_state_dict(model_state_dict)
    # model.to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # # 模型训练
    train(save_path=res101_save_path, epoch_num=EPOCH, val_num=VAL_GAP, batch_size=BATCH_SIZE)
    output_path = os.path.join(cur_path, 'res101result.txt')
    # 对模型进行测试，并生成预测结果，记得改output_path
    test(output_path)
