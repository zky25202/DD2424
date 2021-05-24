import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def printPictureAndSave2(epoch_list, loss_train_list, loss_validation_list, accuracy_train_list,
                         accuracy_validation_list):
    plt.plot(epoch_list, loss_validation_list, color='red', label='validation')
    plt.plot(epoch_list, loss_train_list, color='blue', label='train')

    plt.title('loss graph\n')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('./ResLoss.png')
    plt.show()

    # draw accuracy picture
    plt.plot(epoch_list, accuracy_train_list, color='blue', label='train')
    plt.plot(epoch_list, accuracy_validation_list, color='red', label='validation')
    plt.title('accuracy graph\n')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    # plt.savefig('./pic/accuracy.png')
    plt.savefig('./ResAcc.png')
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                                            nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, block):
        super(ResNet34, self).__init__()
        self.first = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(3, 1, 1))
        self.layer1 = self.make_layer(block, 64, 64, 3, 1)
        self.layer2 = self.make_layer(block, 64, 128, 4, 2)
        self.layer3 = self.make_layer(block, 128, 256, 6, 2)
        self.layer4 = self.make_layer(block, 256, 512, 3, 2)
        self.avg_pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512, 10)

    def make_layer(self, block, in_channels, out_channels, block_num, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for i in range(block_num - 1):
            layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x



root = './data'
transformfun = transforms.Compose([
    transforms.ToTensor(),
])

##############
batch_size=64
############

train_data = dset.CIFAR10(root, train=True, transform=transformfun, target_transform=None, download=True)
test_data = dset.CIFAR10(root, train=False, transform=transformfun, target_transform=None, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
train_len = len(train_data)
test_len = len(test_data)
indices = range(len(train_data))
indices_train = indices[:40000]
indices_val = indices[40000:]
sampler_train = torch.utils.data.sampler.SubsetRandomSampler(indices_train)
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
train_loader = torch.utils.data.DataLoader(dataset =train_data,
                                                batch_size = batch_size,
                                                sampler = sampler_train
                                               )
validation_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          sampler = sampler_val
                                         )


print('train_length:',train_len*0.8, ' validation_length:',train_len*0.2,' test_length:',test_len)

model = ResNet34(ResBlock)
model = model.cuda()
optimizer = optim.Adam(model.parameters(),weight_decay= 0.001)

# these list are used to draw graph
train_loss_list=[]
val_loss_list=[]

train_accuracy_list=[]
val_accuracy_list=[]

epoch_list=[]
i=0

for epoch in range(20):
    total_loss_train = 0
    total_loss_val = 0
    train_correct = 0
    val_correct = 0

    i += 1
    for batch in train_loader:
        images, labels = batch
        out = model(images.cuda())
        loss = F.cross_entropy(out, labels.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # compute the train loss and val loss for each epoch

        # compute the loss for training data
    for batch in train_loader:
        images, labels = batch
        outs = model(images.to(device))
        loss = F.cross_entropy(outs, labels.to(device))
        total_loss_train += loss.item()
        train_correct += outs.argmax(dim=1).eq(labels.to(device)).sum().item()

    total_loss_train = total_loss_train / (train_len * 0.8)
    train_correct /= (train_len * 0.8)
    train_accuracy_list.append(train_correct)
    train_loss_list.append(total_loss_train)

        # compute the loss for the validation data
    for batch in validation_loader:
        images, labels = batch
        outs = model(images.to(device))
        loss = F.cross_entropy(outs, labels.to(device))
        total_loss_val += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        val_correct += outs.argmax(dim=1).eq(labels.to(device)).sum().item()

    total_loss_val = total_loss_val / (train_len / 5)
    val_correct = val_correct / (train_len / 5)

    val_loss_list.append(total_loss_val)
    val_accuracy_list.append(val_correct)
    epoch_list.append(i)

    print('train_loss:', total_loss_train, \
              ' validation_loss:', total_loss_val, \
              ' train_accuracy:', train_correct, \
              ' validation_accuracy:', val_correct)

printPictureAndSave2(epoch_list, train_loss_list, val_loss_list, train_accuracy_list, \
                             val_accuracy_list)

test_correct = 0
for batch in test_loader:
    images, labels = batch
    out = model(images.to(device))
    test_correct += out.argmax(dim=1).eq(labels.to(device)).sum().item()
test_correct = test_correct / test_len

print('test correct accuracy', test_correct)