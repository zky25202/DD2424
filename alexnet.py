import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
def printPictureAndSave2(epoch_list,loss_train_list,loss_validation_list,accuracy_train_list,accuracy_validation_list):	
	

	
	plt.plot(epoch_list,loss_validation_list,color='red',label='validation')
	plt.plot(epoch_list,loss_train_list,color='blue',label='train')
	
	plt.title('loss graph\n')
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.legend()
	
	plt.savefig('./AlexLoss.png')
	plt.show()

	
	
	#draw accuracy picture
	plt.plot(epoch_list,accuracy_train_list,color='blue',label='train')
	plt.plot(epoch_list,accuracy_validation_list,color='red',label='validation')
	plt.title('accuracy graph\n')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.legend()
	#plt.savefig('./pic/accuracy.png')
	plt.savefig('./AlexAcc.png')
	plt.show()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),
            #nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 192, 5, 1, 2),
           # nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, 1, 1),
           # nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
           # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 128 * 6 * 6))
        return x


transform = transforms.Compose([

    transforms.Resize(224),
    #transforms.RandomHorizontalFlip(),
    #transforms.CenterCrop(224),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.465, 0.406], [0.229, 0.224, 0.225]),
])

##############
batch_size=64
############

train_data = dset.CIFAR10(root='./data', train=True, transform=transform, download=False)
test_data = dset.CIFAR10(root='./data', train=False, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
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

model = AlexNet()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 0.05)

# these list are used to draw graph
train_loss_list=[]
val_loss_list=[]

train_accuracy_list=[]
val_accuracy_list=[]

epoch_list=[]
i=0
for epoch in range(20 ):
    total_loss_train = 0
    total_loss_val = 0
    train_correct = 0
    val_correct = 0

    i+=1
    for batch in train_loader:
        images, labels = batch
        outs = model(images.to(device))
        loss = F.cross_entropy(outs, labels.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #compute the train loss and val loss for each epoch
    
    #compute the loss for training data 
    for batch in train_loader:
        images, labels = batch
        outs = model(images.to(device))
        loss = F.cross_entropy(outs, labels.to(device))
        total_loss_train += loss.item()
        train_correct += outs.argmax(dim=1).eq(labels.to(device)).sum().item()
    
    total_loss_train = total_loss_train / (train_len*0.8)
    train_correct /= (train_len*0.8)
    train_accuracy_list.append(train_correct)
    train_loss_list.append(total_loss_train)

    #compute the loss for the validation data
    for batch in validation_loader:
        images, labels = batch
        outs = model(images.to(device))
        loss = F.cross_entropy(outs, labels.to(device))
        total_loss_val += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        val_correct += outs.argmax(dim=1).eq(labels.to(device)).sum().item()
    
    total_loss_val = total_loss_val / (train_len/5)
    val_correct = val_correct / (train_len/5)


    val_loss_list.append(total_loss_val)
    val_accuracy_list.append(val_correct)
    epoch_list.append(i)

    print('train_loss:', total_loss_train,\
        ' validation_loss:', total_loss_val, \
            '  train_accuracy:', train_correct,\
                 ' validation_accuracy  :',val_correct)

printPictureAndSave2(epoch_list,train_loss_list,val_loss_list,train_accuracy_list,\
    val_accuracy_list)



test_correct = 0   
for batch in test_loader:
    images, labels = batch
    out = model(images.to(device))
    test_correct += out.argmax(dim=1).eq(labels.to(device)).sum().item()
test_correct = test_correct / test_len

print('test correct accuracy',test_correct )
