import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.init as weight_init
import matplotlib.pyplot as plt
import pdb
from  Classification.load_and_process import load_fer2013
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#Loading the train set file
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=preprocess,
                               download=True)
#Loading the test set file
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=preprocess)

print(len(train_dataset))
print(train_dataset[0][0].size())

#Plotting
plt.figure()
plt.axis('off')
plt.imshow(train_dataset[110][0].squeeze(),cmap='gray')

# Dataloader
batch_size = 100

#loading the train dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

#loading the test dataset
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

#Plotting 10 digits
pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray")


# Parameters
input_size = 784
hidden_size = 256
num_classes = 10
num_epochs = 25
learning_rate = 0.01
momentum_rate = 0.9

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

        #Weight Initialization
        for m in self.modules():
            if isinstance(m,nn.Linear):
                weight_init.xavier_normal_(m.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

net = Net(input_size, hidden_size, num_classes)

#Other functions
def loss_plot(losses):
    max_epochs = len(losses)
    times = list(range(1, max_epochs+1))
    plt.figure(figsize=(30, 7))
    plt.xlabel("epochs")
    plt.ylabel("cross-entropy loss")
    return plt.plot(times, losses)


#Loss function
criterion = nn.CrossEntropyLoss()

# SGD for Optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum_rate)

##RMSProp
# torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
##Adam
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
##Adagrad
# torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using CUDA ', use_cuda)

#Transfer to GPU device if available
net = net.to(device)
net

# Training
epochLoss = []
for epoch in range(num_epochs):

    total_loss = 0
    cntr = 1
    # For each batch of images in train set
    for i, (images, labels) in enumerate(train_loader):

        images = images.view(-1, 28*28)
        labels = labels

        images, labels = images.to(device), labels.to(device)

        # Initialize gradients to 0
        optimizer.zero_grad()

        # Forward pass (this calls the "forward" function within Net)
        outputs = net(images)

        # Find the loss
        loss = criterion(outputs, labels)

        # Backward pass (Find the gradients of all weights using the loss)
        loss.backward()

        # Update the weights using the optimizer
        # For e.g.: w = w - (delta_w)*lr
        optimizer.step()

        total_loss += loss.item()
        cntr += 1

    print('Epoch [%d/%d] Loss:%.4f'%(epoch+1, num_epochs, total_loss/cntr) )
    epochLoss.append(total_loss/cntr)

correct = 0
total = 0

# For each batch of images in test set
with torch.no_grad():
    for images, labels in test_loader:

        # Get the images
        images = images.view(-1, 28*28)

        images = images.to(device)

        # Find the output by doing a forward pass through the network
        outputs = net(images)

        # Find the class of each sample by taking a max across the probabilities of each class
        _, predicted = torch.max(outputs.data, 1)

        # Increment 'total', and 'correct' according to whether the prediction was correct or not
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))