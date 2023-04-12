# Chapter 5: Convolutional Neural Networks (CNNs)

Greetings, fellow learners of PyTorch! Welcome to the chapter where we will explore Convolutional Neural Networks (CNNs). Building on our previous chapter on Building Neural Networks in PyTorch, we will delve deeper into the world of computer vision by understanding how CNNs help us in solving image recognition problems.

CNNs have revolutionized the way we process visual information. These neural networks are closely modeled based on the functioning of our visual system. The translation-invariance feature of CNNs makes them ideal for local feature extraction from images. These features are then used to classify and recognize objects in images. But how does all of this actually work? Let's find out.

We are honored to have Yann LeCun as our special guest in this chapter. Yann LeCun is a computer scientist and is globally acknowledged for his contributions to deep learning and neural networks. He is the inventor of the convolutional neural network and recipient of multiple awards for his contributions to machine learning. Without further ado, let's begin our journey of building a CNN in PyTorch!

To get started on this chapter, I suggest you have a basic understanding of PyTorch and Python. However, if you are new to this, I have got you covered! You can go back to the previous chapters and begin with the basics.

To make the most out of this chapter, I advise you to have a PyTorch environment set up and good knowledge of the following concepts:
- Feedforward neural networks
- Convolutional layers
- Pooling layers
- Activation functions
- Backpropagation
- Gradient descent

Are you ready to learn how to implement CNNs in PyTorch? Let's get straight into the action!
# The Robin Hood's Challenge: Convolutional Neural Networks (CNNs)

Once upon a time, in the kingdom of PyTorch, there was a great archer named Robin Hood. Robin was famous for his sharpshooting skills which helped him win many challenges in the kingdom. One day, Robin received a letter from the king which stated that he was required to participate in a challenge to prove his worth in PyTorch. The challenge was to build a model that can classify and detect various objects in images.

Robin was puzzled and thought about the best strategy to tackle the challenge. In his confusion, he approached a wise monk who was well-versed in the art of neural networks. The monk told Robin about the power of Convolutional Neural Networks (CNNs) in image recognition and how they are constantly redefining the modern world of machine learning.

The monk explained that a CNN was a type of neural network that uses Convolutional layers and Pooling layers to extract features from the input image. These features are then used to identify the different elements in the image, thus helping in the classification and detection of objects. The monk also told Robin about Yann LeCun, who was the inventor of CNNs and a master in this field.

Inspired by the monk's wise words, Robin began his journey to learn about CNNs. He learned about each component of the network and how they worked together to extract features. With the monk's guidance, Robin implemented a CNN in PyTorch and trained it on a dataset of images to identify different objects.

After several days of training, Robin was ready to put his model to the test in front of the king. The challenge began, and Robin's model flawlessly classified and detected different objects in the images presented to it. The king was amazed by the performance of Robin's model, and Robin emerged as the winner of the challenge.

In the end, Robin realized the power of CNNs and how they can be used in solving complex problems in the world of machine learning. He was grateful to Yann LeCun for his contributions and thanked the wise monk for his guidance. And so, Robin continued his journey in PyTorch, building models and contributing to the community along the way.

The end.

---

In conclusion, Convolutional Neural Networks (CNNs) are an essential part of image recognition and computer vision. In this chapter, we delved deeper into the structure of CNNs and implemented them in PyTorch to classify and detect various objects in images.

We also had the privilege of having Yann LeCun as our special guest, who shared his knowledge and insights on CNNs. By implementing CNNs, we were able to solve Robin Hood's challenge and emerge victorious.

I hope this chapter has inspired you to explore the world of computer vision and continue to learn and grow with PyTorch. Keep building and innovating with CNNs!
# The Code Behind Robin Hood's Challenge: Convolutional Neural Networks (CNNs)

In our Robin Hood story, we saw how CNNs helped Robin Hood win a challenge by classifying and detecting different objects in images. In this section, we will take a closer look at the code used to implement the CNN in PyTorch.

## Dataset

Firstly, we need a dataset to train our model on. In this case, we can use the CIFAR-10 dataset which is a collection of 60,000 images, consisting of 10 different classes. We need to download and load the dataset using the torchvision library in PyTorch.

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

## CNN Architecture

We can define our CNN architecture in PyTorch using nn.Module. We define the structure of our model in the constructor and define the forward pass in the forward method.

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

In our CNN, we have two convolutional layers followed by two fully connected layers. We use the ReLU activation function after each layer and a max-pooling layer after every convolutional layer.

## Training the Model

We can now train our model on the CIFAR-10 dataset using the Stochastic Gradient Descent (SGD) optimizer and Cross-Entropy Loss. We can also set up a loop to iterate over our data and update the parameters of the model.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # 2 epochs

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

## Testing the Model

Finally, we can test our trained model by running it on the test dataset and measuring its accuracy.

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

By running this code, we will get the accuracy of our trained model and see how it performs in classifying and detecting objects in the images.