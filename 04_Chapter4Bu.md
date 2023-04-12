# Chapter 4: Building Neural Networks in PyTorch

In this chapter, we will dive into the world of building neural networks using PyTorch. We will build upon the knowledge gained in the previous chapter about automatic differentiation with PyTorch and apply it to constructing neural networks.

We are excited to introduce our special guest, the legendary Geoffrey Hinton. He is a renowned computer scientist and artificial intelligence expert who has made groundbreaking contributions to the field of deep learning. Among his many accomplishments, he is known for his work on backpropagation, the Boltzmann machine, and Convolutional Neural Networks, to name a few.

We will start by discussing the fundamental concepts of neural networks and their importance in machine learning. Then we will explore the key components of constructing a neural network using PyTorch, including tensors, layers, and activation functions. Along the way, Geoffrey Hinton will share his insights on best practices and his experience in the field of deep learning.

Our journey will culminate with a hands-on project where we will build and train a neural network to classify hand-written digits using the famous MNIST dataset. We will demonstrate how to optimize our model's parameters for maximum accuracy and how to avoid common pitfalls.

By the end of this chapter, you will have a solid foundation in building neural networks with PyTorch and feel confident in tackling more complex problems. So, let's get started on this exciting adventure!
# Chapter 4: Building Neural Networks in PyTorch

## The Mystery

Sherlock Holmes was recently approached by a prominent robotics company in London. The company, which manufactured robotic assistants for various industries, was facing a problem: their robotic assistants would occasionally malfunction, resulting in costly damages to their clients' equipment.

After conducting a preliminary investigation, Sherlock determined that the root cause of the malfunction was a faulty neural network algorithm. The network was designed to help the assistants make sense of the data they collected from their various sensors, but it would occasionally produce nonsensical outputs that would cause the assistants to behave erratically.

The company had employed a team of engineers to design the neural network, but after several unsuccessful attempts to fix the issue, they were at a loss. That's when they turned to Sherlock for help.

## The Solution

Sherlock knew that the key to solving the mystery lay in understanding the inner workings of the neural network. He called upon his old friend, Geoffrey Hinton, to assist with the investigation.

Together, Sherlock and Geoffrey carefully examined the architecture of the neural network and identified a flaw in the design of the output layer. The engineers had used an outdated activation function that was known to produce erratic outputs under certain conditions.

Geoffrey recommended a newer activation function that he had recently published in a journal. It was based on his research on Rectified Linear Units (ReLU), which he had shown to outperform other popular activation functions like sigmoid and hyperbolic tangent.

Sherlock and Geoffrey implemented the new activation function in PyTorch and retrained the neural network with the company's data. After a few minutes of training, they observed a marked improvement in the accuracy of the network's outputs.

The real test came when they deployed the new neural network to one of the malfunctioning robotic assistants. The assistant performed flawlessly, collecting data from its sensors and processing it through the neural network without a single error.

The company was ecstatic with the results and thanked Sherlock and Geoffrey for their expertise. They immediately began implementing the new neural network algorithm into all of their robotic assistants and saw a significant reduction in malfunction incidents.

Sherlock and Geoffrey smiled, satisfied with a job well done. They knew that their experience and expertise in building neural networks with PyTorch had saved the day.
# Chapter 4: Building Neural Networks in PyTorch

## The Solution: Technical Details

To solve the mystery of the malfunctioning robotic assistants, Sherlock Holmes and Geoffrey Hinton had to carefully examine the architecture of the flawed neural network and identify the cause of the output layer's erratic behavior. After identifying the issue, they utilized PyTorch to implement Hinton's newly published activation function and retrain the neural network.

To implement the activation function, they started by importing the necessary PyTorch modules:

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

Next, they defined the neural network architecture with the new Rectified Linear Units (ReLU) activation function as the output layer's activation function:

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
```

Here, the `fc1`, `fc2`, and `fc3` represent fully connected layers in the neural network, with 784 input features, 128 hidden features, and 64 hidden features, respectively. The output layer, represented by `fc3`, utilizes the new ReLU activation function to produce the neural network's output.

The training process utilizes the Adam optimizer and cross-entropy loss function:

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
```

To retrain the neural network with the new activation function, they loaded the company's data and trained the network for several minutes:

```python
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

After retraining the neural network, they deployed it to a malfunctioning robotic assistant and observed it process data from its sensors without producing a single error.

Thus, utilizing PyTorch's ease of use and versatile tools, Sherlock and Geoffrey could expertly resolve the mystery of the faulty neural net and saved the day.