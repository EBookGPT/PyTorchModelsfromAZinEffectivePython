# Chapter 13: Using Pretrained Models

Welcome back to the adventures of King Arthur and the Knights of the Round Table in the land of PyTorch! In our previous chapter, we delved into the exciting world of Transfer Learning, where we learned how to leverage existing models for our own use cases.

In this chapter, we take things up a notch and explore the wonders of using Pretrained Models. Pretrained models are complex neural networks that have already been trained on large datasets, and can be fine-tuned for specific tasks. They are often used to tackle more challenging tasks like image recognition, natural language processing, and speech recognition.

The use of pretrained models can help us save time and resources as we can leverage on the existing knowledge obtained by the pretraining process. Using pretrained models also allows us to tackle problems that may be too challenging to solve without prior training.

We will demonstrate how to access pretrained models using PyTorch and fine-tune them for specific tasks. Along the way, we will learn about different strategies for fine-tuning, such as freezing layers and adjusting learning rates, and how they can affect model performance.

So, let's embark on this exciting journey with the Knights of the Round Table and explore the wonders of using pretrained models in PyTorch!
# The Tale of Training a Dragon using Pretrained Models


It was a beautiful day in Camelot, and the Knights of the Round Table were gathered in the great hall, discussing their next great adventure. Suddenly, they heard a loud roar coming from outside the castle walls. The knights rushed to investigate and discovered a ferocious dragon terrorizing the nearby village.

King Arthur knew that he had to act fast to protect his people, and summoned his most skilled knights for a meeting. They decided that they would need to develop an AI model to help them fight the dragon with precision and accuracy.

Merlin, the wizard, suggested they try using a pretrained model that had been trained on hundreds of thousands of images of dragons. The model was already capable of recognizing dragons with high accuracy, which meant that the King’s knights only needed to obtain images of the specific dragon and feed it to the model for identification.

The knights agreed and began preparations to capture images of the dragon. As they ventured out into the forest, they quickly realized that the dragon was quick and elusive. Capturing images of the dragon was proving to be quite a challenge.

Through perseverance, the knights managed to capture a few images and returned to the castle. The team took these images and trained the model using PyTorch, fine-tuning it to identify the specific features of the dragon they were facing. They experimented with different fine-tuning strategies, such as adjusting learning rates, increasing the number of epochs, and freezing layers, until finally, they achieved the desired level of accuracy.

Using the fine-tuned model, they were able to track the dragon and predict its next move, giving the King’s knights an unprecedented advantage. In a fierce and intense battle, the knights worked together and managed to slay the dragon, bringing peace back to the land of Camelot.

As they retreated back to Camelot, the King’s knights discussed the success of their mission and marveled at the power of pretrained models. They knew that their adventure had paved the way for more effective use of deep learning in the fight against potential disasters in the future.

And thus, the tale of training a dragon using pretrained models was forever etched in the annals of Camelot’s history, inspiring future generations to always strive for innovative solutions to complex problems.
## Explanation of the PyTorch Code

To bring our tale of training a dragon using pretrained models to life, we utilized the PyTorch library, an open-source machine learning library. Specifically, we leveraged PyTorch's vision library, which provides various state-of-the-art pretrained models for image classification.

We first imported the necessary libraries:

```python
import torch
import torchvision
import torchvision.transforms as transforms
```

Next, we loaded the pretrained model:

```python
model = torchvision.models.resnet18(pretrained=True)
```

This loaded the ResNet-18 model, which had been pretrained on the ImageNet dataset, a dataset of over a million images. We added a final layer to the model to adjust its output to match the number of classes we would need to classify. In our story, we needed to identify the presence of the dragon in the image, so our final layer would only have two output classes:

```python
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
```

We also modified the loss and optimizer functions for our specific use case:

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

We then trained the model by iterating over the training set, adjusting the weights of the model according to the loss and optimizing function:

```python
for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

Finally, we tested the model on a set of validation images to determine its accuracy:

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
```

That's all for our PyTorch code in our story! By utilizing pretrained models and fine-tuning our model, we were able to successfully train an AI model to identify the presence of the dragon in the images captured by the King’s knights.