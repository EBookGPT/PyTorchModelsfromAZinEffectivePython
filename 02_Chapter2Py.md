# Chapter 2: PyTorch Tensors

Welcome back, dear reader. I hope you are ready for the next chapter of our PyTorch journey. In the previous chapter, we gained some intuition about what PyTorch is, its advantages over other Deep Learning frameworks, and how we can use it to our advantage in various applications. In this chapter, we are going to take a closer look at PyTorch Tensors, a fundamental data structure in PyTorch.

Tensors are the building blocks of Deep Learning in PyTorch and are essential for performing numerical computations with PyTorch models. They are similar to NumPy's ndarrays, but unlike ndarrays, they can be run on GPUs, which enables us to perform computations much faster. Furthermore, tensors are used to represent the inputs, outputs, and parameters in PyTorch Models.

In this chapter, we'll learn about creating tensors, operating on tensors, and converting tensors to and from NumPy ndarrays. We'll also explore different manipulation operations we can perform on tensors, such as reshaping and slicing, and how to compute on tensors with mathematical operations.

By the end of this chapter, you will have gained enough knowledge about PyTorch Tensors to start implementing your own PyTorch models. So, let's sharpen our fangs and dive deeper into the world of PyTorch tensors!
# Chapter 2: PyTorch Tensors

## The Dracula Story

Dracula sat quietly in his castle, sipping his tea and pondering over the decisions he had made in his immortal life. He had never been one to rest on his laurels, and he was always exploring new ways to improve himself. One day, he heard of a group of young Data Scientists who were using PyTorch, a new Deep Learning framework. Intrigued, he summoned them to his castle for a demonstration.

As the young Data Scientists arrived, Dracula welcomed them warmly and asked them to explain the basics of PyTorch. "PyTorch Tensors is the fundamental data structure in PyTorch," one of them began, "It's similar to NumPy arrays, but can be run on GPUs for faster computation."

Dracula was fascinated and asked them to show him some examples of how Tensors worked. The Data Scientists handed him a code sample that created a tensor and performed a mathematical operation on it.

```
import torch

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

z = x + y
print(z)
```

Dracula was amazed by the simplicity of the code and the speed of the computation. He asked the Data Scientists to explain how to manipulate Tensors.

One of the Data Scientists explained, "You can slice and reshape Tensors the same way you would with NumPy arrays."

```
import torch

x = torch.tensor([[1, 2], [3, 4]])

# Slice Tensor
y = x[:, 1]
print(y)

# Reshape Tensor
z = x.view(4)
print(z)
```

Dracula was pleased with what he saw. He ordered the Data Scientists to develop a PyTorch model to help him select the best blood donors in his castle. They worked tirelessly, creating Tensors for inputs, outputs, and parameters, and training the model using data from various donors.

## The Resolution

After a long night of training the PyTorch model, the Data Scientists were ready to report their findings to Dracula.

"We have analyzed the blood of all your donors," they said. "Using PyTorch, we have discovered the blood type and the iron levels of each donor, and we have determined which donors are best suited for you."

Dracula was thrilled with the results. He thanked the Data Scientists and praised the power of PyTorch Tensors in finding the best donors for his immortal life.

"You have taught me a valuable lesson," he said, "PyTorch Tensors may be simple to use, but they are incredibly powerful tools that can help improve our lives in so many ways."

And with that, Dracula raised his glass to the future of PyTorch and the power of Tensors in Deep Learning.
Sure, let's take a closer look at the code used to resolve the Dracula story.

In the code sample provided by the Data Scientists, they created a PyTorch model to help Dracula select the best blood donors in his castle. They used Tensors to represent the input and output of the model, as well as the model's parameters. Here's the code:

```
import torch

# Data preparation
donors = torch.tensor([[1, 0.5], [2, 0.7], [3, 0.3], [4, 0.8]], dtype=torch.float32)
labels = torch.tensor([0, 1, 0, 1])

# Define the model
class DonorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        return x
    
model = DonorModel()

# Define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(donors)
    loss = criterion(outputs.squeeze(), labels.float())
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test predictions
with torch.no_grad():
    test_donor = torch.tensor([[5, 0.4]], dtype=torch.float32)
    prediction = model(test_donor)
    print(prediction)
```

The first few lines of the code prepare the data for the model. They create a `donors` Tensor to represent the iron and blood type levels of each donor and a `labels` Tensor to represent whether the donor is a good candidate for Dracula or not.

Next, they define the `DonorModel` class, which is a simple neural network with one linear layer. The `forward` function of the model takes an input `x` and passes it through the linear layer to produce an output.

After defining the model, they define the loss function, `BCEWithLogitsLoss`, and the optimizer, `SGD`, which uses stochastic gradient descent to update the model parameters during training.

The for loop trains the model for 100 epochs, with each iteration consisting of a forward pass, a backward pass, and an optimization step.

Finally, they test the model by using it to make a prediction on a new donor with iron and blood type levels `[5, 0.4]`. They create a `test_donor` Tensor to represent this new donor and pass it to the model's `forward` function. The output of the model is then printed to the console.

Overall, this code demonstrates how to use PyTorch Tensors to create a simple Deep Learning model and apply it to a real-world problem. The power of Tensors, combined with the ease of use of PyTorch, make it an excellent tool for Data Scientists and other practitioners of Machine Learning.