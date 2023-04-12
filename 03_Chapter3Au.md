# Chapter 3: Automatic Differentiation with PyTorch

Welcome back, noble readers, to the third chapter of our adventure exploring PyTorch Models from A-Z in Effective Python. In the last chapter, we learned about PyTorch Tensors and how they lay a strong foundation for creating sophisticated machine learning models. Now, it's time to take our journey one step further and delve into the world of automatic differentiation.

Automatic differentiation is a fundamental building block of modern deep learning architectures. It allows us to calculate gradients efficiently, which are crucial for model training, evaluation, and optimization. We'll explore this concept and learn about its essential components, such as the computational graph, differentiation, and backpropagation, in detail.

In this chapter, we'll begin by understanding the basics of automatic differentiation using PyTorch. We'll then expand our knowledge by exploring more advanced topics such as custom gradients, higher-order derivatives, and Jacobians. Along the way, we'll use PyTorch to illustrate these concepts and demonstrate how to apply them in real-world scenarios.

So, come forth, adventurers! Join us on this exciting journey and unlock the power of automatic differentiation with PyTorch!
# The Tale of King Arthur and the Knights of the Automatic Differentiation

Once upon a time, in the land of PyTorch, King Arthur and his Knights of the Automatic Differentiation were on a quest to create a powerful machine learning model that could predict the outcomes of battles. They knew that the heart of this model would be the gradients that would allow it to adjust its predictions based on the data it was given.

So, King Arthur ordered his court magician Merlin to create a magical formula that would help them calculate the gradients for their model effortlessly. Merlin knew that the key to this was a concept called automatic differentiation.

Merlin explained that automatic differentiation was simply a way to calculate gradients automatically, without the need for manual differentiation. It was based on the idea of building a computational graph, which would keep track of all the operations performed on the data, and then calculating the gradients by propagating them backward through the graph.

The knights were amazed at this concept and begged Merlin to teach them more. Merlin then showed them how to perform automatic differentiation in PyTorch, beginning with simple operations like addition and multiplication and working their way up to more complex models.

The knights practiced diligently, building models, training them on data, and optimizing them using the power of automatic differentiation. They were amazed at how quickly they could improve their models and how accurate their predictions had become.

But, like all good stories, there was a twist. One day, a villain named Mordred appeared, plotting to steal the knights' powerful model for himself. He launched a surprise attack on King Arthur's castle, catching the knights off guard.

The knights quickly countered, drawing on their knowledge of automatic differentiation. They used their model to predict Mordred's next moves, and they were able to outmaneuver him at every turn. Mordred was eventually defeated, and King Arthur praised his knights for their bravery and smarts.

In the end, the Knights of the Automatic Differentiation emerged victorious, thanks to their mastery of this essential concept. They had learned that automatic differentiation was not merely a tool, but a weapon that could be used to overcome any foe, and they swore to use it for the good of all in the land of PyTorch.

And so, dear readers, we conclude our tale of King Arthur and the Knights of the Automatic Differentiation. Like them, may you too use the power of automatic differentiation to create mighty machine learning models and emerge victorious in all your battles.
# The Code of King Arthur and the Knights of the Automatic Differentiation

In order to battle Mordred, the Knights of the Automatic Differentiation knew that they had to have a powerful machine learning model that could predict his next moves. They used the power of PyTorch's automatic differentiation to create this model.

Here is an example of how they built and trained their model using PyTorch's automatic differentiation:

```python
import torch
import torch.nn as nn

# Define the architecture
model = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
```

In this code, we first define the architecture of our model using PyTorch's `nn.Sequential` module. This model has two linear layers and one ReLU activation function. We then define our loss function, which is binary cross-entropy with logits, and our optimizer, which is the Adam optimizer with a learning rate of 0.1.

We then train our model using a loop that runs for a fixed number of epochs. In each epoch, we zero out the gradients using `optimizer.zero_grad()`, compute the output of our model using `model(inputs)`, calculate the loss using `criterion(output, labels)`, and then use PyTorch's `backward()` method to compute the gradients and update the weights using `optimizer.step()`.

Thanks to PyTorch's automatic differentiation, the Knights of the Automatic Differentiation were able to build and train this powerful model quickly and accurately. They were then able to use it to outsmart and defeat Mordred in battle, emerging victorious thanks to the power of automatic differentiation.