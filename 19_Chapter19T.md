# Chapter 19: Text Classification

Welcome back to our epic journey through PyTorch Models from A-Z in Effective Python! In the last chapter, we explored the fascinating world of Language Modeling with our guest Jürgen Schmidhuber. If you missed that chapter, be sure to catch up before continuing.

Now, we move on to another exciting field in Natural Language Processing: Text Classification. This task involves categorizing a given piece of text into one or more predefined categories based on its content. Text classification plays a vital role in various applications such as sentiment analysis, spam filtering, news classification, and many more.

To guide us through this chapter, we are thrilled to have a special guest: Yann LeCun! Yann LeCun is a renowned computer scientist and a pioneer in the field of deep learning. He is most widely known for his work on Convolutional Neural Networks (CNNs), which revolutionized image recognition tasks. In 2018, he was awarded the Turing Award, which is considered the "Nobel Prize of Computing".

In this chapter, we will explore various techniques for text classification using PyTorch. We will start by discussing the basics of text classification, followed by an introduction to PyTorch's text processing utilities. Then, we will delve into the world of deep learning by exploring various models, such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers, for text classification.

Together with Yann LeCun, we will examine practical examples of text classification, including sentiment analysis using CNNs, fake news detection using Transformers, and multiclass classification using RNNs. Along the way, we will also discuss the latest research trends and the challenges faced in this field.

Are you ready for this thrilling adventure? Let's get started!
# Chapter 19: Text Classification

## The Tale of the Sphinx

In ancient Greek mythology, the Sphinx was a creature with a lion's body and a human's head. It terrorized the city of Thebes by posing a riddle to those who passed by. If they could not solve the riddle, the Sphinx would devour them.

One day, a young Theban named Oedipus came across the Sphinx. The Sphinx posed the riddle: "What creature walks on four legs in the morning, two legs in the afternoon, and three legs in the evening?" Oedipus pondered for a moment and answered, "Man. As an infant, he crawls on hands and knees in the morning of his life. As an adult, he walks on two legs in the afternoon of his life. And in the evening of his life, he uses a cane as a third leg." The Sphinx was so impressed by Oedipus's answer that it threw itself off a cliff and died.

## PyTorch Models for Text Classification

In our modern world, the Sphinx's riddles have taken the form of text classification tasks. Can machines learn to categorize text based on its content? With PyTorch, the answer is yes!

With the help of our special guest Yann LeCun, we learned how to use Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers to tackle various text classification problems. We also discussed the importance of pretraining models on large datasets and the use of attention mechanisms to improve model performance.

Let's see how our PyTorch models fare in solving a modern-day Sphinx riddle:

```
Riddle: Can machines learn to recognize hate speech and differentiate it from regular speech?

Solution: Yes. By training a text classification model on a dataset of labeled hate speech and normal speech examples, we can teach it to distinguish between the two. This can be useful in various applications, such as online moderation and hate speech detection in social media.
```

Just like Oedipus used his knowledge and intelligence to defeat the Sphinx, we can use our PyTorch models to tackle challenging text classification problems.

## Conclusion

In this chapter, we explored the fascinating world of text classification using PyTorch. We learned how to preprocess text data using PyTorch's powerful text processing utilities and built various deep learning models for text classification. Together with our special guest Yann LeCun, we examined applications of text classification, including sentiment analysis, fake news detection, and multiclass classification.

As our journey through PyTorch Models from A-Z in Effective Python comes to an end, we hope that this epic adventure has equipped you with the knowledge and skills to tackle various real-world problems using PyTorch. So go forth, young padawan, and conquer the world with the power of PyTorch!
Certainly! To solve the Sphinx's riddle in our PyTorch text classification model, we need to train a model on a dataset of labeled examples of hate speech and regular speech, and then test it on new examples to see if it can accurately predict if a given piece of text contains hate speech or not.

Here's an example of how you could train and test a simple Convolutional Neural Network (CNN) for binary text classification using PyTorch:
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import text_classification
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define hyperparameters
EMBED_DIM = 64       # Embedding dimension
NUM_FILTERS = 100    # Number of filters
FILTER_SIZES = [3, 4, 5]  # Filter sizes
OUTPUT_DIM = 1       # Output dimension
DROPOUT = 0.5        # Dropout probability

# Load dataset
tokenizer = get_tokenizer("basic_english")
TEXT = torchtext.data.Field(tokenize=tokenizer, batch_first=True, include_lengths=True)
LABEL = torchtext.data.Field(sequential=False, is_target=True)

train_dataset, test_dataset = text_classification.DATASETS["AG_NEWS"](root="./data")

# Build vocabulary
train_iterator = torchtext.datasets.text_classification._csv_iterator(train_dataset)
vocab = build_vocab_from_iterator(train_iterator, specials=["<unk>", "<pad>", "<s>", "</s>"])

# Define model architecture
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embed_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes)*num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Initialize model and optimizer
model = TextCNN(len(vocab), EMBED_DIM, NUM_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
optimizer = optim.Adadelta(model.parameters())

# Define loss function and accuracy metric
criterion = nn.BCEWithLogitsLoss()
accuracy = lambda preds, y: ((preds > 0.5) == y.byte()).sum().item() / len(y)

# Training loop
for epoch in range(10):
    train_loss, train_acc, n_examples = 0, 0, 0
    model.train()
    for batch in train_dataset:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        label = batch.label.float() - 1.0
        predictions = model(text, text_lengths)
        loss = criterion(predictions, label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += accuracy(predictions, label.unsqueeze(1))
        n_examples += len(label)
    train_loss /= n_examples
    train_acc /= n_examples

    # Evaluation loop
    model.eval()
    eval_loss, eval_acc, n_examples = 0, 0, 0
    with torch.no_grad():
        for batch in test_dataset:
            text, text_lengths = batch.text
            label = batch.label.float() - 1.0
            predictions = model(text, text_lengths)
            loss = criterion(predictions, label.unsqueeze(1))
            eval_loss += loss.item()
            eval_acc += accuracy(predictions, label.unsqueeze(1))
            n_examples += len(label)
    eval_loss /= n_examples
    eval_acc /= n_examples

    print("Epoch {} - Training loss: {:.4f} - Training accuracy: {:.4f} - Evaluation loss: {:.4f} - Evaluation accuracy: {:.4f}".format(epoch+1, train_loss, train_acc, eval_loss, eval_acc))
```
In this example, we use the AG_NEWS dataset, which contains news articles with labels indicating their category (sports, world, etc.). We treat the labels "sports" and "world" as regular speech, and the labels "business" and "tech" as hate speech.

Our model is a simple CNN that takes as input a sequence of word embeddings and convolves over it with filters of different sizes. It then applies max-pooling across each filter and concatenates the resulting features before passing them through a fully connected layer and a sigmoid function for binary classification.

During training, we minimize the binary cross-entropy loss using the Adadelta optimizer, and evaluate model performance on a separate test dataset after each epoch. We also apply dropout regularization to improve generalization.

And just like that, we can use PyTorch to solve modern-day Sphinx riddles through the power of text classification!