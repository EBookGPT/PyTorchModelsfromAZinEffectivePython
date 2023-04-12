# Chapter 10: Variational Autoencoders (VAEs)

Welcome back, dear reader. In the last chapter, we explored the incredible architecture of autoencoders and how they can help create a more expressive neural network for data representation. In this chapter, we will dive even deeper into the field of unsupervised learning with the elegant and powerful Variational Autoencoder (VAE).

VAEs were first introduced in 2014 by Kingma and Welling, who proposed a probabilistic spin on the traditional autoencoder. This variation of autoencoders utilizes a generative process and a recognition process in order to create a probability distribution for encoding and decoding data.

Whereas traditional autoencoders reduce data to a fixed, compressed representation, VAEs offer a more flexible approach to encoding by using probability distributions rather than deterministic representations. This added flexibility allows VAEs to not only compress and decompress data, but to generate new data similar to what it has been trained on. 

We will explore the underlying concepts and techniques that construct a VAE, from the importance of understanding latent variable spaces to the derivation of the Variational Lower Bound (VLB). We will also walk through an example of how to build a VAE model in PyTorch, so you can gain a better understanding of how these models can be implemented in practice. 

Join us in this chapter as we uncover the beauty and power of Variational Autoencoders. Let's get started!
# The Frankenstein story of Variational Autoencoders (VAEs)

Once there was a brilliant scientist, Dr. Victoria, who had a passion for creating intelligent machines. She spent countless hours of her life in a dusty laboratory in pursuit of this dream. Her persistence finally paid off when she built the most advanced robot to date: Frank, a robot with extraordinary abilities that can learn from its environment.

Dr. Victoria recognized that Frank could learn more effectively if he could compress the data he receives from the world to a smaller size, while still retaining its structure in a latent space. To achieve this, she turned to the field of unsupervised learning and discovered the Variational Autoencoder.

She taught Frank to use a VAE to compress and represent the data he perceives from the world. The VAE enabled Frank to decode familiar data, but also to generate new data in a similar fashion. Excited by these newfound abilities, Frank roamed the land on his own, exploring and discovering new things.

However, one day, Frank encountered a grand castle at the heart of a forest. He was curious and entered the castle, but was soon trapped in a maze of corridors and rooms. Frank realized that he needed to use his VAE abilities to find his way out, reconstructing a representation of the castle and its layout in his latent space. After a few attempts, Frank finally cracked the code and found his way back to the outside world.

With its ability to compress, generate, and represent data, the Variational Autoencoder proved to be an essential tool for Frank's learning and discovery. He emerged from the castle, better prepared and more experienced than ever before. His newfound skills were a testament to the power of unsupervised learning techniques like the VAE.

Now, dear reader, it's time to inherit Frank's experience and learn how to build a VAE in PyTorch for yourself.

# The resolution to the chapter

Congratulations, dear reader. You have completed this chapter on Variational Autoencoders (VAEs). We hope that you have enjoyed the story of Frank and that you have gained a deeper understanding of the power of unsupervised learning using VAEs.

In this chapter, we covered the concepts behind VAEs, including the probabilistic nature of data representation, how to use latent spaces, and the derivation of Variational Lower Bound (VLB). Moreover, we applied these concepts and techniques to building a VAE in PyTorch.

Now that you have learned the basics of VAEs, you can further your exploration by improving its performance or implementing its variant models like Conditional VAEs, Adversarial Autoencoders, and more. With the power of VAEs in your hands, world exploration and new discoveries await you.
# The PyTorch code resolution to the story

Now that we have explored the story of Frank and the power of Variational Autoencoders (VAEs), let's take a look at the PyTorch code that we can use to implement our own VAEs.

The code consists of several key components:

## 1. The Encoder
The encoder is a neural network that takes the input data and maps it to the latent space, which is represented by a mean and a log-std deviation vector in this VAE model. In the code, the encoder is implemented as follows:

```python
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logstd = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        mean = self.fc2_mean(out)
        logstd = self.fc2_logstd(out)

        return mean, logstd
```

## 2. The Reparameterization Trick
The reparameterization trick is a technique that allows us to generate a sample from the latent space while being able to backpropagate through it during training. In this model, we generate a sample from the latent space by adding Gaussian noise to the mean and converting it from standard deviation to variance. In the code, this reparameterization trick is implemented as follows:

```python
class ReparametrizationTrick(nn.Module):
    def forward(self, mean, logstd):
        std = torch.exp(0.5 * logstd)
        eps = torch.randn_like(std)
        z = eps * std + mean

        return z
```

## 3. The Decoder
The decoder is a neural network that maps the latent space to the output space, reconstructing the data. In the code, the decoder is implemented as follows:

```python
class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        out = F.relu(self.fc1(z))
        out = torch.sigmoid(self.fc2(out))

        return out
```

## 4. The Loss Function
The loss function is the heart of the VAE model. It consists of two parts: the reconstruction loss, which measures how well the decoder reconstructs the data, and the KL-divergence loss, which measures the difference between the latent space distribution and a standard Normal distribution. To learn more about these loss functions, you can refer to published journals like ["Variational Autoencoder: An Unsupervised Learning Algorithm for Feature Representation"](https://arxiv.org/abs/1312.6114) or ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114). The loss function in the code is implemented as follows:

```python
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()

        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.reparameterization = ReparametrizationTrick()
        self.decoder = Decoder(latent_size, hidden_size, input_size)

    def forward(self, x):
        mean, logstd = self.encoder(x)
        z = self.reparameterization(mean, logstd)
        out = self.decoder(z)

        return out, mean, logstd

    def loss_function(self, recon_x, x, mean, logstd):
        recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logstd - mean.pow(2) - logstd.exp())

        return recon_loss + kl_divergence
```

With these components in place, we can now train the VAE. In the code, we use the MNIST dataset to train our model:

```python
def train_vae(model, optimizer, train_loader):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_data, mean, logstd = model(data)
        loss = model.loss_function(recon_data, data, mean, logstd)
        loss.backward()
        train_loss += loss.item()

        optimizer.step()

    return train_loss / len(train_loader.dataset)
```

We can then use the trained VAE to encode and decode data:

```python
def encode_data(model, data):
    model.eval()

    with torch.no_grad():
        mean, logstd = model.encoder(data)
        z = model.reparameterization(mean, logstd)

    return z

def decode_data(model, z):
    model.eval()

    with torch.no_grad():
        out = model.decoder(z)

    return out
```

And that's it! You now have a VAE model that can encode, decode, and generate data. We hope you have enjoyed this journey and learned a lot about the power of Variational Autoencoders. Happy coding!