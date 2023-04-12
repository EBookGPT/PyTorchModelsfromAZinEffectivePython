# Chapter 21: Sequence-to-Sequence (Seq2Seq) Modeling

Welcome to the 21st chapter of our journey through PyTorch Models from A-Z in Effective Python. In the previous chapter, we explored Sentiment Analysis using PyTorch models. Continuing on with our learning journey, in this chapter, we will delve into Sequence-to-Sequence (Seq2Seq) Modeling.

Human communication involves a host of symbolic systems, including language, visual media, and haptic feedback. Of these, language is perhaps the most fundamental. Seq2Seq models are fundamental to many natural language processing (NLP) tasks such as machine translation, summarization, chatbots, image captioning, and speech recognition.

Seq2Seq models are a type of neural network architecture that is designed to map input sequences onto output sequences. They consist of two primary components: the encoder and the decoder. The encoder is used to encode the input sequence that needs to be translated. The decoder, on the other hand, generates the output sequence.

In this chapter, we will explore various types of Seq2Seq models and their applications. We will learn how to build Seq2Seq models using PyTorch and how to train and evaluate them. We will also explore best practices and recent advancements in Seq2Seq modeling.

Let us embark on our journey to develop our understanding of Seq2Seq modeling in PyTorch!
# Chapter 21: Sequence-to-Sequence (Seq2Seq) Modeling

## The Frankenstein Story:

As the village folk went about their daily chores, they noticed something strange happening in the dark corner of the village. A few brave souls decided to investigate and found a strange creature lurking in the shadows. The creature, with its gnarled face and hunched back, was trying to communicate with them, but they could not understand its language.

The village leader decided to seek the help of the greatest scientist in the land, Dr. Frankenstein, to create a machine that could translate the creature's language into theirs. Dr. Frankenstein, being the expert he was, set out to create a Sequence-to-Sequence (Seq2Seq) model that would allow the creature to communicate and integrate into the village.

Dr. Frankenstein began by building an encoder, which would take in the creature's language and convert it into a fixed-length vector, a representation of the input sequence. He trained the encoder using the creature's language and was satisfied with the results. However, he realized that using only the encoder would not be enough, as the decoder was also critical for the Seq2Seq model to translate the creature's language.

Dr. Frankenstein then set out to build the decoder, which would take the fixed-length vector produced by the encoder and output a sequence in the village folk's language. After careful training of the decoder, Dr. Frankenstein was satisfied with the Seq2Seq model's performance and eagerly awaited the test results.

To his delight, the Seq2Seq model performed exceptionally, accurately translating the creature's language into the village folk's language. The creature was now able to communicate and integrate into the village, filling it with happiness and joy.

## The Resolution:

As we can see from this tale, Seq2Seq models can have a significant impact, enabling communication even between different languages. Sequences are fundamental to many applications, including language processing, image captioning, and summarization. In this chapter, we explored various Seq2Seq models, including vanilla Seq2Seq models and attention-based Seq2Seq models. We explored how to build, train, and evaluate Seq2Seq models using PyTorch in a range of applications.

As we conclude our journey through PyTorch, we hope that you have found this book informative and enriching. We cannot wait to see how you will use the concepts you have learned here to create powerful models that will transform the world.
# Chapter 21: Sequence-to-Sequence (Seq2Seq) Modeling

## The Code behind the Resolution:

To build our Seq2Seq model, we will leverage the power of PyTorch. First, we need to build the architecture of the model. This code snippet demonstrates how to create an encoder layer that takes in an input sequence and returns a hidden state:

```python
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        return outputs, hidden
```

Here, we define an `Encoder` class that takes in `input_dim`, `emb_dim`, `enc_hid_dim`, `dec_hid_dim`, and `dropout` as inputs. `input_dim` represents the input sequence's vocabulary size,`emb_dim` represents the embedding dimension, `enc_hid_dim` represents the hidden dimension size in the encoder, `dec_hid_dim` represents the hidden dimension size in the decoder, and `dropout`is the dropout probability for regularization.

The `forward` method takes in the source (`src`) and passes it through an embedding layer to get the embedded representation. The embedded vector then goes through a bidirectional GRU layer to get two sets of `enc_hid_dim` hidden state vectors, one from the forward RNN and the other from the backward RNN. We concatenate these two hidden states and pass them through a linear layer (`fc`) and apply the tanh activation function. This final `hidden` state will be fed as input to the decoder.

Next, we define the decoder network, as shown in the following code snippet:

```python
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim = 2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        return prediction, hidden.squeeze(0), a.squeeze(1)
```

Here, we define a `Decoder` class that takes in `output_dim`, `emb_dim`, `enc_hid_dim`, `dec_hid_dim`, `dropout`, and `attention` as inputs. `output_dim` is the target vocabulary size, `emb_dim` is the embedding dimension, `enc_hid_dim` is the hidden dimension size in the encoder RNN, `dec_hid_dim`is the hidden dimension size in the decoder RNN, `dropout`is the dropout probability for regularization, and `attention` is the type of attention mechanism used to weigh the importance of the encoder outputs.

The `forward` method takes `input`, `hidden`, and `encoder_outputs`as input. `input`is the previous target token, `hidden`is the previous decoder hidden state, and `encoder_outputs`are the outputs from the encoder RNN.

First, we pass the `input` through an embedding layer and apply a dropout. We then use the attention mechanism (defined by the `attention` parameter) to weigh the importance of the encoder output vectors. We concatenate the weighted vector with the embedded input to get the input to the decoder RNN. We then pass this input, the previous hidden state (`hidden`), through an RNN layer to get the new hidden state (`output`). Finally, we pass this new hidden state through a linear layer (`fc_out`) to get the predicted target token.

In conclusion, the PyTorch framework offers many building blocks to create custom Seq2Seq models. The provided code snippets are part of the Encoder and Decoder classes, which can be combined to create a full Seq2Seq model that can translate between different languages, among other possible applications.