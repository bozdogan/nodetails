_2021-05-24_

NoDetails: Essence of the Text
==============================

**Keywords:** text summarization, machine learning, neural networks, RNN.

## Abstract

Reading some text, especially a long text such as a research paper, requires certain concentration and time. While reading this text or article, people spend time and effort to read, understand and interpret the content. They lose too much time on their work because their attention span is not enough to cover the entire text.

We want to create a tool that helps you with the abundance of text information in our age by creating an easy-to-use tool, leveraging the learning capabilities of the machine to extract the defining features from a body of text. We provide a two-step approach to text summarization. First, we create an extended summary containing a subset of the original text, and then we feed that into our abstractive summarization model to get a concise summary. These two are served together to the user so they can read a more detailed version as they need.

## Problem Statement

Reading a text, especially a research document, generally requires a certain level of concentration and time. People lose a lot of time in their studies because they place an importance on comprehension and interpretation while reading these texts or articles. Time losses should be prevented because it is critical to act quickly in education and business life in today's condition. Our main goal is to develop the NoDetails Project, which is relevant to text summarization, to solve that problem. End-users will be able to shorten their research processes as a result of this project by using the developed application. Accordingly, their speed of reaching the information will increase significantly and they will have the opportunity to devote more time to the business development phases.

In this project, an end-to-end method was developed for creating concise summaries that can capture the attention of readers and convey a significant amount of relevant information. For text summarization implementations, there are two main approaches: abstractive summarization and extractive summarization. Using both of these approaches, the method was improved distinctively from other current applications.
This method covers a machine learning model which is a Recurrent Neural Network (RNN). The RNN model is mainly based on LSTM cells and their ability to understand sequence data. This RNN model has been trained and tested using the dataset that we previously pre-processed. Consequently, a text summarization application was created that is independent of the source text, can generate new sentences, and maintains the meaning of integrity.

## Methodology
 
We started off by designing a deep learning model to acquire the way people summarize a bulk of text. The model needs to treat the text as concepts and come up with its own representation, and this new representation needs to be significantly shorter than the original text without losing much detail. In a sense, we need to get rid of all the noise about the matter and retrive the essence of the text.

We have done research on the subject and ended up using an Encoder/Decoder architecture with an RNN network, which will give a summary provided a text input. All the input sequences, during training or later in inference, need to be a fixed size for this architecture to work. So, we need to structure our data accordingly.
After we do basic preprocessing (convert to lowercase, remove punctuation and stopwords, and expand contractions) we convert our data from list of strings to integer sequences: We assign each word a unique number and convert words to sequence of numbers. We reserve "0" to be used as padding, as all sequences must have the same length for the model. We define a maximum length for inputs, trim/discard longer texts, and apply post-padding to shorter texts to level them all.

We used "Amazon Fine Food Reviews" as our first data set to test the model. Then, we encountered the first problem. Custom layers in our model were causing problems with loading saved models from disk. We solved this by telling Keras, our choice of neural network framework, corresponding class structure explicitly on loading process. 
When trained with the whole dataset (380k text-summary pairs), the model can reasonably summarize input texts, so long as they are related to foods. 

After we proved the concept,  we went on and trained the model with a larger data set. This time with “WikiHow Articles”. We have mixed results of that. It sometimes comes up with a better summary than the original but sometimes completely misunderstands the context. We think that is because of the lack of training corpora.

Extractive summarization is done by sentence scoring and choosing the best sentences based on their scores. 
Detailed Explanation of Algorithms Used

### Abstractive Summary
Essentially, we deal with input and output sequences of different lengths. Because of that, we chose our abstractive summarization model to use Encoder-Decoder architecture, which is common in Natural Language Processing with neural networks. Our abstractive summarization model is an RNN model built on top of Encoder-Decoder architecture. We use three stacked LSTM layers for the Encoder part and an LSTM layer with Attention for the Decoder part of our model. 


Figure 4.1: Encoder-Decoder Model - Training Phase

In the training phase the text body is given as the input to the Encoder, and the summary is given input to the Decoder. The model then tries to form a connection between its inputs. [Figure 4.1] Later on in the inference phase, decoder input is not given and Decoder only makes predictions based on trained Encoder state vectors.

Initially, the inputs are sequences of numbers which correspond to the index of the actual word in a vocabulary dictionary. This initial transformation is made to abstract the meaning of the words from their written forms so the model can make sense to them better. (See nodetails.prep module for implementation. Number forms of the words are referred to as tokens from now on.) These input sequences are fed to Embedding layers. Embedding layers reduce the dimensionality of inputs by transforming tokens from target vocabulary space (typically over 150000 unique words) to latent dimension of  LSTM cells (defaulted to 500 in our model). It is easier for the model to find relationships between tokens because they are not individual data points but vectors that can represent in-between states. The weights are adjusted during training to make meaningful connections.

LSTM cells (short for Long Short-Term Memory) are the RNN units in our model. Their main difference to other RNN forms is they keep an internal cell state which creates an effect of a long-term memory. In Encoder three of them are stacked together to increase this effect.

The LSTM cell in Decoder is initialized by the state vectors of the final LSTM cell in Encoder. Then it is trained to give the target summary given Encoder states and the same summary as its target as inputs. In the inference phase, only the Encoder states will be given and Decoder will produce new tokens based on these states and previous input.

Finally, an attention mechanism is used to further increase the performance of the model. Attention layer makes connections between tokens in the input and target sequences to find out tokens that are used in the same role in both. As any of the previous layers, Attention layers too adjusts its internal weights during training to better fit the data.

### Extractive Summary
Our extractive model is a supplementary algorithm to reduce noise for our abstractive model, as well as providing the user an “extended summary”. Algorithm sets scores for every sentence of the text and selects ones with the best scores among them.

We use this algorithm for longer texts to extract the important parts first. That way, the user can see the reference of the actual sentence (a link that highlights the sentence in the original text) while having a shorter headline generated by our abstractive model.

## Preliminary Results

Note that preprocessing is applied to input texts.

```
Review #38: parent rules stepchildren stepparent extra well cash they personal bank tension relationship trying make withdrawal expect stepparent allow simply biological parent upset stepparent step siblings okay excuse outright rude remember struggle still deserve treated basic human respect seem like best form bonds trying hard time this necessarily true biological parent something exclude stepfamily every single event good time biological family expect stepparent different behaviors reactions reactions biological parent know stepmom might cool something biological allows make assumptions stepparent behave react
Original summary: how to deal with step parents and step siblings
Predicted summary: how to deal with your parents shouting at you
```

```
Review #59: home please avoid damage thing person better ball control ball control essential dribbling agility shuttle runs weekly basis time every time make effort beat timing every attempt also could learn dribbling moves viewing tutorial videos youtube need able cross pass shoot feet many world greatest wingers footed dangerous able shoot drift wide cross work using weak foot control pass shoot ball feel awkward first eventually used feel natural space frequently place regularly right moments wing position cross dribble shoot pass
Original summary: how to be good player in soccer
Predicted summary: how to do the maradona
```

```
Review #277: copy considered plagiarism want write something based poem really like give credit cite simply write based poem scan poem grammatical spelling errors check words stop poem flowing smoothly either remove replace weak basic words replace create better imagery determine whether need sensory words remove read loud would like lastly would like family member friend read review poem well catch something sure specifically senses needed sure check punctuation well though punctuation varies poet poet review punctuation added removed impedes flow poem
Original summary: how to the in poetry
Predicted summary: how to write an effective screenplay
```

```
Review #570: once hamster comfortable friend hold hamster correct position make sure underneath bright light able nails easily this probably work best friend face this hamsters paws pointing directly need talk trimming make sure keeps voices soft relaxed scare hamster shout talk loudly squirms give time relax attempting trim nails once still carefully place nail blades snip repeat nail again sure cutting quick once trimmed nails friend either hold loosely relax place back cage give hamster treats know good forget thank helper
Original summary: how to trim hamster nails
Predicted summary: how to care for newborn hamsters
```
