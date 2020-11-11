import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
# Convert it to a Python list and paste it here
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
             "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
             "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each",
             "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll",
             "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
             "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
             "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
             "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
             "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
             "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's",
             "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
             "would", "you",
             "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

# Imports transfer learning weights and downloads it to a folder
transferurl = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv'
path = 'C:/Users/jes17/OneDrive/Documents/'
resptrain = requests.get(transferurl)
zname = os.path.join('/Users/jes17/OneDrive/Documents/datasets/NLP/', "bbc-text.csv")
zfile = open(zname, 'wb')
zfile.write(resptrain.content)
zfile.close()

sentences = []
labels = []
with open("/Users/jes17/OneDrive/Documents/datasets/NLP/bbc-text.csv", 'r') as csvfile:  # Opens the CSV File
    csv_reader = csv.reader(csvfile, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
        labels.append(row[0])  # Extracts the First Column of the datasets per each row
        sentence = row[1]  # Extracts the second column of the dataset per each row
        for word in stopwords:
            token = " " + word + " "  # extract the word in stop words and adds spaces
            sentence = sentence.replace(token, " ")  # Replaces stop word in the sentence wiht " "
            sentence = sentence.replace("   ", " ")
        sentences.append(sentence)  # appends sentences in the sentence list/dataset

print(len(sentences))
print(sentences[0])
print(len(labels))
print(labels[0])

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_portion = .8

train_size = int(len(sentences) * training_portion)

train_sentences = sentences[:train_size]
train_labels = labels[:train_size]

validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)  # Your Code Here

### Tokenizer Definitions
# This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf...

### Arguments
# num_words - 	the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
# filters - a string where each element is a character that will be filtered from the texts. The default is all punctuation, plus tabs and line breaks, minus the ' character.
# lower - boolean. Whether to convert the texts to lowercase.
# split - str. Separator for word splitting.
# char_level - if True, every character will be treated as a token.
# oov_token - if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls

tokenizer.fit_on_texts(train_sentences)
### fit_on_text () definition Updates internal vocabulary based on a list of texts. In the case where texts contains lists, we assume each entry of the lists to be a token.
# texts - can be a list of strings, a generator of strings (for memory-efficiency), or a list of list of strings.

word_index = tokenizer.word_index  # Your Code here
print(len(word_index))

### word_index Arguments -- Definitions -

train_sequences = tokenizer.texts_to_sequences(train_sentences)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)  # Your Code Here
### Arguments texts_to_sequences() Definition - # Transforms each text in texts to a sequence of integers. # Only top num_words-1 most frequent words will be taken into account. Only words known by the tokenizer will be taken into account.
# texts - A list of texts (strings).

train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)  # Your Code here
### Aruguments --- Definition - pads sequences of numbers per sentence to the same length
# sequences1 - List of sequences (each sequence is a list of integers).
# maxlen - Optional Int, maximum length of all sequences. If not provided, sequences will be padded to the length of the longest individual sequence.
# dtype - (Optional, defaults to int32). Type of the output sequences. To pad sequences with variable length strings, you can use object.
# padding - String, 'pre' or 'post' (optional, defaults to 'pre'): pad either before or after each sequence.
# truncating - String, 'pre' or 'post' (optional, defaults to 'pre'): remove values from sequences larger than maxlen, either at the beginning or at the end of the sequences.
# value - Float or String, padding value. (Optional, defaults to 0.)
print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(train_padded, training_label_seq, epochs=num_epochs,
                    validation_data=(validation_padded, validation_label_seq), verbose=1)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim)
