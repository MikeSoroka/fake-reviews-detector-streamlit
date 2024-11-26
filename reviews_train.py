import tensorflow as tf
import numpy as np
import pandas as pd
import random
from nltk.tokenize import sent_tokenize
import nltk

# nltk.download('punkt_tab')

def perplexity(y_true, y_pred):
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.exp(tf.reduce_mean(cross_entropy))

class TextGenerationModel:
    def __init__(self, vocab_size=5000, max_sequence_percentile=75, embedding_dim=256, lstm_units=(256, 128),
                 tokenizer=None, max_sequence_len=None, model=None):
        self.vocab_size = vocab_size
        self.max_sequence_percentile = max_sequence_percentile
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = None
        self.max_sequence_len = None
        self.model = None

    def preprocess_data(self, reviews, num_sentences=50000):
        text = []
        for rev in reviews:
            sentences = sent_tokenize(rev)
            sentences = [sentence for sentence in sentences if len(sentence.split()) > 3]
            text.extend(sentences)

        training_data = random.sample(text, num_sentences)

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)
        self.tokenizer.fit_on_texts(training_data)
        tokenizer_json = self.tokenizer.to_json()
        with open("reviews_tokenizer_tf.json", "w") as file:
            file.write(tokenizer_json)

        input_sequences = []
        for single_line in training_data:
            token_list = self.tokenizer.texts_to_sequences([single_line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        self.max_sequence_len = np.int32(np.percentile([len(x) for x in input_sequences], self.max_sequence_percentile))
        input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=self.max_sequence_len,
                                                                        padding='pre')

        xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
        return tf.data.Dataset.from_tensor_slices((xs, labels)).batch(64).prefetch(tf.data.AUTOTUNE)

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(self.max_sequence_len - 1,))
        x = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_len - 1,
                                      mask_zero=True)(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units[0], return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units[1]))(x)
        outputs = tf.keras.layers.Dense(self.vocab_size, activation="softmax", name="output_layer")(x)

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0005, decay_steps=10000, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=optimizer,
            metrics=['accuracy', perplexity]
        )

    def train(self, dataset, epochs=30):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="review_gen_model.h5",
            monitor='accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
        # Train the model
        self.model.fit(dataset, epochs=epochs, callbacks=[checkpoint, early_stopping])

    def additional_train(self, dataset, epochs=5, model_path="review_gen_model_copy.h5"):
        self.model = tf.keras.models.load_model(model_path, custom_objects={"perplexity": perplexity})
        self.model.fit(dataset, epochs=epochs)
        self.model.save(model_path)

    def generate_text(self, seed_text, num_words=50):
        if not self.tokenizer or not self.model:
            raise ValueError("Model and tokenizer must be initialized before text generation.")

        for _ in range(num_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=self.max_sequence_len - 1,
                                                                       padding='pre')
            predicted = np.argmax(self.model.predict(token_list), axis=-1)
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text


PATH = "D:/work/Data/datasets/Reviews.csv"
data = pd.read_csv(PATH)
reviews = data['Text']

text_gen_model = TextGenerationModel()
dataset = text_gen_model.preprocess_data(reviews)
text_gen_model.build_model()
text_gen_model.train(dataset, epochs=10)

seed = input("Enter a word or sentence: ")
print("Generated Text:")
print(text_gen_model.generate_text(seed, num_words=20))