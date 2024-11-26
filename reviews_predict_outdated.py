from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tensorflow as tf
import numpy as np
import json


def perplexity(y_true, y_pred):
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.exp(tf.reduce_mean(cross_entropy))

class TextGenerationTester:
    def __init__(self, model_path, tokenizer_path, max_sequence_len):
        self.model = tf.keras.models.load_model(model_path, custom_objects={"perplexity": perplexity})
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.max_sequence_len = max_sequence_len

    @staticmethod
    def load_tokenizer(tokenizer_path):
        with open(tokenizer_path, "r") as file:
            tokenizer_json = file.read()
        return tokenizer_from_json(tokenizer_json)

    def preprocess_test_data(self, test_sentences):
        input_sequences = []
        for single_line in test_sentences:
            token_list = self.tokenizer.texts_to_sequences([single_line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            input_sequences, maxlen=self.max_sequence_len, padding="pre"
        )
        xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
        return tf.data.Dataset.from_tensor_slices((xs, labels)).batch(64)

    def evaluate(self, test_data):
        results = self.model.evaluate(test_data, return_dict=True)
        print(f"Test Results - Loss: {results['loss']:.4f}, Accuracy: {results['accuracy']:.4f}, Perplexity: {results['perplexity']:.4f}")

    def generate_text(self, seed_text='', num_words=50):
        if seed_text == '':
            seed_text = input('Enter a word or a sentence: ')
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


MAX_SEQUENCE_LEN = 18
MODEL_PATH = "review_gen_model.h5"
TOKENIZER_PATH = "reviews_tokenizer_tf.json"

tester = TextGenerationTester(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH, max_sequence_len=MAX_SEQUENCE_LEN)

test_sentences = ["Cats should dominate this world", "The product was so bad I decided to change it later"]
test_dataset = tester.preprocess_test_data(test_sentences)
tester.evaluate(test_dataset)

print(tester.generate_text())
