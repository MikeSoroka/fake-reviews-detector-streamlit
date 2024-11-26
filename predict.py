import torch
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from torch import nn
from torch.nn.quantized.functional import threshold
from torchtext.vocab import build_vocab_from_iterator

def predict_export(text, category, rating):
    # Ensure nltk resources are available
    # nltk.download('punkt')
    # nltk.download('stopwords')

    # Constants
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_DIRECTORY = "deployment_models/best_model_untuned.pt"
    MAX_LENGTH = 100  # Max length for padding text


    # Preprocess text data (tokenization, removing stopwords)
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_tokens)


    # Sample data for prediction (replace with your actual data)
    data = pd.DataFrame({
        'text_': ["This product is great!", "I hate this item."],  # Example reviews
        'category': ["electronics", "electronics"],  # Example categories
        'rating': [5, 1]  # Example ratings
    })

    # Example category mapping (replace with your actual mapping)
    category_to_idx = {'electronics': 0}  # Example category mapping
    data['text_'] = data['text_'].apply(preprocess_text)
    data['category_idx'] = data['category'].map(category_to_idx)


    # Define the Dataset class for prediction
    class ReviewDataset:
        def __init__(self, df, vocab, max_length=MAX_LENGTH):
            self.texts = df['text_']
            self.categories = df['category_idx']
            self.ratings = df['rating']
            self.vocab = vocab
            self.max_length = max_length

        def __getitem__(self, idx):
            text = self.texts.iloc[idx].split()
            category = self.categories.iloc[idx]
            rating = self.ratings.iloc[idx]
            text_tokens = [self.vocab[token] for token in text[:self.max_length]]
            text_tensor = torch.tensor(text_tokens + [self.vocab["<pad>"]] * (self.max_length - len(text_tokens)))
            return text_tensor, torch.tensor(category, dtype=torch.long), torch.tensor(rating, dtype=torch.float)


    # Build vocabulary from the training data (replace with actual vocab if available)
    def yield_tokens(data_iter):
        for text in data_iter:
            yield text.split()


    # Build vocab from a small sample (replace this with your actual training dataset)
    train_data = data['text_'].tolist()
    vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    print("Got data.")


    # Define the model class
    class FakeReviewDetector(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, category_dim, pad_idx):
            super(FakeReviewDetector, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.category_embedding = nn.Embedding(category_dim, embed_dim)
            self.fc = nn.Linear(hidden_dim + embed_dim + 1, output_dim)  # +1 for rating feature
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x, category, rating):
            # Text Embedding
            embedded = self.embedding(x)
            _, (hidden, _) = self.lstm(embedded)

            # Category Embedding
            category_embedded = self.category_embedding(category)

            # Concatenate LSTM hidden state, category embedding, and rating
            combined = torch.cat((hidden[-1], category_embedded, rating.unsqueeze(1)), dim=1)

            # Final Fully Connected Layer
            output = self.fc(combined)
            return self.softmax(output)


    # Load the trained model
    vocab_size = 35709 # len(vocab)
    embed_dim = 128
    hidden_dim = 64
    output_dim = 2
    category_dim = 10 # len(category_to_idx)
    pad_idx = vocab["<pad>"]

    model = FakeReviewDetector(vocab_size, embed_dim, hidden_dim, output_dim, category_dim, pad_idx)
    model.load_state_dict(torch.load(MODEL_DIRECTORY))
    model.to(DEVICE)
    model.eval()
    print("Loaded model from disk.")


    # Make predictions on the input data (directly using the 3 variables: category, text, rating)
    def predict_review(text, category, rating):
        # Preprocess the text
        text = preprocess_text(text)

        # Convert to tensor
        text_tokens = [vocab[token] for token in text.split()[:MAX_LENGTH]]
        text_tensor = torch.tensor(text_tokens + [vocab["<pad>"]] * (MAX_LENGTH - len(text_tokens)))
        category_idx = category_to_idx.get(category, 0)  # Default to 0 if category is unknown
        rating_tensor = torch.tensor(float(rating))

        # Move tensors to the correct device (GPU or CPU)
        text_tensor = text_tensor.to(DEVICE)
        category_idx = torch.tensor(category_idx).to(DEVICE)
        rating_tensor = rating_tensor.to(DEVICE)

        # Model prediction
        with torch.no_grad():
            output = model(text_tensor.unsqueeze(0), category_idx.unsqueeze(0), rating_tensor.unsqueeze(0))
            prediction = torch.argmax(output, dim=1).item()
            return output


    # Example usage
    chatgpt_generated = """I recently purchased the Play & Learn Educational Building Blocks for my 4-year-old, and I couldn't be happier with the product! The set includes 100 colorful, sturdy blocks that are the perfect size for little hands to grab and stack.
    
    From the moment we opened the box, my child was captivated. The vibrant colors and easy-to-handle design make it not only fun but also safe for young kids. I particularly love how these blocks encourage creativity and problem-solving. Within minutes, my little one was building towers, bridges, and even a "castle for teddy bears" (her words!).
    
    What sets this toy apart is its educational value. While my child plays, she's developing fine motor skills and learning about shapes, colors, and even basic physics principles like balance and stability. It's screen-free fun that I feel good about as a parent.
    
    Cleanup is a breeze too. The set comes with a handy storage box, so there's no mess left behind after playtime.
    
    If you're looking for an engaging, durable, and educational toy, I highly recommend the Play & Learn Educational Building Blocks. It's a fantastic investment in both fun and learning!"""

    amazon_review = """ My son i love that toy is good quality"""

    THRESHOLD = 0.15

    print("Predicting...")
    prediction = predict_review(text, category, rating)
    print(prediction)
    print(prediction[0][0])
    if prediction[0][0].item() > THRESHOLD:
        res = "AI-created"
    else:
        res = "Human-created"
    print(f"Prediction: {prediction}")
    return res

    # You can use this function for any number of reviews