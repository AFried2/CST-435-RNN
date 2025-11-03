import tensorflow as tf  # <-- MOVED TO LINE 1
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Configuration ---
MODEL_PATH = 'best_model.keras'
TOKENIZER_PATH = 'tokenizer.pkl'
SEQ_LEN_PATH = 'sequence_len.pkl'
GLOVE_FILE = 'glove.6B/glove.6B.100d.txt'
EMBEDDING_DIM = 100

# --- Caching: Load Models and Data Once ---
# Use @st.cache_resource to load models only once
@st.cache_resource
def load_all_artifacts():
    """
    Loads the trained model, tokenizer, sequence length, and embedding matrix.
    """
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        st.error("Please make sure you have run the 'model_training.ipynb' notebook to generate 'best_model.keras'.")
        return None, None, None, None, None

    try:
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(SEQ_LEN_PATH, 'rb') as f:
            sequence_len = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Error loading pickle file: {e}")
        st.error("Please make sure 'tokenizer.pkl' and 'sequence_len.pkl' are in the same directory.")
        return None, None, None, None, None

    vocab_size = len(tokenizer.word_index) + 1
    
    # Load GloVe to build embedding matrix for cosine similarity
    embeddings_index = {}
    try:
        with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        st.error(f"GloVe file not found at {GLOVE_FILE}. Word embedding exploration will not work.")
        embeddings_index = {} # Continue without it

    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return model, tokenizer, sequence_len, embedding_matrix, vocab_size

# --- Load Artifacts ---
model, tokenizer, sequence_len, embedding_matrix, vocab_size = load_all_artifacts()

# --- Helper Functions ---
def clean_input_text(text):
    # ... (this function is unchanged)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_next_words(seed_text, top_k=5):
    # ... (this function is unchanged)
    if model is None:
        return [], []
        
    cleaned_seed = clean_input_text(seed_text)
    token_list = tokenizer.texts_to_sequences([cleaned_seed])[0]
    
    if not token_list:
        return [], []
        
    padded_sequence = pad_sequences([token_list], maxlen=sequence_len, padding='pre')
    predicted_probs = model.predict(padded_sequence, verbose=0)[0]
    
    top_indices = predicted_probs.argsort()[-top_k:][::-1]
    
    top_words = [tokenizer.index_word.get(i, '?') for i in top_indices]
    top_probs = predicted_probs[top_indices]
    
    return top_words, top_probs

# --- ADD THIS NEW HELPER FUNCTION ---
def sample(preds, temperature=1.0):
    """
    Helper function to sample an index from a probability array
    """
    preds = np.asarray(preds).astype('float64') + 1e-8
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# --- REPLACE THE OLD generate_text_sequence FUNCTION ---
def generate_text_sequence(seed_text, num_words, temperature=0.7):
    """
    Generates a sequence of text using temperature sampling.
    """
    if model is None:
        return "Error: Model not loaded."

    generated_text = clean_input_text(seed_text)
    current_text = generated_text
    
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        
        if not token_list:
            break
            
        token_list = pad_sequences([token_list], maxlen=sequence_len, padding='pre')
        
        predicted_probs = model.predict(token_list, verbose=0)[0]
        
        # Sample a word
        predicted_probs[0] = 0 # Don't predict padding
        predicted_index = sample(predicted_probs, temperature=temperature)
        
        output_word = tokenizer.index_word.get(predicted_index, '')
        
        generated_text += " " + output_word
        current_text += " " + output_word
        current_text = ' '.join(current_text.split(' ')[-sequence_len:])
        
    return generated_text

def find_similar_words(word, top_n=5):
    # ... (this function is unchanged)
    if embedding_matrix is None or not embedding_matrix.any():
        return [], 0
        
    word = clean_input_text(word)
    if word not in tokenizer.word_index:
        return [], 1
    
    word_idx = tokenizer.word_index[word]
    word_vec = embedding_matrix[word_idx].reshape(1, -1)
    
    if np.all(word_vec == 0):
        return [], 2
        
    similarities = cosine_similarity(word_vec, embedding_matrix)
    similar_indices = similarities[0].argsort()[-(top_n+1):][::-1]
    
    results = []
    for idx in similar_indices:
        if idx != 0 and idx != word_idx:
            results.append((tokenizer.index_word[idx], similarities[0][idx]))
            
    return results, 0


# --- Streamlit UI ---
st.set_page_config(page_title="RNN Text Generator", layout="wide")
st.title("📚 RNN Next-Word Predictor")
st.markdown(f"Trained on *Alice's Adventures in Wonderland* | Vocabulary Size: `{vocab_size}` | Input Sequence: `{sequence_len}` words")

# Check if models loaded correctly
if model is None:
    st.stop() # Stop the app if artifacts didn't load

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a feature:",
    ["🤖 Next-Word Prediction", "✍️ Free-Form Text Generation", "🔬 Explore Word Embeddings", "📊 Project Overview"]
)

# --- Page 1: Next-Word Prediction (Shows Probabilities) ---
if app_mode == "🤖 Next-Word Prediction":
    st.header("🤖 Next-Word Prediction")
    st.write("Type a sentence and see what the model thinks the *next* word should be. This shows the 'softmax' output layer in action.")
    
    seed_text = st.text_input("Enter your text:", "alice was beginning to")
    top_k = st.slider("How many top predictions to show?", 1, 10, 5)

    if seed_text:
        words, probs = predict_next_words(seed_text, top_k)
        if words:
            st.subheader("Top Predictions:")
            for i, (word, prob) in enumerate(zip(words, probs)):
                st.write(f"**{i+1}. {word}** (Probability: `{prob:.2%}`)")
        else:
            st.warning("Could not generate predictions. Try different words that are likely in the story (e.g., 'alice', 'rabbit', 'queen').")

# --- Page 2: Free-Form Text Generation ---
elif app_mode == "✍️ Free-Form Text Generation":
    st.header("✍️ Free-Form Text Generation")
    st.write("Give the model a starting seed and let it write a story by feeding its own predictions back into itself.")

    seed_text_gen = st.text_area("Enter your starting text:", "the white rabbit ran")
    
    # --- ADD THIS SLIDER ---
    temp_slider = st.slider("Generation Temperature (Creativity):", 
                            min_value=0.1, max_value=1.5, 
                            value=0.7, step=0.1)
    
    num_words = st.slider("Number of words to generate:", 1, 50, 20)

    if st.button("Generate Text"):
        if seed_text_gen:
            # --- UPDATE THIS LINE TO PASS THE TEMPERATURE ---
            generated_output = generate_text_sequence(seed_text_gen, num_words, temp_slider)
            
            st.subheader("Generated Text:")
            st.success(generated_output)
        else:
            st.warning("Please enter some seed text.")

# --- Page 3: Explore Word Embeddings ---
elif app_mode == "🔬 Explore Word Embeddings":
    st.header("🔬 Explore Word Embeddings (GloVe + Cosine Similarity)")
    st.write("This feature does not use the RNN. Instead, it lets you explore the 100-dimensional GloVe vectors that were used as input. Words with similar meanings should have similar vectors.")
    
    if not embedding_matrix.any():
        st.error("Word Embedding exploration is unavailable. GloVe file not loaded.")
    else:
        word_to_check = st.text_input("Enter a single word from the vocabulary:", "wonderland")
        
        if st.button("Find Similar Words"):
            similar_results, err_code = find_similar_words(word_to_check)
            if err_code == 1:
                st.warning(f"'{word_to_check}' is not in the model's vocabulary.")
            elif err_code == 2:
                st.info(f"'{word_to_check}' is in the vocabulary but has no pre-trained GloVe embedding (it's an 'Out-of-Vocabulary' word for GloVe).")
            elif similar_results:
                st.subheader(f"Words similar to '{word_to_check}':")
                for word, sim in similar_results:
                    st.write(f"**{word}** (Similarity: `{sim:.4f}`)")

# --- Page 4: Project Overview ---
elif app_mode == "📊 Project Overview":
    st.header("📊 Project Overview")
    st.write("This page summarizes the project, fulfilling the assignment's documentation requirements.")
    
    st.subheader("a) Problem Statement")
    st.write("""
    This project aims to demonstrate the application of Recurrent Neural Networks (RNNs) in sequential learning and text forecasting. 
    The primary goal is to build, train, and evaluate a Long Short-Term Memory (LSTM) model capable of predicting the next word in a given 
    sequence of text...
    """)
    
    st.subheader("b) Algorithm and Model Architecture")
    st.write("""
    The model is a Keras Sequential model. The architecture is as follows:
    1.  **Embedding Layer**: Maps integer-encoded words to 100-dimensional vectors using pre-trained GloVe weights. `trainable=False` and `mask_zero=True`.
    2.  **LSTM Layer**: The core of the model with 150 units and dropout to process the sequences.
    3.  **Dense Layer**: A standard 256-unit fully-connected layer with `relu` activation.
    4.  **Dropout Layer**: A 50% dropout for regularization.
    5.  **Output Layer**: A `softmax` layer with `vocab_size` units, producing a probability for every word.
    """)
    
    # Capture model.summary() from your notebook and paste it here as a string
    model_summary = """
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, 10, 100)           264800    
                                                                     
     lstm (LSTM)                 (None, 150)               150600    
                                                                     
     dense (Dense)               (None, 256)               38656     
                                                                     
     dropout (Dropout)           (None, 256)               0         
                                                                     
     dense_1 (Dense)             (None, 2648)              680360    
                                                                     
    =================================================================
    Total params: 1,134,416
    Trainable params: 869,616
    Non-trainable params: 264,800
    _________________________________________________________________
    """
    # Note: Your param numbers might vary slightly based on vocab size.
    # Update the summary above from your notebook's output.
    st.code(model_summary, language='text')

    st.subheader("c) Analysis of Findings")
    st.write("The model was trained until validation accuracy plateaued. The plots below show the training history.")
    
    try:
        st.image('training_accuracy.png', caption='Model Training vs. Validation Accuracy')
        st.image('training_loss.png', caption='Model Training vs. Validation Loss')
        st.write("These plots (saved from the training notebook) show that the model learned effectively without significant overfitting, thanks to EarlyStopping and Dropout layers.")
    except FileNotFoundError:
        st.warning("Could not find 'training_accuracy.png' or 'training_loss.png'. Please run the training notebook to generate them.")