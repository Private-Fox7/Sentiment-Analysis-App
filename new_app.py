import streamlit as st
import joblib
import time
import emoji
import re
import os
import nltk


# Define a writable directory for NLTK data (inside the app directory)
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")

# Ensure the directory exists
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Set the NLTK data path
nltk.data.path.append(NLTK_DATA_DIR)

# Now download necessary resources
nltk.download('punkt', download_dir=NLTK_DATA_DIR)
nltk.download('stopwords', download_dir=NLTK_DATA_DIR)
nltk.download('wordnet', download_dir=NLTK_DATA_DIR)
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import subprocess
import threading
import requests
import base64
from datetime import datetime
from dotenv import load_dotenv  # type: ignore
import nltk


# Load environment variables
load_dotenv()

# Hardcoded GitHub settings
import os
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
import os
GITHUB_REPO = os.getenv("GITHUB_REPO", "Private-Fox7/Sentiment-Analysis-App")
GITHUB_PATH = "feedback_dataset.csv"  # Path to the CSV file in the repo

nltk.download('punkt')
# Set a custom path for NLTK data that's writable in most environments
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download required NLTK resources with better error handling
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, download_dir=nltk_data_path, quiet=True)
            print(f"Successfully downloaded {resource} to {nltk_data_path}")
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")
            
    # Verify the resources are available
    try:
        # Test if resources work
        sample_text = "This is a test sentence."
        tokens = word_tokenize(sample_text)
        stops = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("testing")
        print("NLTK resources verified successfully.")
    except LookupError as e:
        print(f"Resource verification failed: {str(e)}")
    except Exception as e:
        print(f"Unexpected error during verification: {str(e)}")

# Call the download function
download_nltk_data()

# GitHub Integration Functions
def update_github_csv(dataframe):
    """
    Update a CSV file in a GitHub repository using the GitHub REST API
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to save as CSV
    
    Returns:
    bool: True if successful, False otherwise
    """
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Step 1: Check if file exists and get its SHA if it does
    url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}'
    response = requests.get(url, headers=headers)
    
    # Prepare the file content
    csv_content = dataframe.to_csv(index=False)
    encoded_content = base64.b64encode(csv_content.encode()).decode()
    
    if response.status_code == 200:
        # File exists, get the SHA for update
        sha = response.json()['sha']
        data = {
            'message': f'Update feedback data - {time.strftime("%Y-%m-%d %H:%M:%S")}',
            'content': encoded_content,
            'sha': sha
        }
        # Update existing file
        update_response = requests.put(url, headers=headers, json=data)
        return update_response.status_code in (200, 201)
    elif response.status_code == 404:
        # File doesn't exist, create it
        data = {
            'message': f'Add feedback data - {time.strftime("%Y-%m-%d %H:%M:%S")}',
            'content': encoded_content
        }
        # Create new file
        create_response = requests.put(url, headers=headers, json=data)
        return create_response.status_code in (200, 201)
    else:
        # Other error
        print(f"Error checking file: {response.status_code} - {response.text}")
        return False
    

def load_github_csv():
    """
    Load a CSV file from a GitHub repository
    
    Returns:
    pd.DataFrame: DataFrame containing the CSV data, or empty DataFrame if file not found
    """
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    url = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}'
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # File exists, decode content
        content = base64.b64decode(response.json()['content']).decode('utf-8')
        
        # Return as DataFrame
        return pd.read_csv(pd.io.common.StringIO(content))
    elif response.status_code == 404:
        # File doesn't exist yet
        return pd.DataFrame(columns=["text", "label", "timestamp"])
    else:
        # Other error
        print(f"Error loading file: {response.status_code} - {response.text}")
        return pd.DataFrame(columns=["text", "label", "timestamp"])
    
# Create session state variables for async training
if "training_in_progress" not in st.session_state:
    st.session_state.training_in_progress = False
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "training_message" not in st.session_state:
    st.session_state.training_message = ""
if "training_error" not in st.session_state:
    st.session_state.training_error = None
if "last_model_update" not in st.session_state:
    st.session_state.last_model_update = "Not updated yet"
if "github_settings_initialized" not in st.session_state:
    st.session_state.github_settings_initialized = True  # Always initialized

import requests
import requests

# GitHub Raw URL for Accuracy File
GITHUB_ACCURACY_URL = "https://raw.githubusercontent.com/Private-Fox7/Sentiment-Analysis-App/main/accuracy.txt"

def fetch_model_accuracy():
    """Fetch model accuracy from GitHub and format it correctly."""
    response = requests.get(GITHUB_ACCURACY_URL)

    if response.status_code == 200:
        try:
            accuracy = float(response.text.strip())  # Convert to float
            return f"{accuracy * 100:.2f}%"  # Convert from fraction (0.9960) to percentage (99.60%)
        except ValueError:
            return "N/A (Invalid Accuracy Format)"
    else:
        return "N/A (Failed to load)"

# Fetch and display accuracy in Streamlit
model_accuracy = fetch_model_accuracy()


import requests
import joblib
import io

MODEL_URL = "https://raw.githubusercontent.com/Private-Fox7/Sentiment-Analysis-App/main/model.pkl"
VECTORIZER_URL = "https://raw.githubusercontent.com/Private-Fox7/Sentiment-Analysis-App/main/vectorizer.pkl"

def load_model_from_github(url):
    """Loads a model from GitHub into memory without saving locally"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, stream=True, headers=headers)
    print(f"🔗 Fetching model from: {url}")
    print(f"📜 Status Code: {response.status_code}")
    
    if response.status_code == 200:
        model_file = io.BytesIO(response.content)  # Store in memory (not local disk)
        return joblib.load(model_file)
    else:
        print(f"❌ Failed to download model. Status: {response.status_code}")
        return None

@st.cache_resource
def load_models():
    """Loads model and vectorizer from GitHub directly into memory"""
    print("🔄 Attempting to load model and vectorizer...")

    _model = load_model_from_github(MODEL_URL)
    _vectorizer = load_model_from_github(VECTORIZER_URL)

    if _model:
        print("✅ Model loaded successfully!")
    else:
        print("❌ Model could not be loaded.")

    if _vectorizer:
        print("✅ Vectorizer loaded successfully!")
    else:
        print("❌ Vectorizer could not be loaded.")
    
    return _model, _vectorizer

model, vectorizer = load_models()
# Function to run training asynchronously
def retrain_model_async():
    try:
        # Indicate that training is in progress
        st.session_state.training_in_progress = True
        st.session_state.training_message = "🔄 Retraining model in the background..."
        st.session_state.training_complete = False

        # Run training script as a separate process (non-blocking)
        process = subprocess.Popen(["python", "test1.py", "--quick_train"],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)

        # Do not wait for process completion, allow UI to stay responsive
        time.sleep(3)  # Give time for the process to start properly

    except Exception as e:
        st.session_state.training_message = "⚠️ Error starting retraining"
        st.session_state.training_error = str(e)
        st.session_state.training_complete = True
        st.session_state.training_in_progress = False
        return

    # Monitor the process in the background
    while process.poll() is None:
        time.sleep(5)  # Check every 5 seconds
        st.session_state.training_message = "⏳ Model is still training..."

    # Process completed, check exit status
    if process.returncode == 0:
        st.session_state.training_message = "✅ Model retrained successfully!"
        st.session_state.training_complete = True
        st.session_state.training_error = None
        st.session_state.last_model_update = time.strftime("%Y-%m-%d %H:%M:%S")

        # Clear cache and refresh app to load new model
        st.cache_resource.clear()
        st.rerun()
    else:
        st.session_state.training_message = "⚠️ Retraining failed!"
        st.session_state.training_error = process.stderr.read()
        st.session_state.training_complete = True

# Load trained model and vectorizer
try:
    model, vectorizer = load_models()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model = None
    vectorizer = None

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Custom emoji sentiment dictionary - MUST match the one used during training
emoji_sentiments = {
    # Positive emojis
    "slightly_smiling_face": "positive_sentiment",
    "grinning": "positive_sentiment",
    "heart_eyes": "positive_sentiment",
    "thumbs_up": "positive_sentiment",
    "laughing": "positive_sentiment",
    "grinning_face": "positive_sentiment",
    "smile": "positive_sentiment",
    "smiley": "positive_sentiment",
    "wink": "positive_sentiment",
    "blush": "positive_sentiment",
    "heart": "positive_sentiment",
    "star": "positive_sentiment",
    "sun": "positive_sentiment",
    "clap": "positive_sentiment",
    
    # Negative emojis
    "angry": "negative_sentiment",
    "cry": "negative_sentiment",
    "sob": "negative_sentiment",
    "thumbs_down": "negative_sentiment",
    "scream": "negative_sentiment",
    "rage": "negative_sentiment",
    "disappointed": "negative_sentiment",
    "worried": "negative_sentiment",
    "confused": "negative_sentiment",
    "tired_face": "negative_sentiment",
    "fearful": "negative_sentiment",
    "weary": "negative_sentiment"
}

# Function to preprocess text - MUST match training preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert emojis to sentiment indicators
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Replace emoji codes with sentiment words
    words = text.split()
    words = [emoji_sentiments.get(word, word) for word in words]
    text = " ".join(words)
    
    # Basic text preprocessing
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize and remove stopwords
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english')) - {"not", "no", "very", "isn't", "wasn't", "hadn't"}
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)

# Show pop-up message only once when the app loads
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = True
    st.toast("👋 Welcome to the Movie Review Sentiment Analyzer! Type a review to get started!", icon="🎬")

# App Title
st.title("🎬 Movie Review Sentiment Analysis 🎭")
st.write("Enter a movie review below, and we'll analyze its sentiment for you!")

# Sidebar Information
st.sidebar.title("ℹ️ About the App")
st.sidebar.info("""
This app analyzes movie reviews and predicts whether the sentiment is positive or negative using Machine Learning.

**Try using emojis in your review!** 
The model understands emojis like:
- 😊 👍 ❤️ (positive)
- 😡 👎 😢 (negative)
""")
# Display model accuracy in Streamlit sidebar
st.sidebar.info(f"📊 **Model Accuracy:** {model_accuracy}")
st.sidebar.info("""
This app is linked to this GitHub Repository 
   👉 Private-Fox7/Sentiment-Analysis-App
""")


# Show model update timestamp
if "last_model_update" in st.session_state:
    st.sidebar.info(f"🔄 Model last updated: {st.session_state.last_model_update}")

# Show status of model training if in progress
if st.session_state.training_in_progress:
    st.sidebar.warning(st.session_state.training_message)
elif st.session_state.training_complete:
    # Show success or error message
    if st.session_state.training_error:
        st.sidebar.error(st.session_state.training_message)
        with st.sidebar.expander("View Error Details"):
            st.code(st.session_state.training_error)
    else:
        st.sidebar.success(st.session_state.training_message)
        # Reload model if training was successful
        try:
            st.cache_resource.clear()
            model, vectorizer = load_models()
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")

# User Input
user_review = st.text_area("💬 Enter your movie review here:", "", height=150)

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Show preprocessing details")

if not model or not vectorizer:
    st.error("❌ Model or vectorizer is missing. Please check the logs.")
else:
    st.success("✅ Model and vectorizer are loaded successfully!")

if st.sidebar.button("🔄 Reload Model"):
    st.cache_resource.clear()
    st.success("✅ Model cache cleared. The latest model will be loaded from GitHub!")
    model, vectorizer = load_models()  # Ensure new model is fetched from GitHub

    # Temporary placeholders for notifications
    delete_message = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    delete_message.warning("⏳ Deleting old model files... Please wait.")
    st.sidebar.info("""😀😀Please be patient do not close the page, as the model is being trained on more than 25,000 sample files and being added on GitHub Repository.
    Your Patience is appreciated✌✌
    Estimated time 3 minutes """)

    # Start retraining process
    st.sidebar.info("🔄 Starting model retraining with quick training...")
    process = subprocess.Popen(["python", "test1.py", "--quick_train"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,encoding='UTF-8')

    while process.poll() is None:  # While retraining is still running
        progress_bar.progress(50)  # Show progress at 50%
        progress_bar.progress(80)
        progress_bar.progress(95)
        progress_bar.progress(99)
        status_text.info("⏳ Model is training... Please wait.")
        time.sleep(2)  # Update every 2 seconds
    # Monitor the training process
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
        # You can print or log the output for debugging
            print(output.strip())


    # Check retraining result
    if process.returncode == 0:
         progress_bar.progress(100)  # Show complete progress
         status_text.success("🎉 Model updated and loaded into the app!")
         model, vectorizer = load_models()  # Load the new model
         st.session_state.last_model_update = time.strftime("%Y-%m-%d %H:%M:%S")
         st.rerun() 
    else:
        st.sidebar.error("⚠️ Model retraining failed! Check logs.")
        st.text(process.stderr.read())  # Show error message

    # Clear progress and notifications
    progress_bar.empty()
    status_text.empty()

# Initialize session state for retraining
if "show_retraining" not in st.session_state:
    st.session_state.show_retraining = False

if st.button("🔍 Analyze Sentiment"):
    if model is None or vectorizer is None:
        st.error("⚠️ Model or vectorizer is not loaded. Please check your files.")
    elif user_review.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        # Process input using the same preprocessing as during training
        processed_input = preprocess_text(user_review)
        
        # Show preprocessing details if debug mode is on
        if debug_mode:
            st.write("### Preprocessing Steps:")
            st.write(f"**Original Input:** {user_review}")
            
            # Show emoji conversion
            demojized = emoji.demojize(user_review, delimiters=(" ", " "))
            st.write(f"**After emoji conversion:** {demojized}")
            
            st.write(f"**Final processed text:** {processed_input}")
        
        # Perform prediction
        with st.spinner("Analyzing sentiment..."):
            vectorized_text = vectorizer.transform([processed_input])
            prediction = model.predict(vectorized_text)[0]
            
            # Get prediction probability
            confidence = model.predict_proba(vectorized_text).max()
            time.sleep(1.03)  # Slight delay for user experience

        # Display result with effects
        if prediction == "positive":
            st.balloons()  # Fun effect for positive sentiment
            st.success(f"🎉 Yay! Your review is **Positive** (Confidence: {confidence:.2%})! 😊")
        else:
            st.snow()  # Sad effect for negative sentiment
            st.error(f"💔 Oh no! Your review is **Negative** (Confidence: {confidence:.2%}). 😞 Don't worry, maybe the next movie will be better!")
            
        # Emoji explanation
        if any(emoji in user_review for emoji in ['😊', '👍', '❤️', '😡', '👎', '😢']):
            st.info("**Note:** The model detected emojis in your review and used them as sentiment indicators!")

        # Enable feedback section after prediction
        st.session_state.show_feedback = True

# Feedback system - only show after prediction
feedback_file = "feedback_dataset.csv"

# Only show the feedback section after a prediction
if "show_feedback" in st.session_state and st.session_state.show_feedback:
    st.write("### Was this prediction correct?")
    feedback = st.radio("Feedback:", ("Yes", "No"), key="feedback_radio")

    if feedback == "No":
        correct_label = st.selectbox("What should the correct sentiment be?", ("positive", "negative"))

    if st.button("Submit Feedback"):
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create new data entry with timestamp
        new_data = pd.DataFrame({
            "text": [user_review], 
            "label": [correct_label],
            "timestamp": [timestamp]
        })

        # Initialize variables to track storage success
        github_success = False

        try:
            # Load existing data from GitHub
            existing_data = load_github_csv()

            # Add timestamp column if it doesn't exist
            if "timestamp" not in existing_data.columns:
                existing_data["timestamp"] = [timestamp] * len(existing_data)

            # Append new data
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)

            # Update file on GitHub
            github_success = update_github_csv(updated_data)

            if github_success:
                st.success("Thank you! Your feedback has been recorded to GitHub.")
            else:
                st.error("Error saving feedback to GitHub. Check your token and repository settings.")

            # Determine feedback count for retraining decision
            feedback_count = len(updated_data)

            # Check if retraining is needed (every 1 feedback entries)
            if feedback_count % 1 == 0 and feedback_count > 0:
                st.info("Retraining will begin in the background...")

                # Start background training in a separate thread
                training_thread = threading.Thread(target=retrain_model_async)
                training_thread.daemon = True
                training_thread.start()
            else:
                st.info(f"Model will be retrained after {1 - (feedback_count % 1)} more feedback entries.")

            # Reset UI to avoid feedback loop
            st.session_state.show_feedback = False

        except Exception as e:
            st.error(f"Error saving feedback to GitHub: {str(e)}")

# Show feedback data if debug mode is on
if debug_mode:
    with st.sidebar.expander("View Feedback Data"):
        try:
            st.subheader("GitHub Feedback Data")
            github_data = load_github_csv()
            st.dataframe(github_data)
            st.text(f"Total GitHub entries: {len(github_data)}")
        except Exception as e:
            st.error(f"Error loading GitHub feedback data: {str(e)}")

     
