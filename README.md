# 🎬 Movie Review Sentiment Analysis App

> A machine learning-powered application that analyzes movie reviews and predicts sentiment using Natural Language Processing.

## 🌟 Features

- Real-time sentiment analysis of movie reviews
- Emoji-aware processing
- Interactive feedback system
- Continuous model improvement through user feedback
- Beautiful visualization of results
- Debug mode for transparency
- GitHub integration for data persistence
- Automatic model retraining

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Framework**: scikit-learn
- **NLP Processing**: NLTK
- **Data Storage**: GitHub
- **Version Control**: Git

## 📋 Requirements

```bash
streamlit
pandas
numpy
scikit-learn
nltk
emoji
python-dotenv
requests
joblib
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Private-Fox7/Sentiment-Analysis-App.git
cd Sentiment-Analysis-App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```env
GITHUB_TOKEN=your_github_token
GITHUB_REPO=your_github_repo
```

4. Run the application:
```bash
streamlit run new_app.py
```

## 💡 Usage

1. Enter a movie review in the text area
2. Click "Analyze Sentiment"
3. View the prediction and confidence score
4. Provide feedback to help improve the model
5. (Optional) Toggle debug mode to see preprocessing details

## 🎯 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: TF-IDF Vectorization
- **Preprocessing**: 
  - Text cleaning
  - Emoji conversion
  - Stop word removal
  - Lemmatization

## 📊 Performance

- Current model accuracy: 94%
- Continuous improvement through user feedback
- Real-time model retraining

## 🔄 Feedback System

The app includes a feedback mechanism that:
1. Collects user corrections
2. Stores feedback in GitHub
3. Triggers model retraining
4. Updates the model automatically

## 🌐 GitHub Integration

- Automatic data persistence
- Version control for feedback dataset
- Continuous model improvement
- Accuracy tracking

## 🐛 Debug Mode

Toggle debug mode to view:
- Preprocessing steps
- Emoji conversions
- Feature importance
- Feedback data statistics

## 👥 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NLTK for NLP processing
- Streamlit for the web interface
- scikit-learn for machine learning capabilities

## 📬 Contact

Private Fox - [Private-Fox7](https://github.com/Private-Fox7)

Project Link: [https://github.com/Private-Fox7/Sentiment-Analysis-App](https://github.com/Private-Fox7/Sentiment-Analysis-App)
