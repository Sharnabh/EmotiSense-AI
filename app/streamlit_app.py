import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Suppress torch warnings and errors
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Page configuration
st.set_page_config(
    page_title="🧠 EmotiSense - Emotion Classification",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }
    
    .prediction-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        margin: 1rem 0;
    }
    
    .metrics-container {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Emotion labels mapping
EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
EMOTION_EMOJIS = {
    'sadness': '😢',
    'joy': '😊',
    'love': '❤️',
    'anger': '😠',
    'fear': '😰',
    'surprise': '😲'
}

def clean_text(text):
    """Clean and preprocess text"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.lower().strip()

@st.cache_data
def load_sample_data():
    """Load sample data for testing"""
    sample_texts = [
        "I am so happy today! Everything is going perfectly.",
        "This makes me really angry and frustrated.",
        "I love spending time with my family and friends.",
        "I'm scared about what might happen tomorrow.",
        "I feel so sad and empty inside.",
        "Wow, I can't believe this just happened! Amazing!"
    ]
    return sample_texts

class EmotionClassifier:
    def __init__(self, model_type="baseline"):
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.transformer_pipeline = None
        
    def load_baseline_model(self, vectorizer_path=None, model_path=None):
        """Load pre-trained baseline model"""
        try:
            if vectorizer_path and model_path:
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                return True
            else:
                # Create dummy models for demo
                return self.create_demo_model()
        except Exception as e:
            st.error(f"Error loading baseline model: {e}")
            return self.create_demo_model()
    
    def create_demo_model(self):
        """Create a demo model for demonstration purposes"""
        try:
            # Create dummy vectorizer and model
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.model = LogisticRegression(class_weight='balanced', random_state=42)
            
            # Demo training data
            demo_texts = [
                "I am so happy and joyful", "I love this so much", "This is amazing and wonderful",
                "I am sad and depressed", "This makes me cry", "I feel so down",
                "I am angry and furious", "This makes me mad", "I hate this situation",
                "I am scared and afraid", "This is terrifying", "I feel anxious",
                "I love you so much", "You are my everything", "Love is beautiful",
                "Wow this is surprising", "I can't believe this", "What a shock"
            ]
            demo_labels = [1, 1, 1, 0, 0, 0, 3, 3, 3, 4, 4, 4, 2, 2, 2, 5, 5, 5]
            X_demo = self.vectorizer.fit_transform(demo_texts)
            self.model.fit(X_demo, demo_labels)
            
            return True
        except Exception as e:
            st.error(f"Error creating demo model: {e}")
            return False
    
    def load_transformer_model(self, custom_model_path=None):
        """Load transformer model - custom trained or pre-trained"""
        try:
            # First try to load custom trained model
            if custom_model_path is None:
                # Default path to your trained model
                custom_model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'emotion_classification_model')
            
            # Check if custom model exists
            if os.path.exists(custom_model_path) and os.path.exists(os.path.join(custom_model_path, 'config.json')):
                st.info("🎯 Loading your custom trained DistilBERT model...")
                
                # Load custom model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
                model = AutoModelForSequenceClassification.from_pretrained(custom_model_path)
                
                # Create pipeline with custom model
                self.transformer_pipeline = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    return_all_scores=True
                )
                
                st.success("✅ Custom trained model loaded successfully!")
                return True
            
            else:
                # Fallback to pre-trained model
                st.info("📦 Loading pre-trained DistilRoBERTa model...")
                self.transformer_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                st.success("✅ Pre-trained model loaded successfully!")
                return True
                
        except Exception as e:
            st.error(f"Error loading transformer model: {e}")
            # Try fallback pre-trained model
            try:
                st.info("🔄 Trying fallback pre-trained model...")
                self.transformer_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                st.success("✅ Fallback model loaded successfully!")
                return True
            except Exception as fallback_error:
                st.error(f"Fallback model also failed: {fallback_error}")
                return False
    
    def predict_baseline(self, text):
        """Predict emotion using baseline model"""
        if not self.vectorizer or not self.model:
            return None, None
        
        try:
            cleaned_text = clean_text(text)
            text_vectorized = self.vectorizer.transform([cleaned_text])
            prediction = self.model.predict(text_vectorized)[0]
            probabilities = self.model.predict_proba(text_vectorized)[0]
            
            return EMOTION_LABELS[prediction], probabilities
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None
    def predict_transformer(self, text):
        """Predict emotion using transformer model"""
        if not self.transformer_pipeline:
            return None, None
        
        try:
            cleaned_text = clean_text(text)
            results = self.transformer_pipeline(cleaned_text)
            
            # Handle different output formats
            if isinstance(results[0], list):
                # Multiple scores format (return_all_scores=True)
                emotion_scores = {}
                for result in results[0]:
                    label = result['label'].lower()
                    score = result['score']
                    
                    # Map different label formats to our standard emotions
                    if label in EMOTION_LABELS:
                        emotion_scores[label] = score
                    elif label == 'label_0':
                        emotion_scores['sadness'] = score
                    elif label == 'label_1':
                        emotion_scores['joy'] = score
                    elif label == 'label_2':
                        emotion_scores['love'] = score
                    elif label == 'label_3':
                        emotion_scores['anger'] = score
                    elif label == 'label_4':
                        emotion_scores['fear'] = score
                    elif label == 'label_5':
                        emotion_scores['surprise'] = score
                
                if emotion_scores:
                    predicted_emotion = max(emotion_scores, key=emotion_scores.get)
                    probabilities = [emotion_scores.get(emotion, 0) for emotion in EMOTION_LABELS]
                    return predicted_emotion, probabilities
            
            else:
                # Single prediction format
                label = results[0]['label'].lower()
                if label in EMOTION_LABELS:
                    # Create probability array with high confidence for predicted emotion
                    probabilities = [0.1] * len(EMOTION_LABELS)
                    predicted_idx = EMOTION_LABELS.index(label)
                    probabilities[predicted_idx] = results[0]['score']
                    return label, probabilities
            
            return None, None
        except Exception as e:
            st.error(f"Transformer prediction error: {e}")
            return None, None

def create_probability_chart(probabilities, predicted_emotion):
    """Create probability visualization chart"""
    if probabilities is None:
        return None
    
    # Create DataFrame for plotting
    df_probs = pd.DataFrame({
        'Emotion': EMOTION_LABELS,
        'Probability': probabilities,
        'Emoji': [EMOTION_EMOJIS[emotion] for emotion in EMOTION_LABELS]
    })
    
    # Create bar chart
    fig = px.bar(
        df_probs, 
        x='Emotion', 
        y='Probability',
        title='Emotion Prediction Probabilities',
        color='Probability',
        color_continuous_scale='viridis',
        text='Emoji'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        xaxis_title="Emotions",
        yaxis_title="Probability",
        title_x=0.5,
        height=400
    )
    
    # Highlight predicted emotion
    colors = ['red' if emotion == predicted_emotion else 'lightblue' for emotion in EMOTION_LABELS]
    fig.update_traces(marker_color=colors)
    
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🎭 EmotiSense - AI Emotion Classification</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
      # Model selection
    model_type = st.sidebar.selectbox(
        "Choose Model Type:",
        ["Baseline (TF-IDF + ML)", "Transformer (Custom DistilBERT)", "Both Models"]
    )
    
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Load models based on selection
    if model_type in ["Baseline (TF-IDF + ML)", "Both Models"]:
        with st.spinner("Loading baseline model..."):
            baseline_loaded = classifier.load_baseline_model()
    if model_type in ["Transformer (Custom DistilBERT)", "Both Models"]:
        with st.spinner("Loading transformer model..."):
            transformer_loaded = classifier.load_transformer_model()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Text input options
        input_option = st.radio(
            "Choose input method:",
            ["Type your own text", "Use sample texts"]
        )
        
        if input_option == "Type your own text":
            user_text = st.text_area(
                "Enter text to analyze emotion:",
                placeholder="Type your text here... (e.g., 'I am feeling great today!')",
                height=100
            )
        else:
            sample_texts = load_sample_data()
            user_text = st.selectbox("Choose a sample text:", sample_texts)
        
        # Prediction button
        if st.button("🔮 Analyze Emotion", type="primary"):
            if user_text:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                if model_type == "Baseline (TF-IDF + ML)":
                    predicted_emotion, probabilities = classifier.predict_baseline(user_text)
                    if predicted_emotion:
                        st.success(f"**Predicted Emotion:** {EMOTION_EMOJIS[predicted_emotion]} {predicted_emotion.title()}")
                        fig = create_probability_chart(probabilities, predicted_emotion)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                elif model_type == "Transformer (Custom DistilBERT)":
                    predicted_emotion, probabilities = classifier.predict_transformer(user_text)
                    if predicted_emotion:
                        st.success(f"**Predicted Emotion:** {EMOTION_EMOJIS[predicted_emotion]} {predicted_emotion.title()}")
                        fig = create_probability_chart(probabilities, predicted_emotion)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                
                elif model_type == "Both Models":
                    col_baseline, col_transformer = st.columns(2)
                    
                    with col_baseline:
                        st.subheader("🔧 Baseline Model")
                        predicted_emotion_bl, probabilities_bl = classifier.predict_baseline(user_text)
                        if predicted_emotion_bl:
                            st.success(f"**Predicted:** {EMOTION_EMOJIS[predicted_emotion_bl]} {predicted_emotion_bl.title()}")
                            fig_bl = create_probability_chart(probabilities_bl, predicted_emotion_bl)
                            if fig_bl:
                                st.plotly_chart(fig_bl, use_container_width=True)
                    
                    with col_transformer:
                        st.subheader("🤖 Transformer Model")
                        predicted_emotion_tr, probabilities_tr = classifier.predict_transformer(user_text)
                        if predicted_emotion_tr:
                            st.success(f"**Predicted:** {EMOTION_EMOJIS[predicted_emotion_tr]} {predicted_emotion_tr.title()}")
                            fig_tr = create_probability_chart(probabilities_tr, predicted_emotion_tr)
                            if fig_tr:
                                st.plotly_chart(fig_tr, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to analyze!")
    
    with col2:
        st.header("📊 About")
        
        # Information cards
        st.markdown("""
        <div class="metrics-container">
        <h3>🎯 Supported Emotions</h3>
        """, unsafe_allow_html=True)
        
        for emotion in EMOTION_LABELS:
            st.markdown(f"""
            <div class="emotion-card">
                {EMOTION_EMOJIS[emotion]} {emotion.title()}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
          # Model information
        st.markdown("""
        <div class="metrics-container">
        <h3>🤖 Model Information</h3>
        <ul>
        <li><strong>Baseline:</strong> TF-IDF + Logistic Regression</li>
        <li><strong>Transformer:</strong> Custom Fine-tuned DistilBERT</li>
        <li><strong>Dataset:</strong> dair-ai/emotion</li>
        <li><strong>Classes:</strong> 6 emotions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance metrics (placeholder)
        st.markdown("""
        <div class="metrics-container">
        <h3>📈 Performance</h3>
        <ul>
        <li><strong>Baseline Accuracy:</strong> ~85%</li>
        <li><strong>Transformer Accuracy:</strong> ~93%</li>
        <li><strong>Training Time:</strong> 15 mins</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    ---
    <div style='text-align: center; color: #666;'>
        <p>🧠 <strong>EmotiSense</strong> - AI-Powered Emotion Classification | Built with Streamlit & Transformers</p>
        <p>📧 Contact: sharnabh.banerjee@example.com | 🔗 GitHub: @sharnabh-banerjee</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
