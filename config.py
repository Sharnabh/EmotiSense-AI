# EmotiSense Configuration File
# Adjust these settings based on your hardware and requirements

# Model Configuration
MODEL_CONFIG = {
    # Baseline Models
    'baseline': {
        'tfidf_max_features': 5000,
        'tfidf_ngram_range': (1, 2),
        'random_state': 42,
        'use_class_weight': True,  # Handle class imbalance
        'cv_folds': 3
    },
    
    # Transformer Models
    'transformer': {
        'model_name': 'distilbert-base-uncased',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'save_steps': 500,
        'eval_steps': 500,
        'logging_steps': 100
    }
}

# Dataset Configuration
DATASET_CONFIG = {
    'name': 'dair-ai/emotion',
    'train_split': 'train',
    'validation_split': 'validation', 
    'test_split': 'test',
    'text_column': 'text',
    'label_column': 'label',
    'emotion_labels': ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
    'cache_dir': './data/cache'
}

# Training Configuration
TRAINING_CONFIG = {
    'output_dir': './outputs',
    'model_save_dir': './saved_models',
    'results_save_dir': './results',
    'visualizations_dir': './visualizations',
    'logs_dir': './logs'
}

# Streamlit App Configuration
STREAMLIT_CONFIG = {
    'page_title': '🧠 EmotiSense - Emotion Classification',
    'page_icon': '🎭',
    'layout': 'wide',
    'theme': 'light',
    'sidebar_state': 'expanded'
}

# Evaluation Metrics
EVALUATION_CONFIG = {
    'primary_metric': 'accuracy',
    'additional_metrics': ['macro_f1', 'weighted_f1', 'precision', 'recall'],
    'plot_confusion_matrix': True,
    'save_predictions': True,
    'error_analysis': True
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'use_gpu': True,  # Set to False if no GPU available
    'gpu_memory_fraction': 0.8,
    'num_workers': 4,  # For data loading
    'pin_memory': True
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_to_file': True,
    'log_file': 'emotisense.log'
}
