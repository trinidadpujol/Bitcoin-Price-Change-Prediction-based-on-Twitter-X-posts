# NLP Bitcoin Price Impact Analysis

A comprehensive Natural Language Processing project that analyzes the impact of news and social media content on Bitcoin price movements. This project combines data collection, preprocessing, machine learning, and real-time analysis to predict short-term Bitcoin price changes based on textual content.

## 🎯 Project Overview

This project aims to:
- **Collect and process** Bitcoin-related news articles and social media content
- **Filter relevant content** using AI-powered classification
- **Train machine learning models** to predict Bitcoin price movements
- **Analyze the relationship** between textual content and cryptocurrency price changes
- **Generate synthetic content** for training and testing purposes

## 📁 Project Structure

```
NLP_BTC/
├── requirements.txt                 # Project dependencies
├── README.md                       # This file
└── project/             # Main project directory
    ├── training/                   # Training data and preprocessing
    │   ├── notebooks/             # Jupyter notebooks for data processing
    │   │   ├── btc_news_dataset.ipynb      # News dataset creation
    │   │   ├── btc_prices.ipynb            # Bitcoin price data collection
    │   │   ├── news_filtering.py           # AI-powered news classification
    │   │   └── tweet_gen.py                # Synthetic tweet generation
    │   └── data/                  # Training datasets
    │       ├── dataset_cut.csv             # Filtered news dataset
    │       ├── filtered_news.csv           # AI-classified relevant news
    │       └── dataset_target_year_time.csv # Final training dataset
    ├── test/                      # Testing and validation
    │   └── test/
    │       ├── notebooks/         # Testing notebooks
    │       │   ├── csv_processing.ipynb    # Data preprocessing for testing
    │       │   └── tweets_scraping.py      # Twitter data collection
    │       └── data/              # Test datasets
    │           ├── musk_2025.csv           # Elon Musk tweets
    │           ├── saylor_2025.csv         # Michael Saylor tweets
    │           ├── btcnews_2025.csv        # Btcnews tweets
    │           ├── documenting_bitcoin_2025.csv # Documenting Bitcoin tweets
    │           ├── anthony_pompliano_2025.csv   # Anthony Pompliano tweets
    │           └── tweets_dataset_v.csv    # Combined test dataset
    └── model/                     # Machine learning models
        ├── notebooks/             # Model training notebooks
        │   └── pipeline_training.ipynb    # Main training pipeline
        ├── data/                  # Model-specific data
        └── saved_models/          # Trained model files
            └── 128/               # Model with 128 hidden units
                ├── pytorch_model.bin      # Trained PyTorch model
                ├── vocab.txt              # Tokenizer vocabulary
                ├── tokenizer_config.json  # Tokenizer configuration
                └── special_tokens_map.json # Special tokens mapping
```

## 🚀 Features

### 📊 Data Collection & Processing
- **Bitcoin Price Data**: Historical price data from Binance API
- **News Articles**: Bitcoin-related news from various sources
- **Social Media**: Twitter/X content from influential Bitcoin personalities
- **Real-time Processing**: Automated data collection and preprocessing

### 🤖 AI-Powered Content Classification
- **Relevance Filtering**: Uses Azure OpenAI to classify news relevance
- **Directional Analysis**: Determines bullish/bearish sentiment
- **Impact Assessment**: Evaluates potential price-moving content

### 🧠 Machine Learning Models
- **BERT-based Regression**: Fine-tuned FinBERT for price prediction
- **Custom Loss Functions**: Asymmetric Huber loss for directional accuracy
- **Model Evaluation**: Comprehensive metrics including sign accuracy

### 📈 Price Impact Analysis
- **Short-term Predictions**: 1-2 hour price movement forecasts
- **Directional Accuracy**: Focus on predicting price direction
- **Risk Assessment**: Asymmetric penalties for different prediction errors

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd NLP_BTC
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up API keys** (optional for full functionality):
   - Azure OpenAI API key for news classification
   - Binance API credentials for price data
   - Twitter API credentials for social media scraping

## 📋 Dependencies

### Core Data Processing
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computations
- `matplotlib>=3.7.0` - Data visualization
- `seaborn>=0.12.0` - Statistical plotting

### Machine Learning
- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.30.0` - Hugging Face transformers
- `scikit-learn>=1.3.0` - Traditional ML algorithms
- `scipy>=1.10.0` - Scientific computing

### APIs and Web Services
- `openai>=1.0.0` - Azure OpenAI integration
- `python-binance>=1.0.19` - Binance cryptocurrency API
- `requests>=2.31.0` - HTTP requests
- `twscrape>=0.5.0` - Twitter/X scraping

### Data Handling
- `datasets>=2.14.0` - Hugging Face datasets
- `tqdm>=4.65.0` - Progress tracking
- `jupyter>=1.0.0` - Jupyter notebooks

## 🔧 Usage

### 1. Data Collection

**Bitcoin Price Data**:
```python
# Run btc_prices.ipynb to collect historical price data
# This fetches hourly BTC/USDT data from Binance
```

**News Dataset Creation**:
```python
# Run btc_news_dataset.ipynb to create the news dataset
# Filters news from 2020-2025 for relevant Bitcoin content
```

### 2. Content Classification

**News Filtering**:
```python
# Run news_filtering.py to classify news relevance
# Uses Azure OpenAI to determine price-moving potential
```

**Tweet Generation**:
```python
# Run tweet_gen.py to generate synthetic content
# Creates training data for social media impact analysis
```

### 3. Model Training

**Training Pipeline**:
```python
# Run pipeline_training.ipynb to train the BERT model
# Includes data preprocessing, model training, and evaluation
```

### 4. Testing and Validation

**Data Processing**:
```python
# Run csv_processing.ipynb to prepare test data
# Combines multiple social media sources
```

**Social Media Scraping**:
```python
# Run tweets_scraping.py to collect real-time data
# Scrapes tweets from influential Bitcoin personalities
```

## 📊 Model Architecture

### FinBERT-based Regression Model
- **Base Model**: `yiyanghkust/finbert-pretrain`
- **Architecture**: BERT + Regression head (128 hidden units)
- **Input**: Text content (max 128 tokens)
- **Output**: Price change prediction (regression)

### Custom Loss Function
- **Asymmetric Huber Loss**: Different penalties for positive/negative predictions
- **Directional Focus**: Emphasizes correct sign prediction
- **Risk Management**: Higher penalties for negative target mismatches

### Training Configuration
- **Batch Size**: 16
- **Max Length**: 128 tokens
- **Epochs**: 9
- **Learning Rate**: Optimized for regression task

## 📈 Performance Metrics

The model is evaluated using:
- **Mean Squared Error (MSE)**: Overall prediction accuracy
- **Mean Absolute Error (MAE)**: Average prediction error
- **R² Score**: Model fit quality
- **Pearson Correlation**: Linear relationship strength
- **Sign Accuracy**: Directional prediction accuracy
- **Precision/Recall**: For positive/negative class classification

## 🔍 Key Features

### Content Classification
- **Relevance Scoring**: AI-powered news relevance assessment
- **Directional Analysis**: Bullish/bearish sentiment classification
- **Impact Prediction**: Short-term price movement potential

### Data Processing
- **Multi-source Integration**: News, social media, and price data
- **Temporal Alignment**: Precise timestamp matching
- **Quality Filtering**: Automated content relevance assessment

### Model Training
- **Transfer Learning**: Leverages pre-trained financial BERT
- **Custom Loss**: Optimized for cryptocurrency price prediction
- **Comprehensive Evaluation**: Multiple performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Kaggle**: For the dataset of news articles
- **Hugging Face**: For the transformers library and FinBERT model
- **Binance**: For cryptocurrency price data API
- **Azure OpenAI**: For content classification capabilities
- **Twitter/X**: For social media data access
- **University of San Andrés**: For the project

## 📞 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This project requires API keys for full functionality. Please ensure you have the necessary credentials for Azure OpenAI, Binance, and Twitter APIs if you plan to run the complete pipeline. 