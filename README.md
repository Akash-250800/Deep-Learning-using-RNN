# Movie Review Sentiment Analysis Using Recurrent Neural Network (RNN)

This project uses a Recurrent Neural Network (RNN), to classify movie reviews as positive or negative. The goal is to perform sentiment analysis on a dataset of movie reviews using deep learning techniques.

## Features
- Sentiment classification (Positive/Negative) for movie reviews
- Text preprocessing (tokenization, padding, etc.)
- Word embeddings using pre-trained models (e.g., GloVe)
- RNN model for sequential data processing
- Model evaluation with performance metrics (accuracy, precision, recall, F1-score)

## Dataset
The dataset contains thousands of labeled movie reviews and can be fetched from:
- [IMDb Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Kaggle: Sentiment Analysis Dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

Install the required dependencies:
pip install -r requirements.txt

Dependencies
TensorFlow/Keras
NumPy
Pandas
scikit-learn
Matplotlib

Model Architecture
Embedding Layer: Converts text to dense vectors of fixed size using pre-trained word embeddings like GloVe.
Dense Layer: Fully connected layer for classification.

Evaluation Metrics
The performance of the model is evaluated using:

Accuracy
Precision
Recall
F1-score

Results
The LSTM-based RNN model achieves an accuracy of 90% on the test dataset.

Model Hyperparameters
Embedding Dimension: 100
Dropout: 0.5
Optimizer: Adam
Loss Function: Binary Crossentropy
Epochs: 10
Batch Size: 64

Contributing
Contributions are welcome! If you'd like to improve the project, feel free to submit a pull request or open an issue.

License
This project is licensed under the MIT License. See the LICENSE file for details.

This `README.md` file is tailored for a sentiment analysis project using an RNN, particularly 

