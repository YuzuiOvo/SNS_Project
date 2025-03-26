# SNS_Project
SNS_Project - an Oracle chatbot
# Financial Prediction Assistant

## Project introduction

Financial Prediction Assistant is a machine learning-based application designed to predict the price of gold and the US dollar index. The project uses historical data and a variety of machine learning models (such as random forests, neural networks, etc.) to make predictions, providing a user-friendly interface for users to enter future dates and obtain predictions.

## Data set
This project uses the following data sets:
- 'gold_prices.csv' : contains historical gold price data.
- 'dxy_index.csv' : contains historical dollar index data.
- 'merged_data.csv' : Data set that combines gold price and dollar index, including date, gold price, and dollar index.
- 'X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv' : Features and target variables used to train and test the model.
## Technology stack
- ** Programming language: Python
- ** Data processing **: Pandas, NumPy
- ** Machine Learning **: Scikit-learn, TensorFlow (Keras)
- ** Visualization **: Matplotlib, Chart.js
- **Web Framework **: Flask
- ** Models **: Random forest regression, MLP regression, GRU, XGBoost, etc
## How to run the project
1. ** Environmental Preparation **:
- Ensure that Python 3.x is installed.
- Use 'pip' to install the required libraries:
```bash
pip install pandas numpy scikit-learn tensorflow flask matplotlib
```

2. ** Data Preparation **:
- Place the Data set files in the 'Data/' folder.

3. ** Training Model **:
- Run the 'Final VersionX.py' file to train the model and save the trained model and Scaler:
```bash
python scripts/Final VersionX.py
```

4. ** Launch the application **:
- Run 'app_launcher.py' to launch the Flask application:
```bash
python scripts/app_launcher.py
```
- After the application is launched, the browser will automatically open and you can enter the future date in the interface to get the prediction of the gold price and the US dollar index.

## Test project
- You can test the predictive power of the model by entering different future dates. The application will return the forecast gold price and the dollar index.

## Contribution
Contributions of any kind are welcome! If you have suggestions or find problems, please submit a question or pull request.
