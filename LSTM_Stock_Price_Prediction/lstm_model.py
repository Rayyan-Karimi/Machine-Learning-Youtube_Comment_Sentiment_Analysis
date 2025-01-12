import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from data_preparation import prepare_data

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    '''
    1. Data Preparation (prepare_data)
    X_train and y_train:
    Used as input-output pairs for training the LSTM model.
    Each X_train sample is a sequence of window_size (e.g., 10) normalized closing prices.
    Each corresponding y_train sample is the normalized closing price of the next day.
    X_test and y_test:

    Used for validation during training and for evaluation after training.
    X_test consists of sequences of normalized prices, and y_test has the actual normalized values for the next day.
    '''
    
    # Prepare data
    file_path = "data/AAPL_preprocessed.csv"
    # file_path = "data/MSFT_preprocessed.csv"
    X_train, X_test, y_train, y_test, scaler = prepare_data(file_path)
    
    
    '''
    X_train.shape[1] is the sequence length (e.g., 10 days).
    X_train.shape[2] is the number of features (e.g., 1 for closing price).
    Layers:
    First LSTM layer with return_sequences=True: Outputs sequences for further layers.
    Second LSTM layer without return_sequences: Outputs a single value (hidden state).
    Dense layer: Outputs a single value (predicted normalized price).
    '''
    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    print(model.summary())

    '''
     Training the Model
    The fit method trains the model on the training data (X_train and y_train) and validates it on the test data (X_test and y_test):

    Inputs:
    X_train, y_train: Training data.
    validation_data=(X_test, y_test): Used to calculate validation loss during training.
    EarlyStopping Callback:
    Stops training if validation loss does not improve for 5 consecutive epochs.
    Restores the model to the weights with the lowest validation loss.
    Outputs:
    Training loss (loss) and validation loss (val_loss) for each epoch.
    '''
    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )

    '''
    TO IDENTIFY:
    Overfitting (if training loss is low, but validation loss is high).
    Under fitting (if both losses are high).
    '''
    # Plot training history
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    # Evaluate model
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: (Mean Squared Error) {test_loss}")

    # Predict on test data
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Denormalize predictions
    y_test_actual = scaler.inverse_transform(y_test)     # Denormalize actual prices

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label="Actual Prices")
    plt.plot(predictions, label="Predicted Prices")
    plt.legend()
    plt.show()
    plt.savefig('output/lstm_plot_AAPL(Apple stock).png')
    # plt.savefig('result/lstm_plot_MSFT(Microsoft stock).png')
    model.save('models/stock_price_lstm_model.h5')

