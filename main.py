import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Conv1D, GlobalMaxPooling1D, Dropout, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('G:\\deepLearning\\rp_implementation\\Toxic-Comment-Classification-Challenge-master\\data\\train.csv')

X = df['comment_text'].values
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_layer = Input(shape=(100,))
embedding = Embedding(input_dim=max_features, output_dim=128)(input_layer)
bilstm = Bidirectional(LSTM(units=64, return_sequences=True))(embedding)
attention = Attention()([bilstm, bilstm])
conv1d = Conv1D(filters=64, kernel_size=3, activation='relu')(attention)
global_max_pooling = GlobalMaxPooling1D()(conv1d)
dense = Dense(units=128, activation='relu')(global_max_pooling)
dropout = Dropout(rate=0.5)(dense)
output_layer = Dense(units=6, activation='sigmoid')(dropout)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

def predict_toxicity(custom_comments):
    # Preprocess custom input
    sequences = tokenizer.texts_to_sequences(custom_comments)
    padded_sequences = pad_sequences(sequences, maxlen=100)
    
    predictions = model.predict(padded_sequences)
    
    for i, comment in enumerate(custom_comments):
        print(f"\nComment: {comment}")
        print("Predictions:")
        print(f"  Toxic: {predictions[i][0]:.4f}")
        print(f"  Severe Toxic: {predictions[i][1]:.4f}")
        print(f"  Obscene: {predictions[i][2]:.4f}")
        print(f"  Threat: {predictions[i][3]:.4f}")
        print(f"  Insult: {predictions[i][4]:.4f}")
        print(f"  Identity Hate: {predictions[i][5]:.4f}")

custom_comments = [
    "You are a terrible person and deserve to be punished.",  
    "You are the worst scum on this planet, I hope you suffer.", 
    "This is the most obscene thing I have ever seen!",  
    "If you don't stop, I will make sure you regret it.",  
    "You are a complete idiot and an absolute waste of space.",  
    "People like you are the reason why there is so much hatred in this world." 
]

predict_toxicity(custom_comments)
