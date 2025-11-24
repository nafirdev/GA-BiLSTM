# GA-BiLSTM: Genetic Algorithm Optimized Bidirectional LSTM for Time-Series Forecasting

GA-BiLSTM is an enhanced time-series forecasting framework that combines a **Bidirectional LSTM (BiLSTM)** architecture with a **Genetic Algorithm (GA)** for automated hyperparameter optimization.  
This project is an upgraded version of the previous GA-LSTM model — designed to capture deeper temporal patterns and improve prediction accuracy on complex sequential datasets.

---

## 🚀 Why BiLSTM Instead of Standard LSTM?

Traditional LSTMs process sequences **only in one direction** (forward in time).  
However, in many real-world time-series tasks, future context can also help improve prediction accuracy.

### **BiLSTM Advantages**
✔ **Captures both past and future dependencies** by processing data forward *and* backward.  
✔ **Learns richer and more robust temporal patterns** compared to a unidirectional LSTM.  
✔ **Significantly improves performance** on noisy or chaotic datasets.  
✔ **Reduces information loss** from earlier time steps.  
✔ **More stable training** when sequences have long-term dependencies.

### **Why This Project Is an Advancement**
This model integrates:

- **Two stacked Bidirectional LSTM layers**  
- **Dropout regularization** for improved generalization  
- **Adam optimizer with gradient clipping**  
- **Early Stopping** to prevent overfitting  
- **GA-driven optimization** for:
  - LSTM units  
  - Dropout rate  
  - Batch size  
  - Number of epochs  

The result is a **smarter, deeper, and more stable forecasting engine** than the previous GA-LSTM implementation.

---

## ✨ Key Features
- **Automated Hyperparameter Optimization** using a Genetic Algorithm  
- **Bidirectional Sequence Learning** for improved accuracy  
- **Early Stopping** to prevent overfitting  
- **Gradient Clipping** for stable training  
- **Simple, readable, and modular code design**  
- **Works with any time-series dataset**

---

## 📦 Installation
```bash
pip install numpy pandas tensorflow scikit-learn
````

---

## 🧠 Model Architecture (BiLSTM)

The core model consists of:

* BiLSTM layer (return sequences)
* Dropout
* BiLSTM layer
* Dropout
* Dense output layer (5 units for multi-step forecasting)

```python
model = Sequential()
model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
model.add(Dropout(dropout_rate))
model.add(Bidirectional(LSTM(units)))
model.add(Dropout(dropout_rate))
model.add(Dense(5))
```

This structure helps the model see **both backward and forward temporal context**, allowing it to understand the sequence more effectively.

---

## 🧬 Genetic Algorithm Optimization

The GA optimizes four key hyperparameters:

* `lstm_units`
* `dropout_rate`
* `batch_size`
* `epochs`

The algorithm follows this pipeline:

1. Create an initial random population
2. Evaluate each individual with MSE
3. Select top-performing candidates
4. Apply crossover
5. Apply mutation
6. Evolve through generations
7. Output the best hyperparameter combination

Example output:

```
Best Parameters: [64, 0.3, 32, 40]
```

---

## ▶️ Running the Optimization

```python
best_parameters = optimize_lstm_parameters(
    xtrain, ytrain, xval, yval,
    parameter_space,
    population_size=10,
    num_generations=7
)
print('Best Parameters:', best_parameters)
```

---

## 📁 Project Structure

```
GA-BiLSTM/
│
├── GA_BiLSTM.py     # Main script (model + GA)
├── README.md        # Project documentation
```

---

## 🔮 Future Improvements

* Add GRU and Conv1D hybrid architectures
* Add visualization of GA performance per generation
* Multi-step forecasting evaluation (MAE, RMSE, MAPE)
* Save and load best-trained model

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

If you want, I can also generate:

✅ A **version with diagrams & architecture images**
✅ A **research-paper style README**
✅ A **minimal version for PyPI packaging**

Just tell me which style you prefer.

```
```
