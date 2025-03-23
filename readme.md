
---

# Shallow Neural Network with TensorFlow

This repository contains a simple implementation of a shallow neural network using Python and TensorFlow to classify handwritten digits from the MNIST dataset. Follow these steps to recreate this project locally or use it as a starting point for your own machine learning projects.

## Prerequisites

- Python (3.6+)
- TensorFlow (2.x)

## Setup

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/shallow-neural-network-tensorflow.git
cd shallow-neural-network-tensorflow
```

2. **Create and activate a virtual environment (optional but recommended):**

   - On Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - On Unix/Linux/MacOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install required packages:**

```bash
pip install tensorflow numpy matplotlib
```

## Project Structure

```
shallow-neural-network-tensorflow/
│
├── README.md
├── main.py
└── results/
    └── ...
```

## Getting Started

1. **Create a new file named `main.py`** in the project root folder and paste the following code:

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data: normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy}')

# Make predictions using the trained model (example for the first test image)
predictions = model.predict(x_test)
predicted_digit = np.argmax(predictions[0])
print(f'Predicted digit: {predicted_digit}')
```

2. **Run the script:**

```bash
python main.py
```

## Results

The project includes a `results/` folder containing sample output, including:

- Test accuracy of the trained model
- Predicted digit for the first test image (0-indexed)

You can find these results in the `results/` folder after running `main.py`.

## Customization and Further Improvement

- Adjust the neural network architecture by adding more layers, changing the number of neurons in each layer, or trying different activation functions.
- Experiment with different optimizers and learning rates.
- Implement data augmentation techniques to artificially increase the size of your training dataset.
- Tune hyperparameters using techniques like Grid Search or Random Search.

## Contributing

If you find any issues or have suggestions for improving this project, feel free to open an issue or pull request on the GitHub repository. Contributions are welcome!

---

Enjoy building and exploring shallow neural networks with TensorFlow! If you found this `README.md` helpful, please consider giving it a star on GitHub. Happy coding!