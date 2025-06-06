# Kyrgyz Alphabet Recognition Neural Network

A project for recognizing special letters of the Kyrgyz alphabet (Ң, Ө, Ү) using a simple neural network built from scratch in Python.

## Description

This project implements a neural network for classifying images of three special letters from the Kyrgyz alphabet:
- **Ң** (Ng)
- **Ө** (O with diaeresis)  
- **Ү** (Y with diaeresis)

The neural network is built from scratch without using ready-made deep learning frameworks, making the code educational and understandable for learning machine learning fundamentals.

## Features

- ✅ Simple architecture: input layer (784 neurons) → hidden layer (72 neurons) → output layer (3 neurons)
- ✅ Activation function: sigmoid
- ✅ Backpropagation implemented manually
- ✅ Ability to save and load model weights
- ✅ Interactive interface for testing on custom images
- ✅ Results visualization using matplotlib
- ✅ User interface in Kyrgyz language

## Requirements

```bash
numpy
matplotlib
pandas
opencv-python
scikit-learn
Pillow
```

## Installation

1. Clone the repository:
```bash
git clone git@github.com:mirba047/100-prisoners-problem.git
cd 100-prisoners-problem
```

2. Install dependencies:
```bash
pip install numpy matplotlib pandas opencv-python scikit-learn Pillow
```

## Project Structure

```
├── main.py                     # Main script
├── kg_alphabet.csv             # Training images dataset
├── kg_alphabet_labels.csv      # Labels for training data
├── weights/                    # Directory for saved model weights
│   ├── weights_input_to_hidden.csv
│   ├── weights_hidden_to_output.csv
│   ├── bias_input_to_hidden.csv
│   └── bias_hidden_to_output.csv
└── img/self_imgs/             # Directory for custom test images
```

## Usage

1. **Prepare your data**: Ensure you have `kg_alphabet.csv` and `kg_alphabet_labels.csv` files in the project directory.

2. **Run the program**:
```bash
python main.py
```

3. **Choose mode**:
   - **Training mode (1)**: Train a new neural network from scratch
   - **Load weights mode (0)**: Load pre-trained weights and test the model

### Training Mode

When you select training mode, the program will:
- Train the neural network for 5 epochs
- Show loss and accuracy for each epoch
- Display test accuracy after training
- Allow you to test custom images
- Optionally save the trained weights

### Testing Mode

When you load pre-trained weights, you can:
- Test the model on custom images
- See predictions with confidence visualization

## How It Works

### Neural Network Architecture

The network consists of:
- **Input layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden layer**: 72 neurons with sigmoid activation
- **Output layer**: 3 neurons (one for each letter class)

### Image Processing

1. Images are converted to grayscale
2. Normalized to range [0, 1]
3. Colors are inverted (white background becomes black)
4. Reshaped to 784-dimensional vector

### Training Process

1. **Forward propagation**: Calculate outputs through the network
2. **Loss calculation**: Mean squared error
3. **Backpropagation**: Calculate gradients
4. **Weight updates**: Apply gradients with learning rate

## Dataset Format

- **Images**: CSV file where each row represents a flattened 28×28 image (784 values)
- **Labels**: CSV file with corresponding class labels (0, 1, 2)

## Model Performance

The model achieves reasonable accuracy on the three-class classification task. Performance metrics are displayed during training and testing phases.

## Customization

You can easily modify:
- **Learning rate**: Change `learning_rate` variable
- **Network architecture**: Modify hidden layer size
- **Training epochs**: Adjust `epochs` variable
- **Activation functions**: Replace sigmoid with other functions

## Testing Custom Images

1. Place your test images in `./img/self_imgs/` directory
2. Images should be 28×28 pixels for best results
3. Run the program and enter the image filename when prompted

## Contributing

Feel free to contribute by:
- Adding more letters to the dataset
- Improving the neural network architecture
- Adding data augmentation techniques
- Implementing other activation functions
- Adding validation metrics

## Educational Value

This project demonstrates:
- Neural network fundamentals
- Backpropagation algorithm
- Image preprocessing techniques
- Model training and evaluation
- Weight saving/loading mechanisms

Perfect for students learning machine learning concepts without the complexity of modern frameworks.

## License

This project is open source and available under the MIT License.

---

*Note: The user interface is in Kyrgyz language, making this project particularly valuable for Kyrgyz-speaking learners of machine learning.*
