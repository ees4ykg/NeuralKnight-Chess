# NeuralKnight-Chess ♟️

A chess AI built with a convolutional neural network that learns move patterns from elite-level Lichess games. Includes a simple GUI for playing against the trained model.

##  Overview

This is a learning project exploring how neural networks can learn chess through supervised learning on game data. The model uses a basic CNN architecture to predict moves based on board positions, trained on games from high-rated Lichess players.

##  Architecture

### Neural Network Design
- **Input**: 14-channel 8×8 board representation
  - Channels 0-11: One-hot encoded piece positions (6 white + 6 black piece types)
  - Channel 12: Legal move destinations
  - Channel 13: Current turn (1 for white, 0 for black)
- **Architecture**: 
  - Conv2D layer (14→64 channels, 3×3 kernel, ReLU)
  - Conv2D layer (64→128 channels, 3×3 kernel, ReLU)
  - Flatten
  - Fully connected layer (8192→256, ReLU)
  - Output layer (256→num_moves)
- **Output**: Logits for all possible UCI moves seen in training

### Training Setup
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (lr=0.0001)
- **Batch Size**: 64
- **Gradient Clipping**: Max norm 1.0

##  Project Structure

```
NeuralKnight-Chess/
├── data_preprocessing.py    # Converts PGN files to numpy arrays
├── torch_dataset.py         # PyTorch Dataset wrapper
├── torch_model.py           # CNN model definition
├── train.py                 # Training script
├── predict.py               # Move prediction with trained model
├── play_chess_gui.py        # Pygame GUI for playing against AI
├── requirements.txt         # Dependencies
├── raw_training_data/       # PGN files (not included in repo)
├── cleaned_data/            # Preprocessed training data
│   ├── x_data.npy          # Board states
│   ├── y_data.npy          # Move labels
│   └── mapping.npy         # Move to index mapping
└── models/                  # Saved model weights
```

##  Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ees4ykg/NeuralKnight-Chess.git
cd NeuralKnight-Chess
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Play Against the AI

```bash
python play_chess_gui.py
```

- Click pieces to select and move them
- You play White, AI plays Black
- Pawns auto-promote to Queens

#### Train Your Own Model

1. Add PGN files to `raw_training_data/` directory

2. Preprocess the data:
```bash
python data_preprocessing.py
```

3. Train the model:
```bash
python train.py
```

4. Test with command-line interface:
```bash
python predict.py
```

##  Training Data

Currently trained on Lichess Elite database games (June 2021 - November 2022). The dataset contains games from high-rated players in PGN format.

To train on different data, modify the `limit_of_games` parameter in `data_preprocessing.py`.

##  How It Works

### Board Encoding
Each board position is represented as a 14×8×8 tensor:
```
Channels 0-5:   White pieces (P, N, B, R, Q, K)
Channels 6-11:  Black pieces (p, n, b, r, q, k)
Channel 12:     Legal move destinations
Channel 13:     Turn indicator (1=White, 0=Black)
```

### Move Selection
The model outputs probabilities for moves it has seen during training. At inference:
1. Model predicts probability distribution over all learned moves
2. Predictions are sorted by probability
3. First legal move from the sorted list is selected

This ensures the AI only makes valid moves according to chess rules.

##  Limitations

- Model only knows moves it has seen in training data
- No strategic planning or position evaluation
- Performance depends heavily on training data quality
- Can struggle with unusual positions or endgames
- Not competitive with traditional chess engines

##  What I Learned

This project helped me understand:
- Converting game data into neural network inputs
- Building and training CNNs with PyTorch
- Working with the `python-chess` library
- The difference between supervised learning and how actual strong chess engines work

##  Dependencies

- PyTorch - Neural network framework
- python-chess - Chess logic and move validation  
- NumPy - Array operations
- Pygame - GUI
- tqdm - Training progress bars

See `requirements.txt` for versions.

##  Acknowledgments

- [Lichess](https://lichess.org/) for free access to game databases
- [python-chess](https://python-chess.readthedocs.io/) library
- PyTorch team


*Note: This is an educational project demonstrating neural network applications in game AI. It is not intended to compete with traditional chess engines like Stockfish or AlphaZero, although maybe someday we will reach that level ;)*
