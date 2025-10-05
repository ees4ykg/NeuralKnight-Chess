# NeuralKnight-Chess ♟️

A deep learning chess AI that learns to play chess by training on elite-level games from Lichess. The project features a convolutional neural network (CNN) that predicts moves based on board positions, along with an interactive GUI for playing against the AI.

## 🎯 Overview

NeuralKnight-Chess is a personal project that demonstrates how neural networks can learn chess strategy from high-level human games. For now the model is trained on hundreds of thousands of games from elite Lichess players and uses a CNN architecture to predict the best move given any board position.

## 🏗️ Architecture

### Neural Network Design
- **Input**: 14-channel 8×8 board representation
  - Channels 0-11: One-hot encoded piece positions (6 white + 6 black piece types)
  - Channel 12: Legal move destinations
  - Channel 13: Current turn (1 for white, 0 for black)
- **Architecture**: 
  - Conv2D (14→64 channels, 3×3 kernel, ReLU)
  - Conv2D (64→128 channels, 3×3 kernel, ReLU)
  - Flatten
  - Fully Connected (8192→256, ReLU)
  - Fully Connected (256→num_moves)
- **Output**: Probability distribution over all possible UCI moves

### Training Details
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (learning rate: 0.0001)
- **Batch Size**: 64
- **Gradient Clipping**: Max norm of 1.0
- **Hardware**: CUDA-enabled GPU training

## 📁 Project Structure

```
NeuralKnight-Chess/
├── data_preprocessing.py    # PGN to numpy array conversion
├── torch_dataset.py         # PyTorch Dataset class
├── torch_model.py           # Neural network model definition
├── train.py                 # Model training script
├── predict.py               # Move prediction inference
├── play_chess_gui.py        # Interactive Pygame GUI
├── requirements.txt         # Python dependencies
├── raw_training_data/       # PGN files from Lichess elite games
│   └── lichess_elite_*.pgn
├── cleaned_data/            # Preprocessed numpy arrays
│   ├── x_data.npy          # Board states
│   ├── y_data.npy          # Move labels
│   └── mapping.npy         # UCI move to index mapping
└── models/                  # Trained model weights
    ├── 10_epochs_model.pth
    └── model1_heavy_move_to_idx
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ees4ykg/NeuralKnight-Chess.git
cd NeuralKnight-Chess
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Playing Against the AI

Launch the interactive GUI to play against the trained model:

```bash
python play_chess_gui.py
```

**Controls**:
- Click on a piece to select it
- Click on a highlighted square to move
- You play as White, AI plays as Black
- Pawns automatically promote to Queens

#### Training Your Own Model

1. **Prepare Training Data**: Place PGN files in the `raw_training_data/` directory

2. **Preprocess the Data**:
```bash
python data_preprocessing.py
```
This will:
- Parse PGN files
- Convert board positions to 14×8×8 arrays
- Create UCI move to integer mappings
- Save processed data to `cleaned_data/`

3. **Train the Model**:
```bash
python train.py
```
Training progress will display:
- Epoch number
- Average loss
- Training time per epoch

4. **Test Predictions**:
```bash
python predict.py
```
This runs an interactive console where you can play against the model using algebraic notation.

## 📊 Dataset

For now the model is trained on **Lichess Elite database** games:
- **Time Period**: June 2021 - November 2022
- **Game Quality**: Elite-level players only
- **Format**: PGN (Portable Game Notation)
- **Total Positions**: Several million board positions

You can adjust the training dataset size by modifying the `limit_of_games` parameter in `data_preprocessing.py`.

## 🔧 Technical Details

### Board Representation
The board state is encoded as a 14-channel tensor:
```python
Channel  0: White Pawns    Channel  6: Black Pawns
Channel  1: White Knights  Channel  7: Black Knights
Channel  2: White Bishops  Channel  8: Black Bishops
Channel  3: White Rooks    Channel  9: Black Rooks
Channel  4: White Queens   Channel 10: Black Queens
Channel  5: White King     Channel 11: Black King
Channel 12: Legal move destinations
Channel 13: Turn indicator (1=White, 0=Black)
```

### Move Prediction
The model outputs logits for all possible UCI moves. The prediction system:
1. Computes softmax probabilities over all moves
2. Sorts moves by probability (highest first)
3. Returns the highest-probability **legal** move
4. This ensures the AI never makes illegal moves



## 🐛 Known Limitations

- Model learns move patterns but lacks deep strategic understanding
- Opening repertoire limited to training data
- Endgame play may not be as strong as dedicated engines
- No explicit evaluation function (unlike traditional engines)
- Requires GPU for reasonable training time

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests
- Share your training results

## 📝 Future Improvements

Potential enhancements:
- [ ] Add reinforcement learning (self-play)
- [ ] Implement value network for position evaluation
- [ ] Model ensembling
- [ ] Web-based interface
- [ ] Move explanation/analysis
- [ ] Add more diverse training data focused on models weaknesses

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **python-chess**: Chess logic and move validation
- **NumPy**: Numerical computations
- **Pygame**: GUI rendering
- **tqdm**: Progress bars for training

See `requirements.txt` for complete list.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Lichess**: For providing free, high-quality chess game databases
- **python-chess**: For the excellent chess library
- **PyTorch**: For the deep learning framework

## 📧 Contact

For questions or feedback about this project, feel free to open an issue on GitHub.

---

**Note**: This is an educational project demonstrating neural network applications in game AI. It is not intended to compete with traditional chess engines like Stockfish or AlphaZero, although maybe someday we will reach that level ;)
