import torch
import json
import pickle
from torch_model import NeuralKnightChessModel2

# === CONFIGURE THESE ===
MODEL_PATH = "models/10_epochs_model2.pth"  # Path to your trained .pt file
MAPPING_PATH = "models/model1_heavy_move_to_idx"  # Path to your move mapping pickle file
# =======================

# Load the move mapping to get num_classes
with open(MAPPING_PATH, "rb") as f:
    move_to_idx = pickle.load(f)

num_classes = len(move_to_idx)
print(f"Number of classes (unique moves): {num_classes}")

# Create idx_to_move mapping for the browser
idx_to_move = {str(idx): move for move, idx in move_to_idx.items()}

# Save as JSON for the browser
with open("idx_to_move.json", "w") as f:
    json.dump(idx_to_move, f)
print("Saved idx_to_move.json")

# Load the model
model = NeuralKnightChessModel2(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("Model loaded successfully")

# Create dummy input: batch=1, channels=14, height=8, width=8
dummy_input = torch.randn(1, 14, 8, 8)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11
)
print("Exported model.onnx")
print("\n✅ Done! Push these files to your repo:")
print("   - model.onnx")
print("   - idx_to_move.json")
print("   - index.html")
