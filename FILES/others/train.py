import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

digits = load_digits()          
X = digits.data / 16.0          
y = digits.target               

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("[*] Training MLP Perceptron (64 → 64 hidden → 10 outputs) …")
mlp = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    verbose=False,
)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[✓] Test Accuracy : {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, digits=2))

W1 = mlp.coefs_[0]        
b1 = mlp.intercepts_[0]   
W2 = mlp.coefs_[1]        
b2 = mlp.intercepts_[1]   

def to_c_array(name: str, arr: np.ndarray) -> str:
    flat = arr.flatten()
    lines = [f"const float {name}[{len(flat)}] PROGMEM = {{"]
    chunk: list[str] = []
    for v in flat:
        chunk.append(f"{v:.8f}f")
        if len(chunk) == 8:
            lines.append("    " + ", ".join(chunk) + ",")
            chunk = []
    if chunk:
        lines.append("    " + ", ".join(chunk))
    lines.append("};")
    return "\n".join(lines)

out_dir = os.path.join(
    os.path.dirname(__file__),
    "..", "esp32_firmware", "digit_classifier"
)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "model_weights.h")

header = f"""// model_weights.h  –  AUTO-GENERATED.  Do NOT edit by hand.
// Run pc_side/train_and_export.py to regenerate.
//
// Architecture : {W1.shape[0]} inputs → {W1.shape[1]} hidden (ReLU) → {W2.shape[1]} outputs
// Dataset      : sklearn digits (8×8 hand-written numerals, 1797 samples)
// Test accuracy: {acc * 100:.2f}%
// Total params : {W1.size + b1.size + W2.size + b2.size}  ({(W1.size + b1.size + W2.size + b2.size) * 4 / 1024:.1f} KB in PROGMEM)

#pragma once
#include <pgmspace.h>

#define INPUT_SIZE   {W1.shape[0]}
#define HIDDEN_SIZE  {W1.shape[1]}
#define OUTPUT_SIZE  {W2.shape[1]}

// W1  shape ({W1.shape[0]}, {W1.shape[1]})  – row-major
// index  =  input_i * HIDDEN_SIZE + hidden_j
{to_c_array("W1", W1)}

// b1  shape ({b1.shape[0]},)
{to_c_array("b1", b1)}

// W2  shape ({W2.shape[0]}, {W2.shape[1]})  – row-major
// index  =  hidden_i * OUTPUT_SIZE + output_j
{to_c_array("W2", W2)}

// b2  shape ({b2.shape[0]},)
{to_c_array("b2", b2)}
"""

with open(out_path, "w") as f:
    f.write(header)

print(f"[✓] Weights written to  {os.path.abspath(out_path)}")
print(f"    PROGMEM usage        ~{(W1.size + b1.size + W2.size + b2.size) * 4 / 1024:.1f} KB")
