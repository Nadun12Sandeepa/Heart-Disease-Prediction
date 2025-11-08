import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# ✅ Load your trained model
mod = joblib.load("h.pickle")



# ✅ Feature names
FEATURE_NAMES = ['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']

# ✅ Example default raw inputs (unscaled)
DEFAULT_INPUT = [45, 1, 10, 200, 120, 90]

def predict():
    try:
        # Get values from entry boxes
        values = [float(entry.get()) for entry in entries]
        arr = np.array(values).reshape(1, -1)

        # ✅ Scale input values before prediction
        arr_scaled = preprocessing.StandardScaler().fit(arr).transform(X=arr)

        # Predict with scaled values
        result = mod.predict(arr_scaled)
        prediction_label.config(text=f"Prediction: {result[0]}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def clear_inputs():
    for entry in entries:
        entry.delete(0, tk.END)

# ✅ Tkinter UI
root = tk.Tk()
root.title("Heart Disease Prediction")
root.geometry("400x320")

entries = []

tk.Label(root, text="Enter Patient Features:", font=("Arial", 12, "bold")).pack(pady=5)

frame = tk.Frame(root)
frame.pack(pady=5)

# Create labeled entry boxes
for i, feature in enumerate(FEATURE_NAMES):
    tk.Label(frame, text=feature + ":", width=12, anchor="w").grid(row=i, column=0, padx=5, pady=3)
    entry = tk.Entry(frame, width=15)
    entry.grid(row=i, column=1, padx=5, pady=3)
    entry.insert(0, str(DEFAULT_INPUT[i]))  # preload with defaults
    entries.append(entry)

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Predict", command=predict).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Clear", command=clear_inputs).grid(row=0, column=1, padx=10)

# Prediction label
prediction_label = tk.Label(root, text="Prediction: -", font=("Arial", 12, "bold"))
prediction_label.pack(pady=10)

root.mainloop()
