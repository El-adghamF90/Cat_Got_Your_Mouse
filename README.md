# Cat Got Your Mouse 🐱🐭

## Hand Gesture Recognition and Mouse Control Using Machine Learning

Cat Got Your Mouse is a Python project that enables users to control their computer mouse using hand gestures captured by a webcam.

### Features

- Cursor movement using hand gestures
- Left and right mouse clicks
- Scrolling and drag-and-drop
- Volume control through hand movement
- Play/pause control using a fist gesture

### How It Works

1. MediaPipe detects 21 hand landmarks from webcam frames.
2. The system extracts and normalizes the X and Y coordinates, producing 42 numerical features.
3. A trained machine-learning model classifies the hand gesture.
4. PyAutoGUI translates the recognized gesture into a computer action.

### Machine Learning

The project compares Random Forest and Multi-Layer Perceptron (MLP) classifiers. Model selection is based on validation performance, followed by evaluation on a separate test set.

### Technologies

- Python
- OpenCV
- MediaPipe
- NumPy and Pandas
- Scikit-learn
- Joblib
- PyAutoGUI
- Pycaw and Comtypes

### Installation

Clone the repository:

```bash
git clone https://github.com/El-adghamF90/Cat_Got_Your_Mouse.git
cd Cat_Got_Your_Mouse
```

Create and activate a virtual environment on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

Train and evaluate the models:

```bash
python train.py
```

Run the hand-controlled mouse:

```bash
python mouse.py
```

Ensure the annotation data and trained model are available at the paths configured in `common.py`.

### Dataset Acknowledgment

The project references the HaGRID hand gesture recognition dataset:

https://github.com/hukenovs/hagrid

Refer to the original repository for its dataset license and attribution requirements. Only redistribute dataset materials when permitted by their applicable terms.

### Contributors

- [El-adghamF90](https://github.com/El-adghamF90)
