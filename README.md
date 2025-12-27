# Project: Real-Time Driver Drowsiness Detection System

## Overview
The Real-Time Driver Drowsiness Detection System is designed to monitor drivers for signs of drowsiness using computer vision techniques. The system utilizes a hybrid architecture that combines MediaPipe Face Mesh for facial landmark detection and a custom Convolutional Neural Network (CNN) for eye state classification.

## Project Structure
The project is organized into the following directories and files:

```
driver-drowsiness-detection
├── data
│   ├── raw                # Directory for raw data files (currently empty)
│   ├── train
│   │   ├── open          # Directory for training images of open eyes (currently empty)
│   │   └── closed        # Directory for training images of closed eyes (currently empty)
├── src
│   ├── __init__.py       # Marks the directory as a Python package
│   ├── data_loader.py    # Contains the function to organize training data
│   ├── model_arch.py     # Defines the CNN architecture for eye state classification
│   └── detector.py       # Contains the DrowsinessDetector class for inference
├── assets
│   └── alarm.wav         # Placeholder for the audio alarm
├── models                # Directory for storing trained model files (currently empty)
├── train_model.py        # Script to train the CNN model
├── app.py                # Streamlit web application for real-time detection
├── requirements.txt      # Lists project dependencies
└── README.md             # Documentation for the project
```

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd driver-drowsiness-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation
Before training the model, ensure that the MRL dataset is downloaded and placed in the `mrl_dataset_raw/` directory. The dataset should contain the following structure:
```
mrl_dataset_raw/
├── Open-Eyes/            # Contains images of open eyes
└── Close-Eyes/           # Contains images of closed eyes
```
Run the `data_loader.py` script to organize the data:
```
python src/data_loader.py
```
This will randomly sample 2,000 images from each category and place them in the appropriate training directories.

## Training the Model
To train the CNN model, execute the following command:
```
python train_model.py
```
This will train the model using the organized data and save the trained model in the `models/` directory.

## Running the Application
To start the Streamlit web application, run:
```
streamlit run app.py
```
This will launch the application in your web browser, allowing you to monitor real-time video feed and receive alerts for drowsiness and yawning.

## Usage
- Adjust sensitivity thresholds in the sidebar.
- The application will display alerts when drowsiness is detected or when yawning occurs.
- An audio alarm will sound when a drowsy state is detected.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.