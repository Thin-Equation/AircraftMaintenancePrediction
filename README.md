# Aircraft Maintenance Prediction

This project provides a machine learning solution for predicting aircraft engine failures using recurrent neural network (RNN) architectures. 
It can help maintenance teams predict when an aircraft engine will require maintenance within a specified time window.

## Project Overview

Predictive maintenance is a critical area in the aviation industry where unexpected failures can lead to significant safety risks and financial losses. 
This project implements several deep learning models (Simple RNN, Bidirectional RNN, LSTM, and GRU) to predict if an aircraft engine will fail within a particular time window based on historical sensor measurements.

### Problem Statement

Given a sequence of sensor measurements and operational settings from aircraft engines, predict whether the engine will require maintenance within a specified cycle window (default 30 cycles).

### Dataset

The project uses a dataset with the following files:
- `PM_train.txt`: Training data with engine run-to-failure data
- `PM_test.txt`: Testing data without failure points
- `PM_truth.txt`: Ground truth for the test data (remaining useful life)

Each data file contains multiple sensor readings from aircraft engines. Each engine has a different number of operational cycles before it fails.

## Project Structure

```
AircraftMaintenancePrediction/
├── README.md
├── .gitignore                                          # Git ignore file
├── PM_train.txt                                        # Training dataset
├── PM_test.txt                                         # Testing dataset
├── PM_truth.txt                                        # Ground truth data
├── AircraftMaintenanceBackend/                         # FastAPI backend service
│   ├── __init__.py
│   ├── main.py                                         # Main application entry point
│   ├── requirements.txt                                # Python dependencies
│   ├── api/                                            # API endpoints and routes
│   ├── data/                                           # Trained model files
│   ├── models/                                         # PyTorch model implementations
│   └── utils/                                          # Data processing utilities
└── AircraftMaintenanceFrontend/                        # Next.js frontend application
    ├── next.config.js                                  # Next.js configuration
    ├── package.json                                    # Node.js dependencies
    ├── tsconfig.json                                   # TypeScript configuration
    ├── public/                                         # Static assets
    └── src/                                            # Source code
        ├── app/                                        # Next.js app router
        │   ├── predict/                                # Prediction page
        │   └── train/                                  # Model training page
        ├── services/                                   # API and WebSocket services
        └── types/                                      # TypeScript type definitions
```

## Models

The project implements four different neural network architectures:

1. **Simple RNN**: A basic RNN for sequential data processing
2. **Bidirectional RNN**: Processes sequences in both directions to capture more context
3. **LSTM (Long Short-Term Memory)**: Better captures long-term dependencies in the data
4. **GRU (Gated Recurrent Unit)**: A more efficient alternative to LSTM

## Web Application

This project includes a full-stack web application for easy interaction with the predictive maintenance system:

### Backend (FastAPI)

The backend provides RESTful API endpoints for:
- Dataset upload and management
- Model training and selection
- Prediction generation
- Results visualization

To run the backend server:

```bash
cd AircraftMaintenanceBackend
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs

### Frontend (Next.js)

The frontend provides an intuitive user interface featuring:
- Dashboard with dataset statistics
- Data upload functionality
- Model selection and configuration
- Visualization of prediction results

To run the frontend:

```bash
cd AircraftMaintenanceFrontend
npm install
npm run dev
```

The web application will be available at http://localhost:3000

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Thin-Equation/AircraftMaintenancePrediction.git
   cd AircraftMaintenancePrediction
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install backend dependencies:
   ```
   cd AircraftMaintenanceBackend
   pip install -r requirements.txt
   ```

4. Install frontend dependencies:
   ```
   cd ../AircraftMaintenanceFrontend
   npm install
   ```

## Usage

### Using the Web Interface

1. Start the backend server:
   ```bash
   cd AircraftMaintenanceBackend
   uvicorn main:app --reload
   ```

2. Start the frontend development server:
   ```bash
   cd AircraftMaintenanceFrontend
   npm run dev
   ```

3. Open your browser and navigate to http://localhost:3000

4. Use the interface to:
   - Upload training, testing, and truth datasets
   - Select and configure models
   - Train models and generate predictions
   - Visualize results

## Recent Updates

- Fixed WebSocket callback handling in the training page
- Added proper TypeScript type support for training status updates
- Improved error handling and data validation
- Updated project structure documentation

## Model Performance

The models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

LSTM and GRU models generally achieve the best performance due to their ability to capture long-term dependencies in the sensor data.

## Customization

- **Window Size**: Change the window size to adjust how many cycles in advance to predict failure
- **Sequence Length**: Adjust the sequence length to modify the input sequence length
- **Feature Selection**: Select specific sensors for prediction

## Contributing

Contributions to improve the project are welcome! Please feel free to submit a Pull Request.
