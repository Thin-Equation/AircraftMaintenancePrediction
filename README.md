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
├── requirements.txt
├── aircraft-predictive-maintenance-rnn-lstm-gru.ipynb  # Jupyter notebook with analysis
├── PM_train.txt                                        # Training dataset
├── PM_test.txt                                         # Testing dataset
├── PM_truth.txt                                        # Ground truth data
├── backend/                                            # FastAPI backend service
│   ├── data_processing/                                # Data loading and preprocessing
│   ├── models/                                         # PyTorch model implementations
│   ├── utils/                                          # Evaluation and training utilities
│   ├── visualization/                                  # Plotting and visualization tools
│   ├── scripts/                                        # Training and evaluation scripts
│   ├── uploads/                                        # Directory for uploaded datasets
│   └── results/                                        # Model results and outputs
└── frontend/                                           # Next.js frontend application
    ├── app/                                            # Next.js app router
    ├── components/                                     # React components
    ├── public/                                         # Static assets
    └── types/                                          # TypeScript type definitions
```

## Models

The project implements five different neural network architectures:

1. **Simple RNN with Single Feature**: A basic RNN using only one sensor feature
2. **Simple RNN with Multiple Features**: Uses 25 features (sensor readings and settings)
3. **Bidirectional RNN**: Processes sequences in both directions to capture more context
4. **LSTM (Long Short-Term Memory)**: Better captures long-term dependencies in the data
5. **GRU (Gated Recurrent Unit)**: A more efficient alternative to LSTM

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
cd backend
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
cd frontend
npm install
npm run dev
```

The web application will be available at http://localhost:3000

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/AircraftMaintenancePrediction.git
   cd AircraftMaintenancePrediction
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up the web application (optional):
   ```bash
   # Install backend dependencies
   cd backend
   pip install -r requirements.txt
   
   # Install frontend dependencies
   cd ../frontend
   npm install
   ```

## Usage

### Using the Web Interface

1. Start the backend server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm run dev
   ```

3. Open your browser and navigate to http://localhost:3000

4. Use the interface to:
   - Upload training, testing, and truth datasets
   - Select and configure models
   - Train models and generate predictions
   - Visualize results

### Using Scripts

#### Training a Model

To train a model, use the training script:

```bash
cd backend
python scripts/train_pytorch.py
```

#### Evaluating a Model

To evaluate a trained model, use the evaluation script:

```bash
cd backend
python scripts/evaluate_pytorch.py
```

#### Making Predictions

To make predictions with a trained model, use the prediction script:

```bash
cd backend
python scripts/predict_pytorch.py
```

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project is based on the NASA Turbofan Engine Degradation Simulation Dataset.

## Contact

If you have any questions or feedback, please open an issue in the GitHub repository.