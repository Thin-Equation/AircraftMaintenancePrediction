from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import torch
import numpy as np
import sys
import os
import asyncio
from datetime import datetime

# Add the project root to the path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our modules
from AircraftMaintenanceBackend.models.pytorch_models import get_model
from AircraftMaintenanceBackend.utils.data_processing import DataProcessor
from AircraftMaintenanceBackend.models.trainer import ModelTrainer, prepare_data_loaders

# WebSocket connection manager for training notifications
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Connection might be closed or invalid
                pass

# Global WebSocket manager instance to be used in the training process
ws_manager = WebSocketManager()

# Create API router
router = APIRouter(tags=["Aircraft Maintenance"])

# Define data models for API
class SensorReading(BaseModel):
    """Model for a single sensor reading"""
    id: Optional[int] = None
    cycle: int
    setting1: float
    setting2: float
    setting3: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    s7: float
    s8: float
    s9: float
    s10: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float
    s17: float
    s18: float
    s19: float
    s20: float
    s21: float


class PredictionRequest(BaseModel):
    """Model for prediction request"""
    engine_id: int
    readings: List[SensorReading]


class TrainingRequest(BaseModel):
    """Model for training request"""
    model_type: str = "lstm"  # Options: simple_rnn, lstm, birnn, gru
    epochs: int = 200
    batch_size: int = 200
    learning_rate: float = 0.001
    hidden_dim: int = 100
    num_layers: int = 2
    dropout: float = 0.2
    window_size: int = 30
    sequence_length: int = 50


class TrainingResponse(BaseModel):
    """Model for training response"""
    job_id: str
    status: str
    message: str


class PredictionResponse(BaseModel):
    """Model for prediction response"""
    engine_id: int
    prediction: int  # 0 or 1
    probability: float
    message: str


# Dictionary to track training jobs
training_jobs = {}

# Get data paths based on relative or absolute paths
def get_data_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, "PM_train.txt")
    test_path = os.path.join(base_dir, "PM_test.txt")
    truth_path = os.path.join(base_dir, "PM_truth.txt")
    
    # Ensure data files exist
    for path in [train_path, test_path, truth_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
    
    return train_path, test_path, truth_path

# Get model path for specific model type
def get_model_path(model_type):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, "app", "data")
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, f"{model_type.lower()}_model.pt")

# Get the data processor
def get_data_processor(window_size=30, sequence_length=50):
    train_path, test_path, truth_path = get_data_paths()
    return DataProcessor(
        train_path=train_path,
        test_path=test_path,
        truth_path=truth_path,
        window_size=window_size,
        sequence_length=sequence_length
    )

# Async function to send WebSocket notification
async def send_training_notification(job_id: str, status: str, message: str, results: Dict[str, Any] = None):
    notification = {
        "type": "training_update",
        "job_id": job_id,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }
    
    if results:
        notification["results"] = results
        
    await ws_manager.broadcast(notification)

# Background task for training models
def train_model_task(job_id: str, training_params: Dict[str, Any]):
    try:
        # Extract parameters
        model_type = training_params.get("model_type", "lstm")
        epochs = training_params.get("epochs", 200)
        batch_size = training_params.get("batch_size", 200)
        learning_rate = training_params.get("learning_rate", 0.001)
        hidden_dim = training_params.get("hidden_dim", 100)
        num_layers = training_params.get("num_layers", 2)
        dropout = training_params.get("dropout", 0.2)
        window_size = training_params.get("window_size", 30)
        sequence_length = training_params.get("sequence_length", 50)
        
        # Update job status
        training_jobs[job_id]["status"] = "processing"
        training_jobs[job_id]["message"] = "Data processing started"
        
        # Send notification for processing start
        asyncio.run(send_training_notification(job_id, "processing", "Data processing started"))
        
        # Process data
        data_processor = get_data_processor(window_size, sequence_length)
        train_sequences, train_labels, test_sequences, test_labels = data_processor.run_full_processing()
        
        # Update job status
        training_jobs[job_id]["status"] = "training"
        training_jobs[job_id]["message"] = "Model training started"
        
        # Send notification for training start
        asyncio.run(send_training_notification(job_id, "training", "Model training started"))
        
        # Prepare data loaders
        loaders = prepare_data_loaders(
            train_sequences, train_labels,
            X_test=test_sequences, y_test=test_labels,
            batch_size=batch_size
        )
        
        # Create and train model
        input_dim = train_sequences.shape[2]
        model = get_model(
            model_name=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Setup trainer
        trainer = ModelTrainer(model)
        trainer.setup_optimizer(optimizer_type="adam", lr=learning_rate)
        trainer.setup_scheduler(scheduler_type="reduce_lr_on_plateau", patience=5)
        
        # Train the model
        model_path = get_model_path(model_type)
        history = trainer.train(
            loaders["train"], loaders["val"],
            epochs=epochs,
            early_stopping_patience=10,
            model_save_path=model_path
        )
        
        # Evaluate on test set
        test_results = trainer.evaluate(loaders["test"])
        
        # Update job status with results
        results = {
            "accuracy": float(test_results["accuracy"]),
            "precision": float(test_results["precision"]),
            "recall": float(test_results["recall"]),
            "f1_score": float(test_results["f1_score"]),
            "model_path": model_path
        }
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["message"] = "Training completed successfully"
        training_jobs[job_id]["results"] = results
        
        # Send notification for training completion with results
        asyncio.run(send_training_notification(
            job_id=job_id, 
            status="completed", 
            message="Training completed successfully",
            results=results
        ))
        
    except Exception as e:
        # Update job status with error
        error_message = f"Training failed: {str(e)}"
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = error_message
        
        # Send notification for training failure
        asyncio.run(send_training_notification(job_id, "failed", error_message))

# Endpoint for training a model
@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    # Generate a job ID
    job_id = f"train_{len(training_jobs) + 1}_{request.model_type}"
    
    # Initialize job status
    training_jobs[job_id] = {
        "status": "queued",
        "message": "Job queued for processing",
        "params": request.dict()
    }
    
    # Schedule training task in background
    background_tasks.add_task(train_model_task, job_id, request.dict())
    
    return TrainingResponse(
        job_id=job_id,
        status="queued",
        message="Model training job has been queued"
    )


# Endpoint to get training job status
@router.get("/train/{job_id}")
async def get_training_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    return training_jobs[job_id]


# Endpoint for prediction
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Check if we have readings
        if not request.readings:
            raise HTTPException(status_code=400, detail="No sensor readings provided")
        
        # Convert readings to list of dictionaries
        readings_data = [reading.dict() for reading in request.readings]
        
        # Get the model type to use - prefer LSTM by default
        model_type = "lstm"
        model_path = get_model_path(model_type)
        
        # Check if model exists, if not try other models
        if not os.path.exists(model_path):
            for model_name in ["gru", "birnn", "simple_rnn"]:
                potential_path = get_model_path(model_name)
                if os.path.exists(potential_path):
                    model_type = model_name
                    model_path = potential_path
                    break
        
        # If still no model, return error
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, 
                detail="No trained models found. Please train a model first."
            )
        
        # Load the data processor
        data_processor = get_data_processor()
        data_processor.load_data()
        data_processor.process_training_data()  # Need this to initialize scaler
        
        # Prepare input data
        input_tensor = data_processor.prepare_single_input(readings_data)
        
        # Determine input dimension
        input_dim = input_tensor.shape[2]
        
        # Load model
        model = get_model(model_type, input_dim=input_dim)
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Create trainer for prediction
        trainer = ModelTrainer(model, device="cpu")
        
        # Make prediction
        with torch.no_grad():
            model.eval()
            output = model(input_tensor)
            probability = output.item()
            prediction = 1 if probability > 0.5 else 0
        
        # Return result
        return PredictionResponse(
            engine_id=request.engine_id,
            prediction=prediction,
            probability=probability,
            message="Prediction successful" if prediction == 0 else 
                    "WARNING: Engine is predicted to fail within the next 30 cycles"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Endpoint to list available models
@router.get("/models")
async def list_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, "app", "data")
    
    if not os.path.exists(models_dir):
        return {"models": []}
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith("_model.pt"):
            model_type = file.replace("_model.pt", "")
            model_path = os.path.join(models_dir, file)
            models.append({
                "model_type": model_type,
                "path": model_path,
                "size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2)
            })
    
    return {"models": models}