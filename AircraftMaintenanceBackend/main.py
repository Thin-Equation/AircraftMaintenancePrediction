import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import sys
import os
from fastapi.middleware.cors import CORSMiddleware  # Add this import

# Add the project root to the path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now we can import our modules
from api.routes import router as api_router
from api.routes import WebSocketManager

# Create FastAPI app
app = FastAPI(
    title="Aircraft Maintenance Prediction API",
    description="API for predicting aircraft engine failures",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include API routes
app.include_router(api_router)

# WebSocket connection manager
ws_manager = WebSocketManager()

# WebSocket endpoint for training notifications
@app.websocket("/ws/training")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and wait for messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to Aircraft Maintenance Prediction API",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)