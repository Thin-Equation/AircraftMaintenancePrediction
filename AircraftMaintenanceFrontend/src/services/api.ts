import axios from 'axios';
import {
  PredictionRequest,
  PredictionResponse,
  TrainingRequest,
  TrainingResponse,
  TrainingStatus,
  ModelsResponse
} from '@/types';

// Create an axios instance with base URL and default headers
const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// API service methods
export const ApiService = {
  // Training related endpoints
  trainModel: async (trainingRequest: TrainingRequest): Promise<TrainingResponse> => {
    const response = await api.post('/train', trainingRequest);
    return response.data;
  },
  
  getTrainingStatus: async (jobId: string): Promise<TrainingStatus> => {
    const response = await api.get(`/train/${jobId}`);
    return response.data;
  },
  
  // Prediction endpoint
  predict: async (predictionRequest: PredictionRequest): Promise<PredictionResponse> => {
    const response = await api.post('/predict', predictionRequest);
    return response.data;
  },
  
  // Models management
  getModels: async (): Promise<ModelsResponse> => {
    const response = await api.get('/models');
    return response.data;
  },
  
  // Helper method to check if API is available
  healthCheck: async (): Promise<boolean> => {
    try {
      await api.get('/');
      return true;
    } catch {
      // Error ignored intentionally
      return false;
    }
  }
};

export default ApiService;