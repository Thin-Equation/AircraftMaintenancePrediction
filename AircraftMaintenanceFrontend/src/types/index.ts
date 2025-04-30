// API response interfaces
export interface SensorReading {
  id?: number;
  cycle: number;
  setting1: number;
  setting2: number;
  setting3: number;
  s1: number;
  s2: number;
  s3: number;
  s4: number;
  s5: number;
  s6: number;
  s7: number;
  s8: number;
  s9: number;
  s10: number;
  s11: number;
  s12: number;
  s13: number;
  s14: number;
  s15: number;
  s16: number;
  s17: number;
  s18: number;
  s19: number;
  s20: number;
  s21: number;
}

export interface PredictionRequest {
  engine_id: number;
  readings: SensorReading[];
}

export interface PredictionResponse {
  engine_id: number;
  prediction: number; // 0 or 1
  probability: number;
  message: string;
}

export interface TrainingRequest {
  model_type: 'simple_rnn' | 'lstm' | 'birnn' | 'gru';
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  hidden_dim?: number;
  num_layers?: number;
  dropout?: number;
  window_size?: number;
  sequence_length?: number;
}

export interface TrainingResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface TrainingStatus {
  status: 'queued' | 'processing' | 'training' | 'completed' | 'failed';
  message: string;
  params: TrainingRequest;
  results?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    model_path: string;
  };
}

export interface TrainingNotification {
  type: 'training_update';
  job_id: string;
  status: 'queued' | 'processing' | 'training' | 'completed' | 'failed';
  message: string;
  timestamp: string;
  results?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    model_path: string;
  };
}

export interface Model {
  model_type: string;
  path: string;
  size_mb: number;
}

export interface ModelsResponse {
  models: Model[];
}