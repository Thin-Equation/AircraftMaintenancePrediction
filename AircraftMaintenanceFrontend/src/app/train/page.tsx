'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import ApiService from '@/services/api';
import wsService from '@/services/websocket';
import { TrainingRequest, TrainingStatus, TrainingNotification } from '@/types';

export default function TrainPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<TrainingStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  // Initialize form with React Hook Form
  const { register, handleSubmit, formState: { errors }, watch } = useForm<TrainingRequest>({
    defaultValues: {
      model_type: 'lstm',
      epochs: 200,
      batch_size: 200,
      learning_rate: 0.001,
      hidden_dim: 100,
      num_layers: 2,
      dropout: 0.2,
      window_size: 30,
      sequence_length: 50
    }
  });

  const selectedModelType = watch('model_type');

  // Connect to WebSocket and setup notification listener
  useEffect(() => {
    // Connect to WebSocket
    wsService.connect().catch(error => {
      console.error('WebSocket connection error:', error);
    });
    
    // Setup listener
    const handleTrainingNotification = (notification: TrainingNotification) => {
      if (notification.job_id === jobId) {
        setJobStatus({
          status: notification.status,
          message: notification.message,
          results: notification.results,
          params: {} as TrainingRequest // Add missing params field
        });
      }
    };
    
    wsService.onTrainingUpdate(handleTrainingNotification);
    
    return () => {
      wsService.removeCallback(handleTrainingNotification);
    };
  }, [jobId]);

  // Poll for job status updates
  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const status = await ApiService.getTrainingStatus(jobId);
        // Make sure we're passing the complete status object, 
        // including the params if they exist or an empty params object
        setJobStatus(status.params ? status : {
          ...status,
          params: {} as TrainingRequest
        });
        
        // Stop polling if job is complete or failed
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(interval);
          setPollingInterval(null);
        }
      } catch (err) {
        console.error('Error fetching job status:', err);
        // Keep polling even if there's an error
      }
    }, 5000); // Poll every 5 seconds

    setPollingInterval(interval);
    
    return () => {
      clearInterval(interval);
      setPollingInterval(null);
    };
  }, [jobId]);

  // Handle form submission
  const onSubmit = async (data: TrainingRequest) => {
    setLoading(true);
    setError(null);
    setJobId(null);
    setJobStatus(null);

    try {
      const response = await ApiService.trainModel(data);
      setJobId(response.job_id);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error 
        ? err.message 
        : typeof err === 'object' && err !== null && 'response' in err 
          ? (err.response as { data?: { detail?: string } })?.data?.detail || 'An error occurred during training initialization'
          : 'An error occurred during training initialization';
      
      setError(errorMessage);
      console.error('Training error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Train New Model</h1>
        <button 
          onClick={() => router.push('/')}
          className="bg-gray-200 px-4 py-2 rounded hover:bg-gray-300"
        >
          Back to Dashboard
        </button>
      </div>
      
      {/* Job Status */}
      {jobStatus && (
        <div className={`p-4 rounded-lg mb-6 ${
          jobStatus.status === 'failed' 
            ? 'bg-red-100 border border-red-400' 
            : jobStatus.status === 'completed'
            ? 'bg-green-100 border border-green-400'
            : 'bg-blue-100 border border-blue-400'
        }`}>
          <h2 className="text-xl font-bold mb-2">Training Job Status</h2>
          <p className="mb-1">
            <span className="font-semibold">Job ID:</span> {jobId}
          </p>
          <p className="mb-1">
            <span className="font-semibold">Status:</span> {jobStatus.status}
          </p>
          <p className="mb-1">
            <span className="font-semibold">Message:</span> {jobStatus.message}
          </p>
          
          {/* Show results if training is complete */}
          {jobStatus.status === 'completed' && jobStatus.results && (
            <div className="mt-4">
              <h3 className="text-lg font-semibold mb-2">Training Results</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white p-3 rounded shadow-sm">
                  <p className="text-sm font-medium text-gray-500">Accuracy</p>
                  <p className="text-2xl font-bold">{(jobStatus.results.accuracy * 100).toFixed(2)}%</p>
                </div>
                <div className="bg-white p-3 rounded shadow-sm">
                  <p className="text-sm font-medium text-gray-500">Precision</p>
                  <p className="text-2xl font-bold">{(jobStatus.results.precision * 100).toFixed(2)}%</p>
                </div>
                <div className="bg-white p-3 rounded shadow-sm">
                  <p className="text-sm font-medium text-gray-500">Recall</p>
                  <p className="text-2xl font-bold">{(jobStatus.results.recall * 100).toFixed(2)}%</p>
                </div>
                <div className="bg-white p-3 rounded shadow-sm">
                  <p className="text-sm font-medium text-gray-500">F1 Score</p>
                  <p className="text-2xl font-bold">{(jobStatus.results.f1_score * 100).toFixed(2)}%</p>
                </div>
              </div>
              <div className="mt-4 flex justify-center">
                <button
                  onClick={() => router.push('/predict')}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg"
                >
                  Make Predictions with This Model
                </button>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Error message */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 p-4 rounded-lg mb-6">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      )}

      <div className="bg-white p-6 rounded-lg shadow-md">
        {/* Model Description */}
        <div className="mb-6 p-4 bg-blue-50 rounded">
          <h3 className="font-bold text-lg mb-2">About {selectedModelType.toUpperCase()} Model</h3>
          {selectedModelType === 'simple_rnn' && (
            <p>Simple RNN is a basic recurrent neural network model suitable for sequential data. It has limited capacity to capture long-term dependencies but is faster to train.</p>
          )}
          {selectedModelType === 'lstm' && (
            <p>Long Short-Term Memory (LSTM) networks are designed to overcome the vanishing gradient problem and can capture long-term dependencies in sequential data, making them excellent for time-series prediction.</p>
          )}
          {selectedModelType === 'birnn' && (
            <p>Bidirectional RNN processes sequential data in both forward and backward directions, allowing it to capture context from both past and future states for better prediction accuracy.</p>
          )}
          {selectedModelType === 'gru' && (
            <p>Gated Recurrent Unit (GRU) is similar to LSTM but with a simplified architecture. It uses update and reset gates to solve the vanishing gradient problem with fewer parameters than LSTM.</p>
          )}
        </div>

        <form onSubmit={handleSubmit(onSubmit)}>
          {/* Model Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model Type
            </label>
            <select
              {...register('model_type')}
              className="w-full p-2 border rounded"
            >
              <option value="simple_rnn">Simple RNN</option>
              <option value="lstm">LSTM</option>
              <option value="birnn">Bidirectional RNN</option>
              <option value="gru">GRU</option>
            </select>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Training Parameters */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Epochs
              </label>
              <input
                type="number"
                {...register('epochs', { required: true, min: 1, max: 1000 })}
                className="w-full p-2 border rounded"
              />
              {errors.epochs && <span className="text-red-500">Epochs must be between 1-1000</span>}
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Batch Size
              </label>
              <input
                type="number"
                {...register('batch_size', { required: true, min: 1 })}
                className="w-full p-2 border rounded"
              />
              {errors.batch_size && <span className="text-red-500">Required</span>}
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.0001"
                {...register('learning_rate', { required: true, min: 0.0001, max: 1 })}
                className="w-full p-2 border rounded"
              />
              {errors.learning_rate && <span className="text-red-500">Invalid learning rate</span>}
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Hidden Dimension
              </label>
              <input
                type="number"
                {...register('hidden_dim', { required: true, min: 1 })}
                className="w-full p-2 border rounded"
              />
              {errors.hidden_dim && <span className="text-red-500">Required</span>}
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Number of Layers
              </label>
              <input
                type="number"
                {...register('num_layers', { required: true, min: 1, max: 5 })}
                className="w-full p-2 border rounded"
              />
              {errors.num_layers && <span className="text-red-500">Must be between 1-5</span>}
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Dropout Rate
              </label>
              <input
                type="number"
                step="0.1"
                {...register('dropout', { required: true, min: 0, max: 0.9 })}
                className="w-full p-2 border rounded"
              />
              {errors.dropout && <span className="text-red-500">Must be between 0-0.9</span>}
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Window Size (cycles)
              </label>
              <input
                type="number"
                {...register('window_size', { required: true, min: 1 })}
                className="w-full p-2 border rounded"
              />
              <p className="text-sm text-gray-500 mt-1">
                Number of cycles before failure to consider for prediction
              </p>
              {errors.window_size && <span className="text-red-500">Required</span>}
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Sequence Length
              </label>
              <input
                type="number"
                {...register('sequence_length', { required: true, min: 5 })}
                className="w-full p-2 border rounded"
              />
              <p className="text-sm text-gray-500 mt-1">
                Number of time steps to use for prediction
              </p>
              {errors.sequence_length && <span className="text-red-500">Must be at least 5</span>}
            </div>
          </div>

          {/* Submit button */}
          <div className="flex justify-end mt-6">
            <button
              type="submit"
              disabled={loading || jobId !== null}
              className={`px-6 py-2 rounded-lg text-white font-medium ${
                loading || jobId !== null ? 'bg-purple-400' : 'bg-purple-600 hover:bg-purple-700'
              }`}
            >
              {loading ? 'Submitting...' : jobId ? 'Training in Progress' : 'Train Model'}
            </button>
          </div>
        </form>
      </div>

      {/* Training Information */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Training Information</h2>
        <div className="space-y-4 text-gray-600">
          <p>
            Training a model will process the historical aircraft engine data and use it to learn patterns that predict 
            upcoming engine failures. The process runs on the server and may take several minutes depending on your 
            selected parameters.
          </p>
          
          <div>
            <h3 className="font-medium text-blue-800 mb-1">Available Models:</h3>
            <ul className="list-disc list-inside pl-4 space-y-1">
              <li><span className="font-medium">Simple RNN:</span> Basic recurrent neural network model</li>
              <li><span className="font-medium">LSTM:</span> Long Short-Term Memory network for capturing long dependencies</li>
              <li><span className="font-medium">BiRNN:</span> Bidirectional RNN that processes data in both directions</li>
              <li><span className="font-medium">GRU:</span> Gated Recurrent Unit, similar to LSTM but with simpler architecture</li>
            </ul>
          </div>
          
          <p>
            Once training is complete, the model will be available for making predictions through the prediction interface.
          </p>
        </div>
      </div>
    </div>
  );
}