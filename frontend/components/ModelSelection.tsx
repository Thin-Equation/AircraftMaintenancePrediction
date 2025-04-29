'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { ChartIcon, SettingsIcon, CheckIcon } from './Icons';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type Model = {
  name: string;
  path: string;
  size_mb: number;
  modified: string;
};

type TrainingJob = {
  job_id: string;
  status: string;
  progress: number;
  message?: string;
  model_path?: string;
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
};

const ModelSelection = () => {
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'select' | 'train'>('select');
  
  // Training form state
  const [trainingConfig, setTrainingConfig] = useState({
    model_type: 'gru',
    window_size: 30,
    sequence_length: 50,
    epochs: 100,
    batch_size: 200,
    learning_rate: 0.001,
    use_single_feature: false,
    use_cuda: false
  });
  
  // Training job status
  const [currentTrainingJob, setCurrentTrainingJob] = useState<TrainingJob | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  // Fetch available models
  useEffect(() => {
    fetchModels();
  }, []);

  // Clean up polling interval
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  // Poll training job status
  useEffect(() => {
    if (currentTrainingJob && ['queued', 'preparing', 'preprocessing', 'preparing_sequences', 'building_model', 'training', 'evaluating'].includes(currentTrainingJob.status)) {
      const interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_URL}/api/train/status/${currentTrainingJob.job_id}`);
          setCurrentTrainingJob(response.data);
          
          if (['completed', 'error'].includes(response.data.status)) {
            clearInterval(interval);
            if (response.data.status === 'completed') {
              toast.success('Training completed successfully!');
              fetchModels(); // Refresh model list
            } else {
              toast.error(`Training error: ${response.data.message}`);
            }
          }
        } catch (error) {
          console.error('Error polling job status:', error);
          clearInterval(interval);
          toast.error('Failed to get training status');
        }
      }, 2000);
      
      setPollingInterval(interval);
      return () => clearInterval(interval);
    }
  }, [currentTrainingJob]);

  const fetchModels = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${API_URL}/api/models`);
      setAvailableModels(response.data.models || []);
      
      // Select the first model if none is selected
      if (response.data.models?.length > 0 && !selectedModel) {
        setSelectedModel(response.data.models[0].name);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
      toast.error('Failed to fetch available models');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrainingConfigChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target as HTMLInputElement;
    
    setTrainingConfig(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (e.target as HTMLInputElement).checked : 
              type === 'number' ? parseFloat(value) :
              value
    }));
  };

  const startTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      setIsLoading(true);
      const response = await axios.post(`${API_URL}/api/train`, trainingConfig);
      
      if (response.data.job_id) {
        setCurrentTrainingJob({
          job_id: response.data.job_id,
          status: response.data.status,
          progress: 0
        });
        toast.info('Training job started');
      }
    } catch (error) {
      console.error('Error starting training:', error);
      toast.error('Failed to start training job');
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-500';
      case 'error':
        return 'text-red-500';
      case 'training':
        return 'text-blue-500';
      default:
        return 'text-yellow-500';
    }
  };

  const getProgressStepClass = (step: string, currentStatus: string) => {
    const statusOrder = ['queued', 'preparing', 'preprocessing', 'preparing_sequences', 'building_model', 'training', 'evaluating', 'completed'];
    const stepIndex = statusOrder.indexOf(step);
    const currentIndex = statusOrder.indexOf(currentStatus);
    
    if (currentIndex >= stepIndex) {
      return 'bg-primary-500';
    } else {
      return 'bg-gray-300 dark:bg-gray-700';
    }
  };

  return (
    <div className="card mb-6">
      <h2 className="section-title">Model Selection</h2>
      
      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('select')}
            className={`py-2 px-1 ${
              activeTab === 'select'
                ? 'border-b-2 border-primary-500 font-medium text-primary-600 dark:text-primary-400'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            Select Model
          </button>
          <button
            onClick={() => setActiveTab('train')}
            className={`py-2 px-1 ${
              activeTab === 'train'
                ? 'border-b-2 border-primary-500 font-medium text-primary-600 dark:text-primary-400'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            Train New Model
          </button>
        </nav>
      </div>
      
      {/* Select Model Tab */}
      {activeTab === 'select' && (
        <div>
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Select a Pre-trained Model:
            </label>
            
            {isLoading ? (
              <div className="py-4 text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-500 mx-auto"></div>
                <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Loading models...</p>
              </div>
            ) : availableModels.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-800">
                    <tr>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Select
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Model Name
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Size
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Last Modified
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                    {availableModels.map((model) => (
                      <tr 
                        key={model.name} 
                        onClick={() => setSelectedModel(model.name)}
                        className={`cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 ${
                          selectedModel === model.name ? 'bg-primary-50 dark:bg-primary-900/20' : ''
                        }`}
                      >
                        <td className="px-6 py-4 whitespace-nowrap">
                          <input
                            type="radio"
                            name="selectedModel"
                            checked={selectedModel === model.name}
                            onChange={() => setSelectedModel(model.name)}
                            className="text-primary-600 focus:ring-primary-500"
                          />
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            {model.name}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {model.size_mb.toFixed(2)} MB
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {model.modified}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="py-4 text-center border border-dashed border-gray-300 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-800">
                <ChartIcon className="mx-auto text-gray-400 mb-2" width={24} height={24} />
                <p className="text-gray-500 dark:text-gray-400">No models available. Train a new model to get started.</p>
                <button 
                  onClick={() => setActiveTab('train')} 
                  className="mt-4 text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
                >
                  Train a new model
                </button>
              </div>
            )}
          </div>
          
          {selectedModel && (
            <div className="mt-6 bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Selected model: <span className="text-primary-600 dark:text-primary-400">{selectedModel}</span>
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                This model will be used for predictions.
              </p>
            </div>
          )}
        </div>
      )}
      
      {/* Train Model Tab */}
      {activeTab === 'train' && (
        <div>
          {currentTrainingJob && ['queued', 'preparing', 'preprocessing', 'preparing_sequences', 'building_model', 'training', 'evaluating'].includes(currentTrainingJob.status) ? (
            <div className="mb-6">
              <h3 className="font-medium text-lg mb-4">Training in Progress</h3>
              
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium">Progress:</span>
                  <span className={getStatusColor(currentTrainingJob.status)}>
                    {Math.round(currentTrainingJob.progress * 100)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                  <div
                    className="bg-primary-600 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${currentTrainingJob.progress * 100}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="mb-6">
                <p className="text-sm mb-2">
                  <span className={getStatusColor(currentTrainingJob.status)}>
                    Status: {currentTrainingJob.status}
                  </span>
                </p>
                {currentTrainingJob.message && (
                  <p className="text-sm text-gray-600 dark:text-gray-400">{currentTrainingJob.message}</p>
                )}
              </div>
              
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs text-gray-500 dark:text-gray-400">Queue</span>
                <span className="text-xs text-gray-500 dark:text-gray-400">Complete</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`h-2 w-2 rounded-full ${getProgressStepClass('queued', currentTrainingJob.status)}`}></div>
                <div className="h-0.5 flex-grow bg-gray-300 dark:bg-gray-700 relative">
                  <div 
                    className="absolute top-0 left-0 h-full bg-primary-500" 
                    style={{ width: `${currentTrainingJob.progress * 100}%` }}
                  ></div>
                </div>
                <div className={`h-2 w-2 rounded-full ${getProgressStepClass('completed', currentTrainingJob.status)}`}></div>
              </div>
              <div className="flex justify-between mt-2">
                <span className="text-xs text-gray-500 dark:text-gray-400">{currentTrainingJob.job_id}</span>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {new Date().toLocaleString()}
                </span>
              </div>
            </div>
          ) : currentTrainingJob?.status === 'completed' ? (
            <div className="mb-6 p-4 bg-green-50 dark:bg-green-900/10 border border-green-200 dark:border-green-800 rounded-lg">
              <div className="flex items-start">
                <CheckIcon className="text-green-500 mt-0.5 mr-3" width={20} height={20} />
                <div>
                  <h3 className="font-medium text-green-800 dark:text-green-300">Training Completed</h3>
                  <p className="text-sm text-green-700 dark:text-green-400 mt-1">
                    Model has been successfully trained and saved.
                  </p>
                  {currentTrainingJob.metrics && (
                    <div className="mt-3 grid grid-cols-2 gap-2">
                      <div className="text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Accuracy: </span>
                        <span className="font-medium">
                          {(currentTrainingJob.metrics.accuracy * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Precision: </span>
                        <span className="font-medium">
                          {(currentTrainingJob.metrics.precision * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Recall: </span>
                        <span className="font-medium">
                          {(currentTrainingJob.metrics.recall * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="text-sm">
                        <span className="text-gray-600 dark:text-gray-400">F1 Score: </span>
                        <span className="font-medium">
                          {(currentTrainingJob.metrics.f1_score * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  )}
                  <div className="mt-4">
                    <button
                      onClick={() => {
                        setCurrentTrainingJob(null);
                        fetchModels();
                      }}
                      className="text-sm text-white bg-green-600 hover:bg-green-700 py-1 px-3 rounded"
                    >
                      Train Another Model
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : currentTrainingJob?.status === 'error' ? (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-800 rounded-lg">
              <h3 className="font-medium text-red-800 dark:text-red-300">Training Error</h3>
              <p className="text-sm text-red-700 dark:text-red-400 mt-1">
                {currentTrainingJob.message || 'An error occurred during model training.'}
              </p>
              <button
                onClick={() => setCurrentTrainingJob(null)}
                className="mt-3 text-sm text-white bg-red-600 hover:bg-red-700 py-1 px-3 rounded"
              >
                Try Again
              </button>
            </div>
          ) : (
            <form onSubmit={startTraining}>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <div className="mb-4">
                    <label htmlFor="model_type" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Model Type
                    </label>
                    <select
                      id="model_type"
                      name="model_type"
                      value={trainingConfig.model_type}
                      onChange={handleTrainingConfigChange}
                      className="input-field dark:bg-gray-800 dark:text-white dark:border-gray-700"
                      required
                    >
                      <option value="rnn">RNN</option>
                      <option value="lstm">LSTM</option>
                      <option value="gru">GRU (Recommended)</option>
                    </select>
                  </div>
                  
                  <div className="mb-4">
                    <label htmlFor="window_size" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Window Size
                    </label>
                    <input
                      type="number"
                      id="window_size"
                      name="window_size"
                      value={trainingConfig.window_size}
                      onChange={handleTrainingConfigChange}
                      className="input-field dark:bg-gray-800 dark:text-white dark:border-gray-700"
                      min="5"
                      max="100"
                      required
                    />
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      Number of cycles to consider for failure prediction.
                    </p>
                  </div>
                  
                  <div className="mb-4">
                    <label htmlFor="sequence_length" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Sequence Length
                    </label>
                    <input
                      type="number"
                      id="sequence_length"
                      name="sequence_length"
                      value={trainingConfig.sequence_length}
                      onChange={handleTrainingConfigChange}
                      className="input-field dark:bg-gray-800 dark:text-white dark:border-gray-700"
                      min="5"
                      max="200"
                      required
                    />
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      Length of sequences fed into the model.
                    </p>
                  </div>
                  
                  <div className="mb-4">
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="use_single_feature"
                        name="use_single_feature"
                        checked={trainingConfig.use_single_feature}
                        onChange={handleTrainingConfigChange}
                        className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded dark:border-gray-700 dark:bg-gray-800"
                      />
                      <label htmlFor="use_single_feature" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                        Use Single Feature (s2)
                      </label>
                    </div>
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400 ml-6">
                      When enabled, only sensor 2 data will be used for prediction.
                    </p>
                  </div>
                  
                  <div className="mb-4">
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="use_cuda"
                        name="use_cuda"
                        checked={trainingConfig.use_cuda}
                        onChange={handleTrainingConfigChange}
                        className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded dark:border-gray-700 dark:bg-gray-800"
                      />
                      <label htmlFor="use_cuda" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                        Use GPU (CUDA) if available
                      </label>
                    </div>
                  </div>
                </div>
                
                <div>
                  <div className="flex items-center mb-4">
                    <SettingsIcon className="text-gray-500 mr-2" width={20} height={20} />
                    <h3 className="font-medium text-gray-700 dark:text-gray-300">
                      Advanced Settings
                    </h3>
                  </div>
                  
                  <div className="mb-4">
                    <label htmlFor="epochs" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Training Epochs
                    </label>
                    <input
                      type="number"
                      id="epochs"
                      name="epochs"
                      value={trainingConfig.epochs}
                      onChange={handleTrainingConfigChange}
                      className="input-field dark:bg-gray-800 dark:text-white dark:border-gray-700"
                      min="5"
                      max="500"
                      required
                    />
                  </div>
                  
                  <div className="mb-4">
                    <label htmlFor="batch_size" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Batch Size
                    </label>
                    <input
                      type="number"
                      id="batch_size"
                      name="batch_size"
                      value={trainingConfig.batch_size}
                      onChange={handleTrainingConfigChange}
                      className="input-field dark:bg-gray-800 dark:text-white dark:border-gray-700"
                      min="16"
                      max="1024"
                      step="16"
                      required
                    />
                  </div>
                  
                  <div className="mb-4">
                    <label htmlFor="learning_rate" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Learning Rate
                    </label>
                    <input
                      type="number"
                      id="learning_rate"
                      name="learning_rate"
                      value={trainingConfig.learning_rate}
                      onChange={handleTrainingConfigChange}
                      className="input-field dark:bg-gray-800 dark:text-white dark:border-gray-700"
                      min="0.0001"
                      max="0.1"
                      step="0.0001"
                      required
                    />
                  </div>
                  
                  <div className="mt-8 bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
                    <p className="text-sm text-yellow-700 dark:text-yellow-400">
                      <strong>Note:</strong> Training can take several minutes depending on the complexity of the model and dataset size.
                      The training process includes early stopping to prevent overfitting.
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="mt-6 flex justify-end">
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <span className="inline-block animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full mr-2 align-middle"></span>
                      <span>Processing...</span>
                    </>
                  ) : (
                    'Start Training'
                  )}
                </button>
              </div>
            </form>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelSelection;