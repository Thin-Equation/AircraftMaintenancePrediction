'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js';
import { Pie, Bar } from 'react-chartjs-2';
import { DownloadIcon, SearchIcon, AlertCircleIcon, CheckIcon } from './Icons';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface EngineResult {
  engine_id: string;
  probability: number;
  failure_predicted: boolean;
  interpretation: string;
}

interface PredictionResults {
  job_id: string;
  model_name: string;
  model_type: string;
  total_engines: number;
  maintenance_required: number;
  timestamp: string;
  accuracy?: number;
  engines: EngineResult[];
}

const PredictionResults: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<PredictionResults | null>(null);
  const [predictionJobs, setPredictionJobs] = useState<string[]>([]);
  const [selectedJob, setSelectedJob] = useState<string>('');
  
  // Fetch available prediction jobs when component mounts
  useEffect(() => {
    fetchPredictionJobs();
  }, []);

  // Fetch prediction job results when a job is selected
  useEffect(() => {
    if (selectedJob) {
      fetchPredictionResults(selectedJob);
    }
  }, [selectedJob]);

  const fetchPredictionJobs = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${API_URL}/api/predict/jobs`);
      
      if (response.data.jobs && response.data.jobs.length > 0) {
        setPredictionJobs(response.data.jobs);
        // Auto-select the most recent job
        setSelectedJob(response.data.jobs[0]);
      }
    } catch (error) {
      console.error('Error fetching prediction jobs:', error);
      toast.error('Failed to load prediction jobs');
    } finally {
      setIsLoading(false);
    }
  };
  
  const fetchPredictionResults = async (jobId: string) => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${API_URL}/api/predict/results/${jobId}`);
      setResults(response.data);
    } catch (error) {
      console.error('Error fetching prediction results:', error);
      toast.error('Failed to load prediction results');
    } finally {
      setIsLoading(false);
    }
  };
  
  const exportToCsv = async () => {
    if (!results || !selectedJob) return;
    
    try {
      const response = await axios.get(`${API_URL}/api/predict/export/${selectedJob}`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `prediction_${selectedJob}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success('Results exported successfully');
    } catch (error) {
      console.error('Error exporting results:', error);
      toast.error('Failed to export results');
    }
  };
  
  if (isLoading && !results) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    );
  }
  
  if (!results && predictionJobs.length === 0) {
    return (
      <div className="card p-6 text-center">
        <h2 className="text-xl font-semibold mb-3">No Prediction Results</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          No prediction results are available. Make a prediction first to see results here.
        </p>
      </div>
    );
  }
  
  if (!results) return null;

  // Calculate some stats that aren't provided directly by the API
  const totalEngines = results.engines.length;
  const maintenanceRequired = results.maintenance_required;
  const noMaintenanceRequired = totalEngines - maintenanceRequired;
  
  // Prepare data for pie chart
  const pieData = {
    labels: ['Maintenance Required', 'No Maintenance Required'],
    datasets: [
      {
        data: [maintenanceRequired, noMaintenanceRequired],
        backgroundColor: ['#f56565', '#68d391'],
        borderColor: ['#e53e3e', '#38a169'],
        borderWidth: 1,
      },
    ],
  };
  
  // Prepare data for bar chart - limit to first 20 engines for clarity
  const engineIds = results.engines.slice(0, 20).map(engine => engine.engine_id);
  const probabilities = results.engines.slice(0, 20).map(engine => engine.probability);
  
  const barData = {
    labels: engineIds,
    datasets: [
      {
        label: 'Failure Probability',
        data: probabilities,
        backgroundColor: probabilities.map(p => 
          p > 0.7 ? '#f56565' : p > 0.4 ? '#f6ad55' : '#68d391'
        ),
      },
    ],
  };
  
  // Filter engines based on search term and filter status
  const filteredEngines = results.engines.filter(engine => {
    const matchesSearch = engine.engine_id.toString().toLowerCase().includes(searchTerm.toLowerCase());
    
    if (filterStatus === 'maintenance') {
      return matchesSearch && engine.failure_predicted;
    } else if (filterStatus === 'no-maintenance') {
      return matchesSearch && !engine.failure_predicted;
    }
    
    return matchesSearch;
  });

  // Format date from timestamp
  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getModelName = (modelType: string) => {
    const modelNames: {[key: string]: string} = {
      'simple_rnn': 'Simple RNN',
      'rnn': 'RNN',
      'multi_rnn': 'Multi-feature RNN',
      'bidirectional': 'Bidirectional RNN',
      'lstm': 'LSTM',
      'gru': 'GRU'
    };
    
    return modelNames[modelType] || modelType;
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <h2 className="text-xl font-semibold">Prediction Results</h2>
        
        {predictionJobs.length > 0 && (
          <div className="flex items-center">
            <label htmlFor="predictionJob" className="mr-2 text-sm font-medium text-gray-700 dark:text-gray-300">
              Prediction:
            </label>
            <select
              id="predictionJob"
              value={selectedJob}
              onChange={(e) => setSelectedJob(e.target.value)}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm dark:border-gray-700 dark:bg-gray-800 dark:text-white"
            >
              {predictionJobs.map(job => (
                <option key={job} value={job}>
                  {job}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
      
      <div>
        <p className="text-gray-600 dark:text-gray-300">
          Analysis completed on {formatDate(results.timestamp)} using {getModelName(results.model_type)} model
          {results.accuracy !== undefined && ` with ${(results.accuracy * 100).toFixed(1)}% accuracy`}.
        </p>
      </div>
      
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card bg-gray-50 dark:bg-gray-800">
          <h3 className="font-semibold text-center">Total Engines</h3>
          <p className="text-3xl font-bold text-center text-gray-800 dark:text-gray-200 mt-2">{totalEngines}</p>
        </div>
        
        <div className="card bg-red-50 dark:bg-red-900/20">
          <h3 className="font-semibold text-center text-red-800 dark:text-red-300">Maintenance Required</h3>
          <p className="text-3xl font-bold text-center text-red-600 dark:text-red-400 mt-2">{maintenanceRequired}</p>
        </div>
        
        <div className="card bg-green-50 dark:bg-green-900/20">
          <h3 className="font-semibold text-center text-green-800 dark:text-green-300">No Maintenance Required</h3>
          <p className="text-3xl font-bold text-center text-green-600 dark:text-green-400 mt-2">{noMaintenanceRequired}</p>
        </div>
      </div>
      
      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="card">
          <h3 className="font-semibold mb-4">Maintenance Status Distribution</h3>
          <div className="h-64">
            <Pie data={pieData} options={{ maintainAspectRatio: false }} />
          </div>
        </div>
        
        <div className="card">
          <h3 className="font-semibold mb-4">Engine Failure Probabilities (First 20 Engines)</h3>
          <div className="h-64">
            <Bar 
              data={barData} 
              options={{ 
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 1
                  }
                }
              }} 
            />
          </div>
        </div>
      </div>
      
      {/* Engine List */}
      <div className="card">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
          <h3 className="font-semibold">Detailed Engine Results</h3>
          
          <div className="flex flex-wrap gap-2 w-full md:w-auto">
            <div className="relative flex-grow md:flex-grow-0">
              <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" width={16} height={16} />
              <input
                type="text"
                placeholder="Search engines..."
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-md w-full dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <select 
              className="border border-gray-300 rounded-md px-3 py-2 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
            >
              <option value="all">All Engines</option>
              <option value="maintenance">Maintenance Required</option>
              <option value="no-maintenance">No Maintenance Required</option>
            </select>
            
            <button 
              className="btn-secondary flex items-center" 
              onClick={exportToCsv}
            >
              <DownloadIcon className="mr-2" width={16} height={16} />
              Export CSV
            </button>
          </div>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Engine ID
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Status
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Failure Probability
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Recommendation
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
              {filteredEngines.map((engine) => (
                <tr key={engine.engine_id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    {engine.engine_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      engine.failure_predicted
                        ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                        : 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                    }`}>
                      {engine.failure_predicted ? 'Maintenance Required' : 'No Maintenance Required'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                    <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                      <div 
                        className={`h-2.5 rounded-full ${
                          engine.probability > 0.7 
                            ? 'bg-red-500' 
                            : engine.probability > 0.4 
                              ? 'bg-yellow-500' 
                              : 'bg-green-500'
                        }`} 
                        style={{ width: `${engine.probability * 100}%` }}
                      />
                    </div>
                    <span className="text-xs mt-1 block">{(engine.probability * 100).toFixed(1)}%</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                    {engine.failure_predicted ? (
                      <div className="flex items-center text-red-600 dark:text-red-400">
                        <AlertCircleIcon className="mr-2" width={16} height={16} />
                        Schedule maintenance within 30 cycles
                      </div>
                    ) : (
                      <div className="flex items-center text-green-600 dark:text-green-400">
                        <CheckIcon className="mr-2" width={16} height={16} />
                        Continue normal operation
                      </div>
                    )}
                  </td>
                </tr>
              ))}
              
              {filteredEngines.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-6 py-4 text-center text-gray-500 dark:text-gray-400">
                    No engines match your search criteria
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        
        <div className="mt-4 text-sm text-gray-500 dark:text-gray-400 text-center">
          Showing {filteredEngines.length} of {totalEngines} engines
        </div>
      </div>
    </div>
  );
};

export default PredictionResults;