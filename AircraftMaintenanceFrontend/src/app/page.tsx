'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import ApiService from '@/services/api';
import { Model } from '@/types';

export default function Home() {
  const [apiAvailable, setApiAvailable] = useState<boolean | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  // Check API availability and load models when component mounts
  useEffect(() => {
    const checkApiAndLoadModels = async () => {
      try {
        const isAvailable = await ApiService.healthCheck();
        setApiAvailable(isAvailable);
        
        if (isAvailable) {
          const response = await ApiService.getModels();
          setModels(response.models || []);
        }
      } catch (error) {
        console.error('Error checking API status:', error);
        setApiAvailable(false);
      } finally {
        setLoading(false);
      }
    };

    checkApiAndLoadModels();
  }, []);

  return (
    <div className="space-y-8">
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-2xl font-bold mb-4">Welcome to Aircraft Predictive Maintenance System</h2>
        <p className="text-gray-700 mb-4">
          This system helps predict aircraft engine failures using machine learning models built with PyTorch.
          You can train new models or make predictions using existing models.
        </p>
        
        {/* API Status Indicator */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">API Status:</h3>
          {loading ? (
            <p className="text-gray-500">Checking connection...</p>
          ) : apiAvailable ? (
            <div className="flex items-center">
              <div className="h-3 w-3 rounded-full bg-green-500 mr-2"></div>
              <p className="text-green-600">Connected to API server</p>
            </div>
          ) : (
            <div className="flex items-center">
              <div className="h-3 w-3 rounded-full bg-red-500 mr-2"></div>
              <p className="text-red-600">
                Cannot connect to API server. Please ensure the backend is running.
              </p>
            </div>
          )}
        </div>
        
        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <Link href="/predict" className="bg-blue-600 text-white p-4 rounded-lg shadow hover:bg-blue-700 text-center transition">
            Make Prediction
          </Link>
          <Link href="/train" className="bg-purple-600 text-white p-4 rounded-lg shadow hover:bg-purple-700 text-center transition">
            Train New Model
          </Link>
        </div>
      </div>

      {/* Available Models */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Available Models</h2>
        
        {loading ? (
          <p className="text-gray-500">Loading models...</p>
        ) : models.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model Type</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Size (MB)</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {models.map((model, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {model.model_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {model.size_mb}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-gray-500">
            {apiAvailable ? 
              "No models available. Please train a model first." : 
              "Cannot retrieve models. Please check API connection."}
          </div>
        )}
      </div>
      
      {/* System Information */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">System Information</h2>
        <div className="space-y-2 text-gray-600">
          <p><span className="font-semibold">Frontend:</span> Next.js with TypeScript and Tailwind CSS</p>
          <p><span className="font-semibold">Backend:</span> FastAPI with PyTorch models</p>
          <p><span className="font-semibold">Available Models:</span> Simple RNN, LSTM, Bidirectional RNN, GRU</p>
        </div>
      </div>
    </div>
  );
}
