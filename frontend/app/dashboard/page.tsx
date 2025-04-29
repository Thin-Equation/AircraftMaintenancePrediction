'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import DataUpload from '@/components/DataUpload';
import ModelSelection from '@/components/ModelSelection';
import PredictionResults from '@/components/PredictionResults';
import DashboardStats from '@/components/DashboardStats';

// API base URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFiles, setUploadedFiles] = useState<{ [key: string]: File }>({});
  const [selectedModel, setSelectedModel] = useState('');
  const [predictionResults, setPredictionResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch available models when component mounts
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(`${API_URL}/api/models`);
        if (!response.ok) {
          throw new Error('Failed to fetch models');
        }
        const data = await response.json();
        // Selected model is now handled by the ModelSelection component
      } catch (err) {
        console.error('Error fetching models:', err);
      }
    };

    // Check if required files are uploaded
    const checkFiles = async () => {
      try {
        const response = await fetch(`${API_URL}/api/checkfiles`);
        if (!response.ok) {
          throw new Error('Failed to check files');
        }
        const data = await response.json();
        
        // If files are already uploaded, we can set the uploadedFiles state
        if (data.train_file && data.test_file && data.truth_file) {
          setUploadedFiles({
            'train': new File([], 'PM_train.txt'),
            'test': new File([], 'PM_test.txt'),
            'truth': new File([], 'PM_truth.txt')
          });
          
          // Move to model selection if files are already uploaded
          setActiveTab('model');
        }
      } catch (err) {
        console.error('Error checking files:', err);
      }
    };

    fetchModels();
    checkFiles();
  }, []);

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-grow bg-gray-50 dark:bg-gray-900 py-8">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold mb-8">Maintenance Prediction Dashboard</h1>
          
          <DashboardStats />
          
          {/* Tab Navigation */}
          <div className="mb-8">
            <div className="flex border-b border-gray-200">
              <button
                className={`py-2 px-4 font-medium ${
                  activeTab === 'upload'
                    ? 'text-primary-600 border-b-2 border-primary-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => handleTabChange('upload')}
              >
                1. Upload Data
              </button>
              <button
                className={`py-2 px-4 font-medium ${
                  activeTab === 'model'
                    ? 'text-primary-600 border-b-2 border-primary-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => handleTabChange('model')}
                disabled={!uploadedFiles || Object.keys(uploadedFiles).length === 0}
              >
                2. Select Model
              </button>
              <button
                className={`py-2 px-4 font-medium ${
                  activeTab === 'results'
                    ? 'text-primary-600 border-b-2 border-primary-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => handleTabChange('results')}
              >
                3. View Results
              </button>
            </div>
          </div>
          
          {/* Tab Content */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-8">
            {activeTab === 'upload' && (
              <DataUpload />
            )}
            
            {activeTab === 'model' && (
              <ModelSelection />
            )}
            
            {activeTab === 'results' && (
              <PredictionResults />
            )}
            
            {error && (
              <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                {error}
              </div>
            )}
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}