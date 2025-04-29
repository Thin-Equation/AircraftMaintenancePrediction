'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { UploadIcon, CheckIcon, FileIcon } from './Icons';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type DatasetInfo = {
  name: string;
  rows: number;
  columns: number;
  size_kb: number;
  upload_date: string;
};

const DataUpload = () => {
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  const [truthFile, setTruthFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [datasets, setDatasets] = useState<{[key: string]: DatasetInfo}>({});
  const [showDataInfo, setShowDataInfo] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, setFile: (file: File | null) => void) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const fetchDatasetInfo = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/datasets/info`);
      setDatasets(response.data);
      setShowDataInfo(true);
    } catch (error) {
      console.error('Error fetching dataset info:', error);
      toast.error('Failed to fetch dataset information');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!trainFile && !testFile && !truthFile) {
      toast.warning('Please select at least one file to upload');
      return;
    }
    
    const formData = new FormData();
    if (trainFile) formData.append('train_data', trainFile);
    if (testFile) formData.append('test_data', testFile);
    if (truthFile) formData.append('truth_data', truthFile);
    
    try {
      setIsUploading(true);
      const response = await axios.post(`${API_URL}/api/datasets/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      toast.success('Files uploaded successfully');
      setTrainFile(null);
      setTestFile(null);
      setTruthFile(null);
      
      // Refresh dataset info
      fetchDatasetInfo();
    } catch (error) {
      console.error('Error uploading files:', error);
      toast.error('Failed to upload files');
    } finally {
      setIsUploading(false);
    }
  };

  // Fetch dataset info when component mounts
  useEffect(() => {
    fetchDatasetInfo();
  }, []);

  const formatFileSize = (sizeInKB: number): string => {
    if (sizeInKB < 1000) return `${sizeInKB.toFixed(2)} KB`;
    return `${(sizeInKB / 1000).toFixed(2)} MB`;
  };

  return (
    <div className="card mb-6">
      <h2 className="section-title">Data Management</h2>
      
      <div className="mb-4">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Upload training, test, and ground truth data files in the required format for model training and evaluation.
        </p>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid sm:grid-cols-3 gap-4">
          {/* Training Data Upload */}
          <div className="border border-dashed border-gray-300 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-800 p-4 relative">
            <label className="flex flex-col items-center cursor-pointer">
              <div className="mb-2">
                <UploadIcon className="text-gray-500" width={24} height={24} />
              </div>
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Training Data</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 text-center mb-2">
                Upload PM_train.txt file
              </p>
              <input
                type="file"
                className="hidden"
                onChange={(e) => handleFileChange(e, setTrainFile)}
                accept=".txt,.csv"
              />
              <button
                type="button"
                className="text-xs py-1 px-3 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                onClick={() => {
                  const input = document.querySelectorAll('input[type="file"]')[0];
                  if (input) {
                    (input as HTMLInputElement).click();
                  }
                }}
              >
                Select File
              </button>
            </label>
            {trainFile && (
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center">
                  <FileIcon className="text-primary-500 mr-2" width={16} height={16} />
                  <span className="text-xs text-gray-700 dark:text-gray-300 truncate" title={trainFile.name}>
                    {trainFile.name}
                  </span>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {(trainFile.size / 1024).toFixed(1)} KB
                </p>
              </div>
            )}
          </div>
          
          {/* Test Data Upload */}
          <div className="border border-dashed border-gray-300 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-800 p-4">
            <label className="flex flex-col items-center cursor-pointer">
              <div className="mb-2">
                <UploadIcon className="text-gray-500" width={24} height={24} />
              </div>
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Test Data</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 text-center mb-2">
                Upload PM_test.txt file
              </p>
              <input
                type="file"
                className="hidden"
                onChange={(e) => handleFileChange(e, setTestFile)}
                accept=".txt,.csv"
              />
              <button
                type="button"
                className="text-xs py-1 px-3 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                onClick={() => {
                  const input = document.querySelectorAll('input[type="file"]')[1];
                  if (input) {
                    (input as HTMLInputElement).click();
                  }
                }}
              >
                Select File
              </button>
            </label>
            {testFile && (
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center">
                  <FileIcon className="text-primary-500 mr-2" width={16} height={16} />
                  <span className="text-xs text-gray-700 dark:text-gray-300 truncate" title={testFile.name}>
                    {testFile.name}
                  </span>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {(testFile.size / 1024).toFixed(1)} KB
                </p>
              </div>
            )}
          </div>
          
          {/* Ground Truth Data Upload */}
          <div className="border border-dashed border-gray-300 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-800 p-4">
            <label className="flex flex-col items-center cursor-pointer">
              <div className="mb-2">
                <UploadIcon className="text-gray-500" width={24} height={24} />
              </div>
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Ground Truth Data</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 text-center mb-2">
                Upload PM_truth.txt file
              </p>
              <input
                type="file"
                className="hidden"
                onChange={(e) => handleFileChange(e, setTruthFile)}
                accept=".txt,.csv"
              />
              <button
                type="button"
                className="text-xs py-1 px-3 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                onClick={() => {
                  const input = document.querySelectorAll('input[type="file"]')[2];
                  if (input) {
                    (input as HTMLInputElement).click();
                  }
                }}
              >
                Select File
              </button>
            </label>
            {truthFile && (
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center">
                  <FileIcon className="text-primary-500 mr-2" width={16} height={16} />
                  <span className="text-xs text-gray-700 dark:text-gray-300 truncate" title={truthFile.name}>
                    {truthFile.name}
                  </span>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {(truthFile.size / 1024).toFixed(1)} KB
                </p>
              </div>
            )}
          </div>
        </div>
        
        <div className="flex justify-end">
          <button
            type="submit"
            className="btn-primary"
            disabled={isUploading || (!trainFile && !testFile && !truthFile)}
          >
            {isUploading ? (
              <>
                <span className="inline-block animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full mr-2 align-middle"></span>
                <span>Uploading...</span>
              </>
            ) : (
              'Upload Data'
            )}
          </button>
        </div>
      </form>
      
      {showDataInfo && Object.keys(datasets).length > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-3">Available Datasets</h3>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Dataset
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Rows
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Columns
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Size
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Upload Date
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                {Object.entries(datasets).map(([name, info]) => (
                  <tr key={name}>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <CheckIcon className="text-green-500 mr-2" width={16} height={16} />
                        <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                          {name}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {info.rows.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {info.columns}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {formatFileSize(info.size_kb)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {info.upload_date}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      
      {showDataInfo && Object.keys(datasets).length === 0 && (
        <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <p className="text-sm text-yellow-700 dark:text-yellow-400">
            No datasets found. Please upload data files to get started.
          </p>
        </div>
      )}
    </div>
  );
};

export default DataUpload;