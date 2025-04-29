'use client';

import React, { useState, useEffect } from 'react';
import { FaPlane, FaTools, FaCheckDouble, FaChartLine } from 'react-icons/fa';

// Backend URL - can be set in environment variables in production
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const DashboardStats = () => {
  const [stats, setStats] = useState({
    totalEngines: 0,
    pendingMaintenance: 0,
    healthyEngines: 0,
    modelAccuracy: 0
  });
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDashboardStats = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_URL}/api/stats`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        setStats(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching dashboard stats:', err);
        setError('Failed to load dashboard statistics. Please try again later.');
        // Use fallback data if API call fails
        setStats({
          totalEngines: 100,
          pendingMaintenance: 12,
          healthyEngines: 88,
          modelAccuracy: 92.5
        });
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardStats();
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="card flex items-center animate-pulse">
            <div className="bg-gray-200 dark:bg-gray-700 p-3 rounded-full mr-4 w-12 h-12"></div>
            <div className="flex-1">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
              <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-8 p-4 bg-red-100 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md text-red-700 dark:text-red-400">
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <div className="card flex items-center">
        <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded-full mr-4">
          <FaPlane className="text-blue-600 dark:text-blue-400 text-xl" />
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Total Engines</p>
          <h3 className="text-2xl font-bold">{stats.totalEngines}</h3>
        </div>
      </div>
      
      <div className="card flex items-center">
        <div className="bg-red-100 dark:bg-red-900/30 p-3 rounded-full mr-4">
          <FaTools className="text-red-600 dark:text-red-400 text-xl" />
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Pending Maintenance</p>
          <h3 className="text-2xl font-bold">{stats.pendingMaintenance}</h3>
        </div>
      </div>
      
      <div className="card flex items-center">
        <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-full mr-4">
          <FaCheckDouble className="text-green-600 dark:text-green-400 text-xl" />
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Healthy Engines</p>
          <h3 className="text-2xl font-bold">{stats.healthyEngines}</h3>
        </div>
      </div>
      
      <div className="card flex items-center">
        <div className="bg-purple-100 dark:bg-purple-900/30 p-3 rounded-full mr-4">
          <FaChartLine className="text-purple-600 dark:text-purple-400 text-xl" />
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Model Accuracy</p>
          <h3 className="text-2xl font-bold">{stats.modelAccuracy}%</h3>
        </div>
      </div>
    </div>
  );
};

export default DashboardStats;