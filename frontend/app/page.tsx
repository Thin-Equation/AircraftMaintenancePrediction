'use client';

import Link from 'next/link';
import { FaRocket, FaChartLine, FaUpload, FaCogs } from 'react-icons/fa';
import Header from '@/components/Header';
import Footer from '@/components/Footer';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-grow">
        {/* Hero Section */}
        <section className="bg-gradient-to-r from-primary-600 to-primary-800 text-white py-20">
          <div className="container mx-auto px-4">
            <div className="flex flex-col md:flex-row items-center">
              <div className="md:w-1/2 mb-10 md:mb-0">
                <h1 className="text-4xl md:text-5xl font-bold mb-4">
                  Aircraft Maintenance Prediction
                </h1>
                <p className="text-xl mb-8">
                  Predict aircraft engine failures using state-of-the-art machine learning models.
                  Reduce downtime, increase safety, and optimize maintenance schedules.
                </p>
                <div className="flex flex-col sm:flex-row gap-4">
                  <Link href="/dashboard" className="btn-primary text-center">
                    Launch Dashboard
                  </Link>
                  <Link href="/docs" className="bg-white text-primary-600 hover:bg-gray-100 font-semibold py-2 px-4 rounded-lg transition-all duration-300 text-center">
                    View Documentation
                  </Link>
                </div>
              </div>
              <div className="md:w-1/2 flex justify-center">
                <img 
                  src="/hero-image.svg" 
                  alt="Aircraft maintenance prediction" 
                  className="max-w-full rounded-lg shadow-xl"
                />
              </div>
            </div>
          </div>
        </section>
        
        {/* Features Section */}
        <section className="py-16 bg-gray-50 dark:bg-gray-900">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              <div className="card flex flex-col items-center text-center p-6">
                <div className="bg-primary-100 dark:bg-primary-900 p-3 rounded-full mb-4">
                  <FaUpload className="text-primary-600 dark:text-primary-300 text-3xl" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Data Upload</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Upload your aircraft sensor data for analysis and prediction
                </p>
              </div>
              
              <div className="card flex flex-col items-center text-center p-6">
                <div className="bg-primary-100 dark:bg-primary-900 p-3 rounded-full mb-4">
                  <FaCogs className="text-primary-600 dark:text-primary-300 text-3xl" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Multiple Models</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Choose from RNN, LSTM, or GRU models for optimal prediction performance
                </p>
              </div>
              
              <div className="card flex flex-col items-center text-center p-6">
                <div className="bg-primary-100 dark:bg-primary-900 p-3 rounded-full mb-4">
                  <FaChartLine className="text-primary-600 dark:text-primary-300 text-3xl" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Visualization</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Interactive charts and graphs for data exploration and result analysis
                </p>
              </div>
              
              <div className="card flex flex-col items-center text-center p-6">
                <div className="bg-primary-100 dark:bg-primary-900 p-3 rounded-full mb-4">
                  <FaRocket className="text-primary-600 dark:text-primary-300 text-3xl" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Custom Training</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  Train models with custom parameters for your specific use case
                </p>
              </div>
            </div>
          </div>
        </section>
        
        {/* How It Works Section */}
        <section className="py-16">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
            <div className="flex flex-col md:flex-row gap-8">
              <div className="md:w-1/2">
                <div className="space-y-8">
                  <div className="flex gap-4">
                    <div className="flex-shrink-0 bg-primary-600 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold">1</div>
                    <div>
                      <h3 className="text-xl font-semibold mb-2">Upload Sensor Data</h3>
                      <p className="text-gray-600 dark:text-gray-300">
                        Upload your aircraft engine sensor data in the supported format.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex gap-4">
                    <div className="flex-shrink-0 bg-primary-600 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold">2</div>
                    <div>
                      <h3 className="text-xl font-semibold mb-2">Select Model</h3>
                      <p className="text-gray-600 dark:text-gray-300">
                        Choose from different model architectures (RNN, LSTM, GRU).
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex gap-4">
                    <div className="flex-shrink-0 bg-primary-600 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold">3</div>
                    <div>
                      <h3 className="text-xl font-semibold mb-2">Train or Use Pre-trained</h3>
                      <p className="text-gray-600 dark:text-gray-300">
                        Train a new model on your data or use our pre-trained models.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex gap-4">
                    <div className="flex-shrink-0 bg-primary-600 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold">4</div>
                    <div>
                      <h3 className="text-xl font-semibold mb-2">Get Predictions</h3>
                      <p className="text-gray-600 dark:text-gray-300">
                        Receive predictions on which engines require maintenance attention.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="md:w-1/2 flex items-center justify-center">
                <img src="/workflow.svg" alt="Workflow" className="max-w-full rounded-lg" />
              </div>
            </div>
          </div>
        </section>
        
        {/* CTA Section */}
        <section className="py-16 bg-primary-600 text-white">
          <div className="container mx-auto px-4 text-center">
            <h2 className="text-3xl font-bold mb-4">Ready to optimize your aircraft maintenance?</h2>
            <p className="text-xl mb-8 max-w-2xl mx-auto">
              Start using our prediction tools today to reduce costs and improve safety.
            </p>
            <Link href="/dashboard" className="bg-white text-primary-600 hover:bg-gray-100 font-semibold py-3 px-6 rounded-lg text-lg transition-all duration-300">
              Get Started Now
            </Link>
          </div>
        </section>
      </main>
      
      <Footer />
    </div>
  );
}