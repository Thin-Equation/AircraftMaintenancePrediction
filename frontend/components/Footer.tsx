'use client';

import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { FaGithub, FaTwitter, FaLinkedin, FaEnvelope } from 'react-icons/fa';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand column */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <div className="relative w-8 h-8">
                <Image
                  src="/logo.svg"
                  alt="Aircraft Maintenance Prediction"
                  fill
                  className="object-contain"
                />
              </div>
              <span className="text-gray-900 dark:text-white font-bold">
                Aircraft Maintenance
              </span>
            </div>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Advanced predictive maintenance for aircraft engines using machine learning and deep learning techniques.
            </p>
            <div className="flex space-x-4">
              <a href="https://github.com" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white">
                <FaGithub size={20} />
              </a>
              <a href="https://twitter.com" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white">
                <FaTwitter size={20} />
              </a>
              <a href="https://linkedin.com" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white">
                <FaLinkedin size={20} />
              </a>
            </div>
          </div>
          
          {/* Navigation Links */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-4">
              Navigation
            </h3>
            <ul className="space-y-2">
              <li>
                <Link 
                  href="/" 
                  className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 text-sm"
                >
                  Home
                </Link>
              </li>
              <li>
                <Link 
                  href="/dashboard" 
                  className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 text-sm"
                >
                  Dashboard
                </Link>
              </li>
              <li>
                <Link 
                  href="#about" 
                  className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 text-sm"
                >
                  About Us
                </Link>
              </li>
              <li>
                <Link 
                  href="#contact" 
                  className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 text-sm"
                >
                  Contact
                </Link>
              </li>
            </ul>
          </div>
          
          {/* Resources */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-4">
              Resources
            </h3>
            <ul className="space-y-2">
              <li>
                <Link 
                  href="#docs" 
                  className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 text-sm"
                >
                  Documentation
                </Link>
              </li>
              <li>
                <Link 
                  href="#api" 
                  className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 text-sm"
                >
                  API Reference
                </Link>
              </li>
              <li>
                <Link 
                  href="#models" 
                  className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 text-sm"
                >
                  Model Details
                </Link>
              </li>
              <li>
                <Link 
                  href="#blog" 
                  className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 text-sm"
                >
                  Blog
                </Link>
              </li>
            </ul>
          </div>
          
          {/* Contact */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-4">
              Contact Us
            </h3>
            <ul className="space-y-2">
              <li className="flex items-start">
                <FaEnvelope className="text-gray-600 dark:text-gray-400 mr-2 mt-1 flex-shrink-0" />
                <span className="text-gray-600 dark:text-gray-400 text-sm">
                  contact@aircraftmaintenance.ai
                </span>
              </li>
              <li className="text-gray-600 dark:text-gray-400 text-sm">
                123 Tech Park,<br />
                Innovation Street,<br />
                Aerospace City, 12345
              </li>
            </ul>
          </div>
        </div>
        
        <div className="border-t border-gray-200 dark:border-gray-800 mt-8 pt-8 text-center">
          <p className="text-gray-600 dark:text-gray-400 text-sm">
            © {currentYear} Aircraft Maintenance Prediction. All rights reserved.
          </p>
          <div className="mt-2 flex justify-center space-x-6 text-sm">
            <Link 
              href="/privacy" 
              className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400"
            >
              Privacy Policy
            </Link>
            <Link 
              href="/terms" 
              className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400"
            >
              Terms of Service
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;