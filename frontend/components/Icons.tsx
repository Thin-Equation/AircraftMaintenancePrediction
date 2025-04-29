/**
 * Custom SVG icon components
 */
import React from 'react';

export type IconProps = {
  className?: string;
  width?: number;
  height?: number;
  strokeWidth?: number;
  fill?: string;
  stroke?: string;
  onClick?: () => void;
};

export const MenuIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <line x1="3" y1="12" x2="21" y2="12"></line>
    <line x1="3" y1="6" x2="21" y2="6"></line>
    <line x1="3" y1="18" x2="21" y2="18"></line>
  </svg>
);

export const CloseIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

export const SunIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="5"></circle>
    <line x1="12" y1="1" x2="12" y2="3"></line>
    <line x1="12" y1="21" x2="12" y2="23"></line>
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
    <line x1="1" y1="12" x2="3" y2="12"></line>
    <line x1="21" y1="12" x2="23" y2="12"></line>
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
  </svg>
);

export const MoonIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
  </svg>
);

export const UploadIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="17 8 12 3 7 8"></polyline>
    <line x1="12" y1="3" x2="12" y2="15"></line>
  </svg>
);

export const CheckIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <polyline points="20 6 9 17 4 12"></polyline>
  </svg>
);

export const AlertCircleIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="8" x2="12" y2="12"></line>
    <line x1="12" y1="16" x2="12.01" y2="16"></line>
  </svg>
);

export const ChartIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <line x1="18" y1="20" x2="18" y2="10"></line>
    <line x1="12" y1="20" x2="12" y2="4"></line>
    <line x1="6" y1="20" x2="6" y2="14"></line>
  </svg>
);

export const SettingsIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="3"></circle>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
  </svg>
);

export const AirplaneIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
    <path d="M14.05 2.65l-2.94 2.93M8.12 17.88l8.23-8.23M17.18 13.06l2.93-2.94"></path>
  </svg>
);

export const DownloadIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="7 10 12 15 17 10"></polyline>
    <line x1="12" y1="15" x2="12" y2="3"></line>
  </svg>
);

export const FileIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
    <line x1="16" y1="13" x2="8" y2="13"></line>
    <line x1="16" y1="17" x2="8" y2="17"></line>
    <polyline points="10 9 9 9 8 9"></polyline>
  </svg>
);

export const SearchIcon: React.FC<IconProps> = ({ 
  className = '',
  width = 24,
  height = 24,
  strokeWidth = 1.5,
  stroke = 'currentColor',
  fill = 'none',
  onClick
}) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={width} 
    height={height} 
    viewBox="0 0 24 24" 
    fill={fill} 
    stroke={stroke} 
    strokeWidth={strokeWidth} 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className={className}
    onClick={onClick}
    aria-hidden="true"
  >
    <circle cx="11" cy="11" r="8"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
  </svg>
);