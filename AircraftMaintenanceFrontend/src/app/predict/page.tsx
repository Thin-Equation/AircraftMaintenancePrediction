'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm, useFieldArray } from 'react-hook-form';
import ApiService from '@/services/api';
import { PredictionRequest, PredictionResponse, SensorReading } from '@/types';

const defaultSensorReading: SensorReading = {
  cycle: 1,
  setting1: 0.0,
  setting2: 0.0,
  setting3: 0.0,
  s1: 0.0, s2: 0.0, s3: 0.0, s4: 0.0, s5: 0.0,
  s6: 0.0, s7: 0.0, s8: 0.0, s9: 0.0, s10: 0.0,
  s11: 0.0, s12: 0.0, s13: 0.0, s14: 0.0, s15: 0.0,
  s16: 0.0, s17: 0.0, s18: 0.0, s19: 0.0, s20: 0.0, s21: 0.0
};

export default function PredictPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);

  // Initialize form with React Hook Form
  const { register, handleSubmit, control, setValue, formState: { errors } } = useForm<PredictionRequest>({
    defaultValues: {
      engine_id: 1,
      readings: Array(50).fill(null).map((_, index) => ({
        ...defaultSensorReading,
        cycle: index + 1
      }))
    }
  });

  // Use fieldArray to handle the array of readings
  const { fields } = useFieldArray({
    control,
    name: "readings"
  });

  // Handle form submission
  const onSubmit = async (data: PredictionRequest) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await ApiService.predict(data);
      setResult(response);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error 
        ? err.message 
        : typeof err === 'object' && err !== null && 'response' in err 
          ? (err.response as { data?: { detail?: string } })?.data?.detail || 'An error occurred during prediction'
          : 'An error occurred during prediction';
      
      setError(errorMessage);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Handle CSV upload
  const handleCsvUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadedFileName(file.name);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const rows = text.split('\n').filter(row => row.trim());
        
        // Parse CSV header to identify columns
        const headers = rows[0].split(',').map(h => h.trim());
        
        // Convert CSV data to readings
        const readings: SensorReading[] = [];
        
        for (let i = 1; i < Math.min(rows.length, 51); i++) {
          const values = rows[i].split(',');
          // Create a new reading with all required properties from defaultSensorReading
          const reading: SensorReading = { ...defaultSensorReading };
          
          // Update values from CSV where they match expected properties
          headers.forEach((header, index) => {
            if (values[index] && header in reading) {
              // Type-safe way to update properties
              switch (header) {
                case 'cycle': reading.cycle = parseFloat(values[index]); break;
                case 'setting1': reading.setting1 = parseFloat(values[index]); break;
                case 'setting2': reading.setting2 = parseFloat(values[index]); break;
                case 'setting3': reading.setting3 = parseFloat(values[index]); break;
                case 's1': reading.s1 = parseFloat(values[index]); break;
                case 's2': reading.s2 = parseFloat(values[index]); break;
                case 's3': reading.s3 = parseFloat(values[index]); break;
                case 's4': reading.s4 = parseFloat(values[index]); break;
                case 's5': reading.s5 = parseFloat(values[index]); break;
                case 's6': reading.s6 = parseFloat(values[index]); break;
                case 's7': reading.s7 = parseFloat(values[index]); break;
                case 's8': reading.s8 = parseFloat(values[index]); break;
                case 's9': reading.s9 = parseFloat(values[index]); break;
                case 's10': reading.s10 = parseFloat(values[index]); break;
                case 's11': reading.s11 = parseFloat(values[index]); break;
                case 's12': reading.s12 = parseFloat(values[index]); break;
                case 's13': reading.s13 = parseFloat(values[index]); break;
                case 's14': reading.s14 = parseFloat(values[index]); break;
                case 's15': reading.s15 = parseFloat(values[index]); break;
                case 's16': reading.s16 = parseFloat(values[index]); break;
                case 's17': reading.s17 = parseFloat(values[index]); break;
                case 's18': reading.s18 = parseFloat(values[index]); break;
                case 's19': reading.s19 = parseFloat(values[index]); break;
                case 's20': reading.s20 = parseFloat(values[index]); break;
                case 's21': reading.s21 = parseFloat(values[index]); break;
              }
            }
          });
          
          readings.push(reading);
        }

        // Update form with parsed data
        if (readings.length > 0) {
          // Update the form values with the readings
          setValue('readings', readings);
          alert(`CSV processed! ${readings.length} readings loaded.`);
        } else {
          setError('No valid readings found in the CSV file. Please check the format.');
        }
      } catch (err) {
        setError('Failed to parse CSV file. Please ensure it has the correct format.');
        console.error('CSV parsing error:', err);
      }
    };
    reader.readAsText(file);
  };

  // Handle text file upload
  const handleTextFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadedFileName(file.name);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.split('\n').filter(line => line.trim());
        
        // Parse text file data into readings
        const readings: SensorReading[] = [];
        
        // Check if it's a space-delimited file (common format for text files)
        for (let i = 0; i < Math.min(lines.length, 51); i++) {
          const values = lines[i].trim().split(/\s+/);
          
          // Skip the line if it doesn't have enough values
          if (values.length < 26) continue; // We need at least 26 values (all sensors + settings + cycle)
          
          // Create a sensor reading with default values
          const reading: SensorReading = { ...defaultSensorReading };
          
          // Assume a typical text file format where columns are in order:
          // cycle, setting1, setting2, setting3, s1, s2, ..., s21
          try {
            reading.cycle = parseFloat(values[0]);
            reading.setting1 = parseFloat(values[1]);
            reading.setting2 = parseFloat(values[2]);
            reading.setting3 = parseFloat(values[3]);
            reading.s1 = parseFloat(values[4]);
            reading.s2 = parseFloat(values[5]);
            reading.s3 = parseFloat(values[6]);
            reading.s4 = parseFloat(values[7]);
            reading.s5 = parseFloat(values[8]);
            reading.s6 = parseFloat(values[9]);
            reading.s7 = parseFloat(values[10]);
            reading.s8 = parseFloat(values[11]);
            reading.s9 = parseFloat(values[12]);
            reading.s10 = parseFloat(values[13]);
            reading.s11 = parseFloat(values[14]);
            reading.s12 = parseFloat(values[15]);
            reading.s13 = parseFloat(values[16]);
            reading.s14 = parseFloat(values[17]);
            reading.s15 = parseFloat(values[18]);
            reading.s16 = parseFloat(values[19]);
            reading.s17 = parseFloat(values[20]);
            reading.s18 = parseFloat(values[21]);
            reading.s19 = parseFloat(values[22]);
            reading.s20 = parseFloat(values[23]);
            reading.s21 = parseFloat(values[24]);
          } catch (err) {
            console.error("Error parsing value:", err);
            continue;
          }
          
          readings.push(reading);
        }

        // Update form with parsed data
        if (readings.length > 0) {
          // Update the form values with the readings
          setValue('readings', readings);
          alert(`Text file processed! ${readings.length} readings loaded.`);
        } else {
          setError('No valid readings found in the text file. Please check the format.');
        }
      } catch (err) {
        setError('Failed to parse text file. Please ensure it has the correct format.');
        console.error('Text file parsing error:', err);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Make Prediction</h1>
        <button 
          onClick={() => router.push('/')}
          className="bg-gray-200 px-4 py-2 rounded hover:bg-gray-300"
        >
          Back to Dashboard
        </button>
      </div>

      {/* Result display */}
      {result && (
        <div className={`p-4 rounded-lg mb-6 ${result.prediction === 1 
          ? 'bg-red-100 border border-red-400' 
          : 'bg-green-100 border border-green-400'}`}
        >
          <h2 className="text-xl font-bold mb-2">Prediction Result</h2>
          <p className="mb-1">
            <span className="font-semibold">Engine ID:</span> {result.engine_id}
          </p>
          <p className="mb-1">
            <span className="font-semibold">Prediction:</span> {result.prediction === 1 
              ? 'Engine failure likely within 30 cycles' 
              : 'No failure expected within 30 cycles'}
          </p>
          <p className="mb-1">
            <span className="font-semibold">Probability:</span> {(result.probability * 100).toFixed(2)}%
          </p>
          <p className="mb-1">
            <span className="font-semibold">Message:</span> {result.message}
          </p>
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
        <form onSubmit={handleSubmit(onSubmit)}>
          {/* Engine ID */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Engine ID
            </label>
            <input
              type="number"
              {...register('engine_id', { required: true, min: 1 })}
              className="w-full p-2 border rounded"
            />
            {errors.engine_id && <span className="text-red-500">Engine ID is required</span>}
          </div>

          {/* File Upload Section */}
          <div className="mb-6">
            <h3 className="text-lg font-medium text-gray-800 mb-2">Upload Sensor Data</h3>
            
            {/* CSV Upload */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                CSV Format
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={handleCsvUpload}
                className="w-full p-2 border rounded"
              />
              <p className="text-sm text-gray-500 mt-1">
                Upload a CSV file with sensor readings. The file should have columns matching sensor names.
              </p>
            </div>

            {/* Text File Upload */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Text File Format
              </label>
              <input
                type="file"
                accept=".txt,.text,.dat"
                onChange={handleTextFileUpload}
                className="w-full p-2 border rounded"
              />
              <p className="text-sm text-gray-500 mt-1">
                Upload a space-separated text file with sensor readings. Each line should contain values in order: cycle, setting1-3, s1-s21.
              </p>
            </div>
            
            {uploadedFileName && (
              <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded">
                <p className="text-sm text-blue-800">
                  <span className="font-medium">File uploaded:</span> {uploadedFileName}
                </p>
              </div>
            )}
          </div>

          {/* Sample readings editor (simplified for the interface) */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Sample Readings (showing first 5 of {fields.length})
            </label>
            <div className="border rounded overflow-hidden">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Cycle</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Setting 1</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Setting 2</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Setting 3</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Sensor 2</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Sensor 6</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {fields.slice(0, 5).map((field, index) => (
                    <tr key={field.id}>
                      <td className="px-3 py-2">
                        <input
                          type="number"
                          {...register(`readings.${index}.cycle` as const)}
                          className="w-16 p-1 border rounded"
                        />
                      </td>
                      <td className="px-3 py-2">
                        <input
                          type="number"
                          step="0.01"
                          {...register(`readings.${index}.setting1` as const)}
                          className="w-16 p-1 border rounded"
                        />
                      </td>
                      <td className="px-3 py-2">
                        <input
                          type="number"
                          step="0.01"
                          {...register(`readings.${index}.setting2` as const)}
                          className="w-16 p-1 border rounded"
                        />
                      </td>
                      <td className="px-3 py-2">
                        <input
                          type="number"
                          step="0.01"
                          {...register(`readings.${index}.setting3` as const)}
                          className="w-16 p-1 border rounded"
                        />
                      </td>
                      <td className="px-3 py-2">
                        <input
                          type="number"
                          step="0.01"
                          {...register(`readings.${index}.s2` as const)}
                          className="w-16 p-1 border rounded"
                        />
                      </td>
                      <td className="px-3 py-2">
                        <input
                          type="number"
                          step="0.01"
                          {...register(`readings.${index}.s6` as const)}
                          className="w-16 p-1 border rounded"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="bg-gray-50 px-4 py-3 text-sm text-gray-500">
                Showing 5 of {fields.length} readings. CSV upload is recommended for complete data.
              </div>
            </div>
          </div>

          {/* Submit button */}
          <div className="flex justify-end mt-6">
            <button
              type="submit"
              disabled={loading}
              className={`px-6 py-2 rounded-lg text-white font-medium ${
                loading ? 'bg-blue-400' : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {loading ? 'Processing...' : 'Make Prediction'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}