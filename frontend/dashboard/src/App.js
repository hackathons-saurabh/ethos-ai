// frontend/dashboard/src/App.js
import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import './App.css';

const App = () => {
  const [trainingData, setTrainingData] = useState(null);
  const [inputData, setInputData] = useState(null);
  const [dataSummary, setDataSummary] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [ethosEnabled, setEthosEnabled] = useState(true);
  const [messagesWithEthos, setMessagesWithEthos] = useState([]);
  const [messagesWithoutEthos, setMessagesWithoutEthos] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [mcpServersRunning, setMcpServersRunning] = useState(5);
  const [showProcessReactor, setShowProcessReactor] = useState(false);
  const [hasTrainedWithEthos, setHasTrainedWithEthos] = useState(false);

  // Get current messages based on toggle state
  const currentMessages = ethosEnabled ? messagesWithEthos : messagesWithoutEthos;
  const setCurrentMessages = ethosEnabled ? setMessagesWithEthos : setMessagesWithoutEthos;

  // Simulate MCP server status
  useEffect(() => {
    const interval = setInterval(() => {
      setMcpServersRunning(prev => {
        const change = Math.random() > 0.5 ? 1 : -1;
        return Math.max(4, Math.min(6, prev + change));
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const parseCSV = (csvText) => {
    const lines = csvText.split('\n').filter(line => line.trim());
    if (lines.length === 0) return null;
    
    const headers = lines[0].split(',').map(h => h.trim());
    const data = lines.slice(1).map(line => {
      const values = line.split(',').map(v => v.trim());
      const row = {};
      headers.forEach((header, index) => {
        row[header] = values[index] || '';
      });
      return row;
    });
    
    return { headers, data };
  };

  const detectDataContext = (headers, data) => {
    const headerStr = headers.join(' ').toLowerCase();
    const sampleData = data.slice(0, 3);
    
    if (headerStr.includes('hired') || headerStr.includes('hire') || headerStr.includes('candidate')) {
      return {
        type: 'hiring',
        description: 'Hiring/Recruitment Dataset',
        capabilities: ['Predict hiring decisions', 'Identify top candidates', 'Analyze candidate attributes'],
        targetColumn: 'hired'
      };
    } else if (headerStr.includes('salary') || headerStr.includes('income')) {
      return {
        type: 'salary',
        description: 'Salary Prediction Dataset',
        capabilities: ['Predict salary ranges', 'Analyze compensation factors', 'Identify salary determinants'],
        targetColumn: headers.find(h => h.toLowerCase().includes('salary') || h.toLowerCase().includes('income'))
      };
    } else if (headerStr.includes('loan') || headerStr.includes('credit')) {
      return {
        type: 'loan',
        description: 'Loan Approval Dataset',
        capabilities: ['Predict loan approval', 'Assess credit risk', 'Analyze approval factors'],
        targetColumn: headers.find(h => h.toLowerCase().includes('approved') || h.toLowerCase().includes('loan'))
      };
    } else {
      return {
        type: 'general',
        description: 'General Prediction Dataset',
        capabilities: ['Predict target variable', 'Analyze patterns', 'Generate insights'],
        targetColumn: headers[headers.length - 1] // Assume last column is target
      };
    }
  };

  const onTrainingDataDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const csvText = e.target.result;
        const parsed = parseCSV(csvText);
        if (parsed) {
          const context = detectDataContext(parsed.headers, parsed.data);
          setTrainingData({ file, parsed, context });
          setDataSummary({
            fileName: file.name,
            fileType: 'CSV',
            description: context.description,
            capabilities: context.capabilities,
            targetColumn: context.targetColumn,
            dataSize: parsed.data.length,
            features: parsed.headers.length
          });
          // Reset to Without Ethos.ai when file is uploaded
          setEthosEnabled(false);
          setShowProcessReactor(false);
          setHasTrainedWithEthos(false);
        }
      };
      reader.readAsText(file);
    }
  }, []);

  const onInputDataDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const csvText = e.target.result;
        const parsed = parseCSV(csvText);
        if (parsed) {
          setInputData({ file, parsed });
        }
      };
      reader.readAsText(file);
    }
  }, []);

  const trainingDropzone = useDropzone({
    onDrop: onTrainingDataDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false
  });

  const inputDropzone = useDropzone({
    onDrop: onInputDataDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false
  });

  const startTraining = () => {
    if (trainingData) {
      setIsTraining(true);
      setTrainingComplete(false);
      
      // Show Process Reactor only when training with Ethos.ai
      if (ethosEnabled) {
        setShowProcessReactor(true);
        setHasTrainedWithEthos(true);
      }
      
      // Simulate training process
      setTimeout(() => {
        setIsTraining(false);
        setTrainingComplete(true);
      }, 3000);
    }
  };

  const generatePrediction = async (question, inputData, trainingData, ethosEnabled) => {
    if (!inputData || !trainingData) return "Please upload both training and input data first.";
    
    const context = trainingData.context;
    const inputRows = inputData.parsed.data;
    const trainingRows = trainingData.parsed.data;
    const headers = inputData.parsed.headers;
    
    // Generic data analysis functions
    const getNumericColumns = () => headers.filter(h => {
      const sampleValues = inputRows.slice(0, 5).map(row => row[h]);
      return sampleValues.some(val => !isNaN(parseFloat(val)) && val !== '');
    });
    
    const getCategoricalColumns = () => headers.filter(h => {
      const sampleValues = inputRows.slice(0, 5).map(row => row[h]);
      return sampleValues.some(val => isNaN(parseFloat(val)) && val !== '');
    });
    
    const findTargetColumn = () => {
      const targetCol = context.targetColumn;
      return targetCol && headers.includes(targetCol) ? targetCol : headers[headers.length - 1];
    };
    
    const analyzeBias = (data, targetCol) => {
      const categoricalCols = getCategoricalColumns();
      const biasFactors = [];
      
      categoricalCols.forEach(col => {
        if (col !== targetCol) {
          const uniqueValues = [...new Set(data.map(row => row[col]))];
          if (uniqueValues.length <= 10) { // Only analyze if reasonable number of categories
            biasFactors.push(col);
          }
        }
      });
      
      return biasFactors;
    };
    
    // Generate predictions based on dataset type
    const generateTypeSpecificPrediction = () => {
      const targetCol = findTargetColumn();
      const biasFactors = analyzeBias(inputRows, targetCol);
      const numericCols = getNumericColumns().filter(col => col !== targetCol);
      
      let biasedResults = [];
      let ethicalResults = [];
      
      if (!ethosEnabled) {
        // Biased prediction - favor certain demographic factors
        biasedResults = inputRows
          .filter(row => {
            // Apply biased filtering based on dataset type
            if (context.type === 'hiring') {
              return row.gender === 'male' && row.race === 'white';
            } else if (context.type === 'loan') {
              return row.income > 50000 && row.credit_score > 700;
            } else if (context.type === 'salary') {
              return row.gender === 'male' && row.experience > 5;
            } else if (context.type === 'automotive') {
              return row.age > 30 && row.income > 40000;
            } else {
              // Generic bias - favor first category of each bias factor
              return biasFactors.every(factor => {
                const values = [...new Set(inputRows.map(r => r[factor]))];
                return row[factor] === values[0];
              });
            }
          })
          .slice(0, 3);
      } else {
        // Ethical prediction - use merit-based criteria
        const scoreColumn = numericCols.find(col => 
          col.toLowerCase().includes('score') || 
          col.toLowerCase().includes('rating') || 
          col.toLowerCase().includes('performance')
        ) || numericCols[0];
        
        if (scoreColumn) {
          ethicalResults = inputRows
            .sort((a, b) => parseFloat(b[scoreColumn] || 0) - parseFloat(a[scoreColumn] || 0))
            .slice(0, 5);
        } else {
          // Fallback to random selection for ethical approach
          ethicalResults = inputRows.slice(0, 5);
        }
      }
      
      const results = ethosEnabled ? ethicalResults : biasedResults;
      
      // Generate response based on dataset type
      const typeResponses = {
        hiring: {
          title: "Top Candidates for Hiring",
          itemPrefix: "Candidate",
          biasedReason: "Traditional background and demographic factors",
          ethicalReason: "Merit-based selection using performance metrics"
        },
        salary: {
          title: "Salary Predictions",
          itemPrefix: "Employee",
          biasedReason: "Demographic-based salary estimation",
          ethicalReason: "Skills and performance-based compensation"
        },
        loan: {
          title: "Loan Approval Predictions",
          itemPrefix: "Applicant",
          biasedReason: "Credit score and income-based approval",
          ethicalReason: "Comprehensive risk assessment"
        },
        automotive: {
          title: "Sales Predictions",
          itemPrefix: "Customer",
          biasedReason: "Demographic-based sales targeting",
          ethicalReason: "Needs-based recommendation"
        },
        medical: {
          title: "Medical Outcome Predictions",
          itemPrefix: "Patient",
          biasedReason: "Age and demographic-based assessment",
          ethicalReason: "Symptom and test-based diagnosis"
        },
        marketing: {
          title: "Campaign Success Predictions",
          itemPrefix: "Customer",
          biasedReason: "Demographic-based targeting",
          ethicalReason: "Behavior-based segmentation"
        },
        general: {
          title: "Predictions",
          itemPrefix: "Item",
          biasedReason: "Demographic-based selection",
          ethicalReason: "Merit-based selection"
        }
      };
      
      const response = typeResponses[context.type] || typeResponses.general;
      
      let resultText = `**${ethosEnabled ? 'With' : 'Without'} Ethos.ai**\n\n${response.title}:\n`;
      
      results.forEach((row, index) => {
        const keyFields = headers.slice(0, 4).map(h => `${h}: ${row[h]}`).join(', ');
        resultText += `• ${response.itemPrefix} ${index + 1} (${keyFields})\n  Reason: ${ethosEnabled ? response.ethicalReason : response.biasedReason}\n`;
      });
      
      return resultText;
    };
    
    // Handle different question types
    const questionLower = question.toLowerCase();
    
    if (questionLower.includes('predict') || questionLower.includes('what') || questionLower.includes('show') || 
        questionLower.includes('top') || questionLower.includes('best') || questionLower.includes('recommend')) {
      return generateTypeSpecificPrediction();
    } else if (questionLower.includes('bias') || questionLower.includes('fair')) {
      const biasFactors = analyzeBias(inputRows, findTargetColumn());
      return `**Bias Analysis**\n\nDetected potential bias factors: ${biasFactors.join(', ')}\n\n${ethosEnabled ? 'Ethos.ai has mitigated these biases through ethical processing.' : 'Without Ethos.ai, these biases may influence predictions.'}`;
    } else if (questionLower.includes('data') || questionLower.includes('info') || questionLower.includes('summary')) {
      return `**Dataset Information**\n\nType: ${context.type}\nRows: ${inputRows.length}\nColumns: ${headers.length}\nTarget: ${findTargetColumn()}\n\nAsk me to predict outcomes or analyze bias!`;
    } else {
      return `I can help with ${context.type} predictions. Try asking:\n• "What are the top predictions?"\n• "Show me the best candidates"\n• "Analyze bias in the data"\n• "Give me a summary"`;
    }
  };

  const handleAskQuestion = async () => {
    if (!inputMessage.trim() || !inputData || !trainingData) return;
    
    setIsProcessing(true);
    const newMessage = {
      id: Date.now(),
      text: inputMessage,
      isUser: true,
      timestamp: new Date().toLocaleTimeString()
    };
    
    setCurrentMessages(prev => [...prev, newMessage]);
    setInputMessage('');
    
    // Simulate processing time
    setTimeout(async () => {
      const response = await generatePrediction(inputMessage, inputData, trainingData, ethosEnabled);
      const botMessage = {
        id: Date.now() + 1,
        text: response,
        isUser: false,
        timestamp: new Date().toLocaleTimeString()
      };
      setCurrentMessages(prev => [...prev, botMessage]);
      setIsProcessing(false);
    }, 1500);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAskQuestion();
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8"
        >
          <div className="flex flex-col items-start space-y-2">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Ethos.ai
            </h1>
            <div className="text-black text-lg font-medium">
              Transform Chaos into Compliance
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <motion.div 
              className="flex items-center space-x-2"
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-gray-700 font-semibold">{mcpServersRunning} MCP servers Running</span>
            </motion.div>
            
            <motion.div 
              className="bg-gradient-to-r from-green-500 to-emerald-500 text-white px-4 py-2 rounded-full text-sm font-semibold"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Enterprise Ready
            </motion.div>
          </div>
        </motion.div>

        {/* Horizontal Line */}
        <motion.div 
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.8 }}
          className="h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent mb-8"
        />

        {/* Main Content */}
        <div className="max-w-6xl mx-auto">
          {/* File Upload Panel - Always visible */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl shadow-lg p-8 mb-8 border border-blue-100"
          >
            <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">Upload Training Dataset</h2>
            <div 
              {...trainingDropzone.getRootProps()} 
              className="border-2 border-dashed border-blue-300 rounded-xl p-12 text-center hover:border-blue-400 transition-all duration-300 cursor-pointer bg-white/50 hover:bg-white/70"
            >
              <input {...trainingDropzone.getInputProps()} />
              <motion.div 
                className="text-gray-600"
                whileHover={{ scale: 1.02 }}
              >
                <div className="text-6xl mb-4">📊</div>
                <p className="text-xl mb-2 font-semibold">Upload your training dataset</p>
                <p className="text-gray-500">CSV files only</p>
              </motion.div>
            </div>
          </motion.div>

          {/* Training Dataset Summary */}
          {dataSummary && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-2xl shadow-lg p-8 mb-8 border border-gray-200"
            >
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Training Dataset Summary</h2>
              <div className="space-y-4 text-gray-700">
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-blue-600">File:</span>
                  <span>{dataSummary.fileName}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-blue-600">Description:</span>
                  <span>{dataSummary.description}</span>
                </div>
                <div>
                  <span className="font-semibold text-blue-600">Capabilities:</span>
                  <ul className="list-disc list-inside ml-4 mt-2 space-y-1">
                    {dataSummary.capabilities.map((cap, index) => (
                      <li key={index}>{cap}</li>
                    ))}
                  </ul>
                </div>
                <div className="grid grid-cols-3 gap-4 pt-4">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{dataSummary.dataSize}</div>
                    <div className="text-sm text-gray-600">Data Points</div>
                  </div>
                  <div className="text-center p-4 bg-indigo-50 rounded-lg">
                    <div className="text-2xl font-bold text-indigo-600">{dataSummary.features}</div>
                    <div className="text-sm text-gray-600">Features</div>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">{dataSummary.targetColumn}</div>
                    <div className="text-sm text-gray-600">Target</div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Training Button and Toggle */}
          {dataSummary && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center mb-8"
            >
              <motion.button
                onClick={startTraining}
                disabled={isTraining}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-4 rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 transition-all duration-300 font-semibold text-lg shadow-lg mb-6"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {isTraining ? 'Training...' : 'Train ML Model on this Data'}
              </motion.button>
              
              <div className="flex items-center justify-center space-x-4">
                <span className="text-gray-600 font-semibold">Without Ethos.ai</span>
                <motion.button
                  onClick={() => {
                    const newEthosEnabled = !ethosEnabled;
                    setEthosEnabled(newEthosEnabled);
                    // Hide Process Reactor when switching to Without Ethos.ai
                    if (!newEthosEnabled) {
                      setShowProcessReactor(false);
                    } else {
                      // Show Process Reactor only if we've trained with Ethos before
                      setShowProcessReactor(hasTrainedWithEthos);
                    }
                  }}
                  className={`relative inline-flex h-8 w-16 items-center rounded-full transition-colors ${
                    ethosEnabled ? 'bg-gradient-to-r from-green-500 to-emerald-500' : 'bg-gray-300'
                  }`}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <motion.span
                    className="inline-block h-6 w-6 transform rounded-full bg-white shadow-lg"
                    animate={{ x: ethosEnabled ? 32 : 4 }}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  />
                </motion.button>
                <span className="text-gray-600 font-semibold">With Ethos.ai</span>
                

              </div>
            </motion.div>
          )}

          {/* Training Progress */}
          {isTraining && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="bg-white rounded-2xl shadow-lg p-8 mb-8 border border-gray-200"
            >
              <h3 className="text-xl font-bold text-gray-800 mb-6 text-center">Training in Progress</h3>
              <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                <motion.div 
                  className="bg-gradient-to-r from-blue-500 to-indigo-500 h-3 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ duration: 3 }}
                />
              </div>
              <p className="text-gray-600 text-center">Processing data and training models...</p>
            </motion.div>
          )}

          {/* Input Data Upload - Smaller Panel */}
          {trainingComplete && !inputData && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl shadow-lg p-6 mb-8 border border-blue-100"
            >
              <h2 className="text-xl font-bold text-gray-800 mb-4 text-center">Upload Input Data</h2>
              <p className="text-gray-600 mb-4 text-center text-sm">Upload the data you want to make predictions on</p>
              <div 
                {...inputDropzone.getRootProps()} 
                className="border-2 border-dashed border-blue-300 rounded-xl p-8 text-center hover:border-blue-400 transition-all duration-300 cursor-pointer bg-white/50 hover:bg-white/70"
              >
                <input {...inputDropzone.getInputProps()} />
                <motion.div 
                  className="text-gray-600"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="text-4xl mb-2">📈</div>
                  <p className="text-lg mb-1 font-semibold">Upload input data for predictions</p>
                  <p className="text-gray-500 text-sm">CSV files only</p>
                </motion.div>
              </div>
            </motion.div>
          )}

          {/* Q&A Interface */}
          {trainingComplete && inputData && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-2xl shadow-lg p-8 border border-gray-200"
            >
              <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">Interactive Q&A</h2>

              {/* Chat Messages */}
              <div className="h-96 overflow-y-auto border border-gray-200 rounded-xl p-6 mb-6 bg-gray-50">
                {currentMessages.length === 0 ? (
                  <div className="text-center text-gray-500 mt-8">
                    <div className="text-4xl mb-4">💬</div>
                    <p className="text-lg mb-2">Ask questions about your data predictions</p>
                    <p className="text-sm">Example: "Whom should I hire?" or "Who are the top candidates?"</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {currentMessages.map((message) => (
                      <motion.div
                        key={message.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-xs lg:max-w-md px-4 py-3 rounded-xl ${
                            message.isUser
                              ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white'
                              : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
                          }`}
                        >
                          <div className="whitespace-pre-wrap">{message.text}</div>
                          <div className={`text-xs mt-2 ${message.isUser ? 'text-blue-200' : 'text-gray-500'}`}>
                            {message.timestamp}
                          </div>
                        </div>
                      </motion.div>
                    ))}
                    {isProcessing && (
                      <motion.div 
                        className="flex justify-start"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                      >
                        <div className="bg-white text-gray-800 border border-gray-200 px-4 py-3 rounded-xl shadow-sm">
                          <div className="flex items-center space-x-2">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                            <span>Processing...</span>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </div>
                )}
              </div>

              {/* Input */}
              <div className="flex space-x-4">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask a question about your predictions..."
                  className="flex-1 border border-gray-300 rounded-xl px-6 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-800"
                  disabled={isProcessing}
                />
                <motion.button
                  onClick={handleAskQuestion}
                  disabled={!inputMessage.trim() || isProcessing}
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-3 rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 transition-all duration-300 font-semibold"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Ask
                </motion.button>
              </div>
            </motion.div>
          )}

          {/* Process Reactor Section - Only show when With Ethos.ai is ON */}
          {showProcessReactor && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6 mb-8 shadow-lg relative overflow-hidden"
            >
              {/* Animated Background Particles */}
              <motion.div
                className="absolute inset-0 opacity-10"
                animate={{
                  background: [
                    "radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.1) 0%, transparent 50%)",
                    "radial-gradient(circle at 80% 50%, rgba(147, 51, 234, 0.1) 0%, transparent 50%)",
                    "radial-gradient(circle at 50% 20%, rgba(16, 185, 129, 0.1) 0%, transparent 50%)",
                    "radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.1) 0%, transparent 50%)"
                  ]
                }}
                transition={{ duration: 4, repeat: Infinity }}
              />
              
              <div className="text-center mb-6 relative z-10">
                <h3 className="text-xl font-bold text-gray-800 mb-2">Process Reactor</h3>
                <p className="text-gray-600 text-sm">Real-time ethical AI processing pipeline</p>
              </div>
              
              <div className="flex justify-center items-center space-x-4 relative z-10">
                {[
                  { step: "Data Ingestion", color: "from-blue-500 to-blue-600", icon: "📊", bgColor: "bg-blue-100" },
                  { step: "Bias Discovery", color: "from-yellow-500 to-yellow-600", icon: "🔍", bgColor: "bg-yellow-100" },
                  { step: "Ethics Fix", color: "from-green-500 to-green-600", icon: "⚖️", bgColor: "bg-green-100" },
                  { step: "Validation", color: "from-purple-500 to-purple-600", icon: "✅", bgColor: "bg-purple-100" },
                  { step: "Deployment", color: "from-indigo-500 to-indigo-600", icon: "🚀", bgColor: "bg-indigo-100" }
                ].map((phase, index) => (
                  <motion.div
                    key={phase.step}
                    className="flex flex-col items-center relative"
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.3 }}
                  >
                    {/* Process Box */}
                    <motion.div
                      className={`w-20 h-20 bg-gradient-to-br ${phase.color} rounded-lg shadow-lg flex items-center justify-center text-white text-2xl font-bold relative overflow-hidden`}
                      whileHover={{ scale: 1.05, y: -5 }}
                      animate={{
                        boxShadow: [
                          "0 10px 25px rgba(0,0,0,0.1)",
                          "0 15px 35px rgba(0,0,0,0.2)",
                          "0 10px 25px rgba(0,0,0,0.1)"
                        ]
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        delay: index * 0.4
                      }}
                    >
                      <span className="text-2xl relative z-10">{phase.icon}</span>
                      
                      {/* Animated Pulse Effect */}
                      <motion.div
                        className="absolute inset-0 bg-white rounded-lg"
                        animate={{
                          scale: [1, 1.2, 1],
                          opacity: [0.3, 0, 0.3]
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          delay: index * 0.4
                        }}
                      />
                      
                      {/* Processing Particles */}
                      <motion.div
                        className="absolute inset-0"
                        animate={{
                          rotate: [0, 360]
                        }}
                        transition={{
                          duration: 8,
                          repeat: Infinity,
                          ease: "linear"
                        }}
                      >
                        {[...Array(6)].map((_, i) => (
                          <motion.div
                            key={i}
                            className="absolute w-1 h-1 bg-white rounded-full"
                            style={{
                              left: '50%',
                              top: '50%',
                              transform: 'translate(-50%, -50%)',
                              transformOrigin: '0 0'
                            }}
                            animate={{
                              x: [0, Math.cos(i * 60 * Math.PI / 180) * 30],
                              y: [0, Math.sin(i * 60 * Math.PI / 180) * 30],
                              opacity: [0, 1, 0]
                            }}
                            transition={{
                              duration: 2,
                              repeat: Infinity,
                              delay: i * 0.3 + index * 0.2
                            }}
                          />
                        ))}
                      </motion.div>
                    </motion.div>
                    
                    {/* Step Label */}
                    <motion.div
                      className="mt-3 text-center"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: index * 0.3 + 0.5 }}
                    >
                      <div className="text-sm font-semibold text-gray-700">{phase.step}</div>
                      <div className="text-xs text-gray-500 mt-1">Phase {index + 1}</div>
                    </motion.div>
                    
                    {/* Connection Arrow (except for last item) */}
                    {index < 4 && (
                      <motion.div
                        className="absolute top-10 left-full w-12 h-0.5 bg-gradient-to-r from-gray-300 to-gray-400"
                        style={{ left: 'calc(100% + 4px)' }}
                        initial={{ scaleX: 0 }}
                        animate={{ scaleX: 1 }}
                        transition={{ delay: index * 0.3 + 0.8 }}
                      >
                        {/* Moving Data Particles */}
                        <motion.div
                          className="absolute top-0 left-0 w-2 h-2 bg-blue-500 rounded-full"
                          animate={{
                            x: [0, 48],
                            opacity: [0, 1, 0]
                          }}
                          transition={{
                            duration: 1.5,
                            repeat: Infinity,
                            delay: index * 0.5
                          }}
                        />
                        <motion.div
                          className="absolute top-0 left-0 w-1 h-1 bg-green-500 rounded-full"
                          animate={{
                            x: [0, 48],
                            opacity: [0, 1, 0]
                          }}
                          transition={{
                            duration: 1.5,
                            repeat: Infinity,
                            delay: index * 0.5 + 0.3
                          }}
                        />
                        <motion.div
                          className="absolute top-0 left-0 w-1 h-1 bg-purple-500 rounded-full"
                          animate={{
                            x: [0, 48],
                            opacity: [0, 1, 0]
                          }}
                          transition={{
                            duration: 1.5,
                            repeat: Infinity,
                            delay: index * 0.5 + 0.6
                          }}
                        />
                      </motion.div>
                    )}
                  </motion.div>
                ))}
              </div>
              
              {/* Processing Flow Animation */}
              <motion.div
                className="mt-6 relative"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.5 }}
              >
                {/* Flow Line */}
                <motion.div
                  className="absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500 via-green-500 to-indigo-500"
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ duration: 2, delay: 1.8 }}
                />
                
                {/* Flow Particles */}
                {[...Array(8)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute top-1/2 w-2 h-2 bg-white rounded-full shadow-lg"
                    style={{ left: `${(i * 12.5)}%` }}
                    animate={{
                      y: [-10, 10, -10],
                      opacity: [0, 1, 0]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: i * 0.2 + 2
                    }}
                  />
                ))}
              </motion.div>
              
              {/* Status Bar */}
              <motion.div
                className="mt-6 bg-white rounded-lg p-4 border border-gray-200 relative z-10"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.5 }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <motion.div
                      className="w-3 h-3 bg-green-500 rounded-full"
                      animate={{
                        scale: [1, 1.5, 1],
                        opacity: [1, 0.5, 1]
                      }}
                      transition={{
                        duration: 1,
                        repeat: Infinity
                      }}
                    />
                    <span className="text-sm font-semibold text-gray-700">Processing Active</span>
                  </div>
                  <div className="text-sm text-gray-500">Ethical AI Pipeline Running</div>
                </div>
                
                {/* Progress Bar */}
                <motion.div
                  className="mt-3 h-1 bg-gray-200 rounded-full overflow-hidden"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 2 }}
                >
                  <motion.div
                    className="h-full bg-gradient-to-r from-blue-500 to-green-500"
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 3, delay: 2.2 }}
                  />
                </motion.div>
              </motion.div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;