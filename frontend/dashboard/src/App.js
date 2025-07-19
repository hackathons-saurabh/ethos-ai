// frontend/dashboard/src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

// Components
const BiasScoreIndicator = ({ score, label }) => {
  const getColorClass = (score) => {
    if (score < 0.3) return 'low-bias';
    if (score < 0.7) return 'medium-bias';
    return 'high-bias';
  };

  return (
    <div className="bias-indicator">
      <div className="bias-label">{label}</div>
      <div className="bias-bar-container">
        <div 
          className={`bias-bar ${getColorClass(score)}`}
          style={{ width: `${score * 100}%` }}
        />
      </div>
      <div className="bias-score">{(score * 100).toFixed(1)}%</div>
    </div>
  );
};

const DemoCard = ({ demo, onRun, isActive }) => {
  return (
    <div className={`demo-card ${isActive ? 'active' : ''}`}>
      <div className="demo-icon">{demo.icon}</div>
      <h3>{demo.title}</h3>
      <p>{demo.description}</p>
      <button onClick={() => onRun(demo)} className="run-demo-btn">
        Run Demo
      </button>
    </div>
  );
};

const ComparisonPanel = ({ results }) => {
  if (!results) return null;

  return (
    <div className="comparison-panel">
      <div className="comparison-column without-ethos">
        <h3>❌ Without Ethos</h3>
        <div className="result-content">
          <p className="prediction">{results.without.prediction}</p>
          <BiasScoreIndicator score={results.without.biasScore} label="Bias Score" />
          <div className="warning">⚠️ {results.without.warning}</div>
        </div>
      </div>
      
      <div className="comparison-column with-ethos">
        <h3>✅ With Ethos</h3>
        <div className="result-content">
          <p className="prediction">{results.with.prediction}</p>
          <BiasScoreIndicator score={results.with.biasScore} label="Bias Score" />
          <div className="success">✓ {results.with.success}</div>
        </div>
      </div>
    </div>
  );
};

const ChatInterface = ({ scenario, ethosEnabled }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = {
      text: input,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages([...messages, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: input,
        scenario: scenario,
        ethos_enabled: ethosEnabled
      });

      const botMessage = {
        text: response.data.response,
        sender: 'bot',
        timestamp: response.data.timestamp,
        biasScore: response.data.bias_score,
        ethosEnabled: response.data.ethos_enabled
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender}`}>
            <div className="message-content">{msg.text}</div>
            {msg.biasScore !== undefined && (
              <div className="message-meta">
                <span className={`bias-badge ${msg.ethosEnabled ? 'fair' : 'biased'}`}>
                  Bias: {(msg.biasScore * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>
        ))}
        {isLoading && <div className="typing-indicator">AI is thinking...</div>}
      </div>
      
      <div className="chat-input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask a question..."
          className="chat-input"
        />
        <button onClick={sendMessage} className="send-btn">Send</button>
      </div>
    </div>
  );
};

const MCPStatusPanel = ({ isProcessing }) => {
  const servers = [
    { name: 'bias-detector', status: isProcessing ? 'active' : 'idle' },
    { name: 'data-cleaner', status: isProcessing ? 'active' : 'idle' },
    { name: 'fairness-evaluator', status: isProcessing ? 'active' : 'idle' },
    { name: 'compliance-logger', status: isProcessing ? 'active' : 'idle' },
    { name: 'prediction-server', status: isProcessing ? 'active' : 'idle' }
  ];

  return (
    <div className="mcp-status-panel">
      <h3>MCP Pipeline Status</h3>
      <div className="server-list">
        {servers.map((server, idx) => (
          <div key={idx} className={`server-item ${server.status}`}>
            <div className={`status-dot ${server.status}`} />
            <span>{server.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [activeDemo, setActiveDemo] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [comparatorMode, setComparatorMode] = useState(true);
  const [ethosEnabled, setEthosEnabled] = useState(true);
  const [showChat, setShowChat] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState(null);

  const demos = [
    {
      id: 'hiring',
      title: 'ML Hiring Bias',
      icon: '📊',
      description: 'Fix gender bias in hiring predictions',
      scenario: 'hiring'
    },
    {
      id: 'support',
      title: 'Support Bot Bias',
      icon: '🛡️',
      description: 'Remove stereotypes from customer service',
      scenario: 'support'
    },
    {
      id: 'llm',
      title: 'LLM Toxicity',
      icon: '⚡',
      description: 'Clean offensive language from AI',
      scenario: 'llm'
    }
  ];

  const runDemo = async (demo) => {
    setActiveDemo(demo.id);
    setIsProcessing(true);
    setResults(null);

    // Simulate processing with mock results
    setTimeout(() => {
      const mockResults = {
        without: {
          prediction: getMockPrediction(demo.id, false),
          biasScore: 0.85,
          warning: 'High bias detected - Legal risk!'
        },
        with: {
          prediction: getMockPrediction(demo.id, true),
          biasScore: 0.05,
          success: 'Fair & compliant - Ship it!'
        }
      };
      
      setResults(mockResults);
      setIsProcessing(false);
    }, 3000);
  };

  const getMockPrediction = (demoId, withEthos) => {
    const predictions = {
      hiring: {
        without: 'Female: 20% hire chance (MIT degree ignored)',
        with: 'Female: 87% hire chance (strong qualifications)'
      },
      support: {
        without: '"Women handle family bookings best"',
        with: '"Anyone can help with bookings"'
      },
      llm: {
        without: '"Tech bros are arrogant jerks"',
        with: '"Tech professionals are innovative"'
      }
    };

    return predictions[demoId][withEthos ? 'with' : 'without'];
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', file.name);
    formData.append('target_column', 'target');

    try {
      const response = await axios.post(`${API_BASE_URL}/upload/dataset`, formData);
      setUploadedDataset(response.data);
      console.log('Dataset uploaded:', response.data);
    } catch (error) {
      console.error('Upload error:', error);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo">🛡️</div>
            <div>
              <h1>Ethos AI</h1>
              <p className="tagline">Bias-Free Intelligence Platform</p>
            </div>
          </div>
          
          <div className="header-controls">
            <div className="toggle-container">
              <span>Comparator Mode</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={comparatorMode}
                  onChange={(e) => setComparatorMode(e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
            
            <div className="status-badge">
              <div className="pulse-dot" />
              <span>5 MCP Servers Active</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Demo Selection */}
        <section className="demo-section">
          <h2>Choose Your Demo</h2>
          <div className="demo-grid">
            {demos.map(demo => (
              <DemoCard
                key={demo.id}
                demo={demo}
                onRun={runDemo}
                isActive={activeDemo === demo.id}
              />
            ))}
          </div>
        </section>

        {/* File Upload */}
        <section className="upload-section">
          <div className="upload-container">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              id="file-upload"
              className="file-input"
            />
            <label htmlFor="file-upload" className="upload-label">
              📁 Upload Dataset (CSV)
            </label>
            {uploadedDataset && (
              <div className="upload-info">
                ✅ Uploaded: {uploadedDataset.dataset_info.name} 
                ({uploadedDataset.dataset_info.rows} rows)
              </div>
            )}
          </div>
        </section>

        {/* Results Display */}
        {activeDemo && (
          <section className="results-section">
            <div className="section-header">
              <h2>Demo Results</h2>
              <button 
                onClick={() => setShowChat(!showChat)}
                className="toggle-chat-btn"
              >
                {showChat ? 'Hide' : 'Show'} Chat Interface
              </button>
            </div>

            {/* MCP Status */}
            <MCPStatusPanel isProcessing={isProcessing} />

            {/* Comparison Results */}
            {comparatorMode && results && (
              <ComparisonPanel results={results} />
            )}

            {/* Chat Interface */}
            {showChat && (
              <div className="chat-section">
                <div className="chat-controls">
                  <label className="ethos-toggle">
                    <input
                      type="checkbox"
                      checked={ethosEnabled}
                      onChange={(e) => setEthosEnabled(e.target.checked)}
                    />
                    <span className="toggle-label">
                      {ethosEnabled ? '✅ Ethos ON' : '❌ Ethos OFF'}
                    </span>
                  </label>
                </div>
                <ChatInterface 
                  scenario={activeDemo} 
                  ethosEnabled={ethosEnabled}
                />
              </div>
            )}

            {/* Processing Animation */}
            {isProcessing && (
              <div className="processing-overlay">
                <div className="processing-spinner" />
                <p>Processing with MCP Pipeline...</p>
              </div>
            )}
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>Ethos AI - Eliminating Bias in AI Systems | Hackathon 2025</p>
      </footer>
    </div>
  );
}

export default App;