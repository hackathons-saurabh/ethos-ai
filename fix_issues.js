const fs = require('fs');

// Read the file
let content = fs.readFileSync('frontend/dashboard/src/App.js', 'utf8');

// Fix dropzone configuration to accept both CSV and JSON
content = content.replace(
  /accept: \{ 'text\/csv': \['\.csv'\] \}/g,
  "accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] }"
);

// Fix input dropzone configuration
content = content.replace(
  /const inputDropzone = useDropzone\(\{\s+onDrop: onInputDataDrop,\s+accept: \{ 'text\/csv': \['\.csv'\] \},\s+multiple: false\s+\}\);/g,
  `const inputDropzone = useDropzone({
    onDrop: onInputDataDrop,
    accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] },
    multiple: false
  });`
);

// Add JSON parsing functionality
const jsonParseFunction = `
  const parseJSON = (jsonText) => {
    try {
      const data = JSON.parse(jsonText);
      return data;
    } catch (error) {
      return null;
    }
  };

  const detectDataContext = (headers, data) => {
`;

// Replace the existing detectDataContext with JSON support
content = content.replace(
  /const detectDataContext = \(headers, data\) => \{/g,
  `const detectDataContext = (parsed, fileType) => {
    if (fileType === 'json') {
      // Generic JSON detection - works with any service logs
      const hasConversations = parsed.conversations && Array.isArray(parsed.conversations);
      const hasQueries = parsed.conversations?.[0]?.query;
      const hasResponses = parsed.conversations?.[0]?.raw_response;
      
      if (hasConversations && hasQueries && hasResponses) {
        return {
          type: 'service',
          description: 'Service Support Bot Training Dataset',
          capabilities: ['Train customer service bots', 'Remove stereotypes and bias', 'Generate ethical responses'],
          targetColumn: 'raw_response',
          dataType: 'conversations'
        };
      }
      
      // Generic JSON fallback
      return {
        type: 'general',
        description: 'General JSON Dataset',
        capabilities: ['Process JSON data', 'Analyze patterns', 'Generate insights'],
        targetColumn: 'data',
        dataType: 'json'
      };
    }

    // Generic CSV detection - works with any CSV structure
    const headers = parsed.headers;
    const data = parsed.data;
    
    // Analyze data structure dynamically
    const hasNumericData = headers.some(h => {
      const sampleValues = data.slice(0, 5).map(row => row[h]);
      return sampleValues.some(val => !isNaN(parseFloat(val)) && val !== '');
    });
    
    const hasCategoricalData = headers.some(h => {
      const sampleValues = data.slice(0, 5).map(row => row[h]);
      return sampleValues.some(val => isNaN(parseFloat(val)) && val !== '');
    });
    
    // Generic description based on data structure
    let description = 'General Dataset';
    let capabilities = ['Analyze patterns', 'Generate insights'];
    
    if (hasNumericData && hasCategoricalData) {
      description = 'Mixed Data Dataset';
      capabilities = ['Predict outcomes', 'Analyze correlations', 'Identify patterns'];
    } else if (hasNumericData) {
      description = 'Numeric Dataset';
      capabilities = ['Predict values', 'Analyze trends', 'Generate insights'];
    } else if (hasCategoricalData) {
      description = 'Categorical Dataset';
      capabilities = ['Classify data', 'Analyze categories', 'Generate insights'];
    }
    
    return {
      type: 'general',
      description,
      capabilities,
      targetColumn: headers[headers.length - 1] || headers[0]
    };
  };`
);

// Add JSON parsing function before detectDataContext
content = content.replace(
  /const detectDataContext = \(parsed, fileType\) => \{/g,
  `const parseJSON = (jsonText) => {
    try {
      const data = JSON.parse(jsonText);
      return data;
    } catch (error) {
      return null;
    }
  };

  const detectDataContext = (parsed, fileType) => {`
);

// Update the onTrainingDataDrop function to handle both CSV and JSON
const newTrainingDropFunction = `
  const onTrainingDataDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        let parsed = null;
        let fileType = 'csv';
        let context = null;

        if (file.name.endsWith('.json')) {
          parsed = parseJSON(content);
          fileType = 'json';
          if (parsed) {
            context = detectDataContext(parsed, 'json');
            setCurrentTrack('service');
          }
        } else {
          parsed = parseCSV(content);
          if (parsed) {
            context = detectDataContext(parsed, 'csv');
            setCurrentTrack('ml');
          }
        }

        if (parsed && context) {
          setTrainingData({ file, parsed, context });
          setDataSummary({
            fileName: file.name,
            fileType: fileType.toUpperCase(),
            description: context.description,
            capabilities: context.capabilities,
            targetColumn: context.targetColumn,
            dataSize: fileType === 'json' ? (parsed.conversations?.length || 0) : parsed.data.length,
            features: fileType === 'json' ? Object.keys(parsed.conversations?.[0] || {}).length : parsed.headers.length
          });
          // Reset to Without Ethos.ai when file is uploaded
          setEthosEnabled(false);
          setShowProcessReactor(false);
          setHasTrainedWithEthos(false);
        }
      };
      reader.readAsText(file);
    }
  }, []);`;

// Replace the existing onTrainingDataDrop function
content = content.replace(
  /const onTrainingDataDrop = useCallback\(\(acceptedFiles\) => \{[\s\S]*?\}, \[\]\);/g,
  newTrainingDropFunction
);

// Write the fixed content
fs.writeFileSync('frontend/dashboard/src/App.js', content);
console.log('All issues fixed successfully!');
