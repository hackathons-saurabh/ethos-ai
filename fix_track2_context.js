const fs = require('fs');

// Read the file
let content = fs.readFileSync('frontend/dashboard/src/App.js', 'utf8');

// Add JSON parsing function
const jsonParseFunction = `
  const parseJSON = (jsonText) => {
    try {
      const data = JSON.parse(jsonText);
      return { headers: Object.keys(data[0] || {}), data: data };
    } catch (e) {
      return null;
    }
  };

`;

// Add the JSON parsing function after parseCSV
content = content.replace(
  /const parseCSV = \(csvText\) => \{[\s\S]*?\};/,
  `const parseCSV = (csvText) => {
    const lines = csvText.split('\\n').filter(line => line.trim());
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

  const parseJSON = (jsonText) => {
    try {
      const data = JSON.parse(jsonText);
      return { headers: Object.keys(data[0] || {}), data: data };
    } catch (e) {
      return null;
    }
  };`
);

// Fix the detectDataContext function to handle JSON files
content = content.replace(
  /const detectDataContext = \(headers, data\) => \{[\s\S]*?\};/,
  `const detectDataContext = (headers, data, isJSON = false) => {
    if (isJSON) {
      return {
        type: 'service',
        description: 'Customer Service Support Bot',
        capabilities: ['Handle customer inquiries', 'Provide support assistance', 'Process service requests'],
        targetColumn: 'response'
      };
    }
    
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
  };`
);

// Fix the onTrainingDataDrop function to handle JSON files
content = content.replace(
  /const onTrainingDataDrop = useCallback\(\(acceptedFiles\) => \{[\s\S]*?\}, \[\]\);/,
  `const onTrainingDataDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const fileContent = e.target.result;
        const isJSON = file.name.endsWith('.json');
        
        let parsed;
        if (isJSON) {
          parsed = parseJSON(fileContent);
        } else {
          parsed = parseCSV(fileContent);
        }
        
        if (parsed) {
          const context = detectDataContext(parsed.headers, parsed.data, isJSON);
          setCurrentTrack(isJSON ? 'service' : 'ml');
          setTrainingData({ file, parsed, context });
          setDataSummary({
            fileName: file.name,
            fileType: isJSON ? 'JSON' : 'CSV',
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
  }, []);`
);

// Fix the onInputDataDrop function to handle JSON files
content = content.replace(
  /const onInputDataDrop = useCallback\(\(acceptedFiles\) => \{[\s\S]*?\}, \[\]\);/,
  `const onInputDataDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const fileContent = e.target.result;
        const isJSON = file.name.endsWith('.json');
        
        let parsed;
        if (isJSON) {
          parsed = parseJSON(fileContent);
        } else {
          parsed = parseCSV(fileContent);
        }
        
        if (parsed) {
          setInputData({ file, parsed });
        }
      };
      reader.readAsText(file);
    }
  }, []);`
);

// Fix the service bot logic to handle JSON parsing better
content = content.replace(
  /\/\/ Try to parse the question as JSON to get user demographics[\s\S]*?\/\/ Question is not JSON, use as is[\s\S]*?}/,
  `// Try to parse the question as JSON to get user demographics
      let userDemo = null;
      let actualQuestion = question;
      try {
        const parsedQuestion = JSON.parse(question);
        if (parsedQuestion.user_demo) {
          userDemo = parsedQuestion.user_demo;
          actualQuestion = parsedQuestion.query || question;
        }
      } catch (e) {
        // Question is not JSON, use as is
        actualQuestion = question;
      }`
);

// Write the updated content back
fs.writeFileSync('frontend/dashboard/src/App.js', content);

console.log('✅ Track 2 context and JSON parsing fixed successfully!'); 