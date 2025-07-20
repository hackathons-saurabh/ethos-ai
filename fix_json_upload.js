const fs = require('fs');

// Read the file
let content = fs.readFileSync('frontend/dashboard/src/App.js', 'utf8');

// Add parseJSON function if it doesn't exist
if (!content.includes('const parseJSON')) {
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
}

// Fix the onTrainingDataDrop function to handle JSON files properly
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

// Fix the onInputDataDrop function to handle JSON files properly
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

// Write the updated content back
fs.writeFileSync('frontend/dashboard/src/App.js', content);

console.log('✅ JSON file upload fixed successfully!'); 