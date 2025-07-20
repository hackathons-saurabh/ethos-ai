const fs = require('fs');

// Read the file
let content = fs.readFileSync('frontend/dashboard/src/App.js', 'utf8');

// Add currentTrack variable if missing
if (!content.includes('currentTrack')) {
  content = content.replace(
    /const \[hasTrainedWithEthos, setHasTrainedWithEthos\] = useState\(false\);/,
    `const [hasTrainedWithEthos, setHasTrainedWithEthos] = useState(false);
  const [currentTrack, setCurrentTrack] = useState("ml");`
  );
}

// Fix training dropzone to accept JSON
content = content.replace(
  /accept: \{ 'text\/csv': \['\.csv'\] \}/,
  "accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] }"
);

// Fix input dropzone to accept JSON
content = content.replace(
  /const inputDropzone = useDropzone\(\{\s+onDrop: onInputDataDrop,\s+accept: \{ 'text\/csv': \['\.csv'\] \},\s+multiple: false\s+\}\);/,
  `const inputDropzone = useDropzone({
    onDrop: onInputDataDrop,
    accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] },
    multiple: false
  });`
);

// Add JSON parsing function
const jsonParseFunction = `
  const parseJSON = (jsonText) => {
    try {
      const data = JSON.parse(jsonText);
      return data;
    } catch (error) {
      return null;
    }
  };
`;

// Add JSON parsing function after parseCSV
content = content.replace(
  /const parseCSV = \(csvText\) => \{[\s\S]*?\};/,
  `const parseCSV = (csvText) => {
    const lines = csvText.trim().split('\\n');
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

  ${jsonParseFunction}
`
);

// Add JSON detection in onTrainingDataDrop
const jsonDetectionLogic = `
          // Detect track based on file type
          setCurrentTrack(file.name.endsWith('.json') ? 'service' : 'ml');
`;

// Add JSON detection after context detection
content = content.replace(
  /const context = detectDataContext\(parsed\.headers, parsed\.data\);/,
  `const context = detectDataContext(parsed.headers, parsed.data);
          setCurrentTrack(file.name.endsWith('.json') ? 'service' : 'ml');`
);

// Fix upload text
content = content.replace(
  /CSV files only/g,
  "CSV files for ML predictions, JSON files for service bots"
);

// Fix button text
content = content.replace(
  /{isTraining \? 'Training\.\.\.' : 'Train ML Model on this Data'}/g,
  "{isTraining ? 'Training...' : currentTrack === 'service' ? 'Train Support Bot on this Data' : 'Train ML Model on this Data'}"
);

// Fix input data upload panel to only show for Track 1
content = content.replace(
  /{trainingComplete && !inputData && \(/g,
  "{trainingComplete && !inputData && currentTrack === 'ml' && ("
);

// Fix Q&A interface to show for Track 2 or when input data is available
content = content.replace(
  /{trainingComplete && inputData && \(/g,
  "{trainingComplete && (currentTrack === 'service' || inputData) && ("
);

// Write the updated content back
fs.writeFileSync('frontend/dashboard/src/App.js', content);

console.log('✅ JSON upload support restored successfully!'); 