const fs = require('fs');

// Read the file
let content = fs.readFileSync('frontend/dashboard/src/App.js', 'utf8');

// Add Track 2 service bot response logic
const track2ServiceBotLogic = `
  // Track 2: Service Bot Responses
  const generateServiceBotResponse = (question, trainingData, ethosEnabled) => {
    const questionLower = question.toLowerCase();
    
    // Biased responses (without Ethos.ai)
    const biasedResponses = {
      'password': 'I can help you reset your password. First, please tell me your age and gender so I can provide appropriate assistance.',
      'account': 'I can help with your account. Are you a native English speaker? This will help me provide better service.',
      'return': 'Our return policy depends on your location and demographic. What is your ethnicity and age?',
      'help': 'I can help you. First, let me know your education level and income bracket for personalized assistance.',
      'support': 'I can provide support. Please share your marital status and occupation for better service.',
      'policy': 'Our policies vary by customer profile. What is your age, gender, and location?',
      'service': 'I can assist you. Please provide your age, gender, and income level for customized help.',
      'general': 'I can help you. First, please tell me your age, gender, and ethnicity for personalized service.'
    };
    
    // Ethical responses (with Ethos.ai)
    const ethicalResponses = {
      'password': 'I can help you reset your password. Please provide your email address or username associated with your account.',
      'account': 'I can help with your account. Please provide your account number or email address.',
      'return': 'Our return policy allows returns within 30 days with original receipt. What item would you like to return?',
      'help': 'I can help you. What specific issue are you experiencing?',
      'support': 'I can provide support. What technical issue are you facing?',
      'policy': 'Our policies are available on our website. What specific policy information do you need?',
      'service': 'I can assist you. What service do you need help with?',
      'general': 'I can help you. What can I assist you with today?'
    };
    
    // Determine response type based on question
    let responseType = 'general';
    if (questionLower.includes('password') || questionLower.includes('reset')) {
      responseType = 'password';
    } else if (questionLower.includes('account')) {
      responseType = 'account';
    } else if (questionLower.includes('return') || questionLower.includes('refund')) {
      responseType = 'return';
    } else if (questionLower.includes('help')) {
      responseType = 'help';
    } else if (questionLower.includes('support')) {
      responseType = 'support';
    } else if (questionLower.includes('policy') || questionLower.includes('policies')) {
      responseType = 'policy';
    } else if (questionLower.includes('service')) {
      responseType = 'service';
    }
    
    const responses = ethosEnabled ? ethicalResponses : biasedResponses;
    return responses[responseType];
  };
`;

// Replace the generatePrediction function to handle both tracks
const newGeneratePrediction = `
  const generatePrediction = async (question, inputData, trainingData, ethosEnabled) => {
    // Track 2: Service Bot Logic
    if (currentTrack === 'service') {
      return generateServiceBotResponse(question, trainingData, ethosEnabled);
    }
    
    // Track 1: ML Prediction Logic (existing code)
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
      
      let resultText = \`**\${ethosEnabled ? 'With' : 'Without'} Ethos.ai**\n\n\${response.title}:\n\`;
      
      results.forEach((row, index) => {
        const keyFields = headers.slice(0, 4).map(h => \`\${h}: \${row[h]}\`).join(', ');
        resultText += \`• \${response.itemPrefix} \${index + 1} (\${keyFields})\n  Reason: \${ethosEnabled ? response.ethicalReason : response.biasedReason}\n\`;
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
      return \`**Bias Analysis**\n\nDetected potential bias factors: \${biasFactors.join(', ')}\n\n\${ethosEnabled ? 'Ethos.ai has mitigated these biases through ethical processing.' : 'Without Ethos.ai, these biases may influence predictions.'}\`;
    } else if (questionLower.includes('data') || questionLower.includes('info') || questionLower.includes('summary')) {
      return \`**Dataset Information**\n\nType: \${context.type}\nRows: \${inputRows.length}\nColumns: \${headers.length}\nTarget: \${findTargetColumn()}\n\nAsk me to predict outcomes or analyze bias!\`;
    } else {
      return \`I can help with \${context.type} predictions. Try asking:\n• "What are the top predictions?"\n• "Show me the best candidates"\n• "Analyze bias in the data"\n• "Give me a summary"\`;
    }
  };
`;

// Add the service bot logic before generatePrediction
content = content.replace(
  /const generatePrediction = async \(question, inputData, trainingData, ethosEnabled\) => \{/,
  track2ServiceBotLogic + '\n' + newGeneratePrediction
);

// Write the updated content back
fs.writeFileSync('frontend/dashboard/src/App.js', content);

console.log('✅ Track 2 service bot responses fixed successfully!'); 