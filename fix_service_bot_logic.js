const fs = require('fs');

// Read the file
let content = fs.readFileSync('frontend/dashboard/src/App.js', 'utf8');

// Fix the generatePrediction function to handle Track 2 properly
const newGeneratePrediction = `
  const generatePrediction = async (question, inputData, trainingData, ethosEnabled) => {
    // Track 2: Service Bot Responses
    if (currentTrack === 'service') {
      const questionLower = question.toLowerCase();
      
      // Try to parse the question as JSON to get user demographics
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
      }
      
      // Biased responses (without Ethos.ai) - consider user demographics
      const getBiasedResponse = (type, userDemo) => {
        const responses = {
          'password': userDemo ? 
            \`I can help you reset your password. Since you're a \${userDemo.gender || 'user'} aged \${userDemo.age || 'unknown'}, I'll provide \${userDemo.gender === 'female' ? 'extra patient' : 'standard'} assistance. What's your education level?\` :
            'I can help you reset your password. First, please tell me your age and gender so I can provide appropriate assistance.',
          'account': userDemo ?
            \`I can help with your account. Since you're a \${userDemo.gender || 'user'} with \${userDemo.income_level || 'unknown'} income, I'll provide \${userDemo.income_level === 'high' ? 'premium' : 'standard'} service. Are you a native English speaker?\` :
            'I can help with your account. Are you a native English speaker? This will help me provide better service.',
          'return': userDemo ?
            \`Our return policy depends on your profile. As a \${userDemo.gender || 'customer'} aged \${userDemo.age || 'unknown'} with \${userDemo.income_level || 'standard'} income, you qualify for \${userDemo.income_level === 'high' ? 'premium' : 'standard'} return terms. What is your ethnicity?\` :
            'Our return policy depends on your location and demographic. What is your ethnicity and age?',
          'help': userDemo ?
            \`I can help you. As a \${userDemo.gender || 'user'} with \${userDemo.income_level || 'standard'} income, I'll provide \${userDemo.income_level === 'high' ? 'VIP' : 'standard'} assistance. What's your education level?\` :
            'I can help you. First, let me know your education level and income bracket for personalized assistance.',
          'support': userDemo ?
            \`I can provide support. Since you're a \${userDemo.gender || 'user'} aged \${userDemo.age || 'unknown'}, I'll offer \${userDemo.age > 50 ? 'senior' : 'standard'} support. What's your marital status?\` :
            'I can provide support. Please share your marital status and occupation for better service.',
          'policy': userDemo ?
            \`Our policies vary by customer profile. As a \${userDemo.gender || 'customer'} with \${userDemo.income_level || 'standard'} income, you have \${userDemo.income_level === 'high' ? 'premium' : 'standard'} access. What's your location?\` :
            'Our policies vary by customer profile. What is your age, gender, and location?',
          'service': userDemo ?
            \`I can assist you. Based on your profile as a \${userDemo.gender || 'user'} with \${userDemo.income_level || 'standard'} income, I'll provide \${userDemo.income_level === 'high' ? 'premium' : 'standard'} service. What's your occupation?\` :
            'I can assist you. Please provide your age, gender, and income level for customized help.',
          'general': userDemo ?
            \`I can help you. As a \${userDemo.gender || 'user'} aged \${userDemo.age || 'unknown'} with \${userDemo.income_level || 'standard'} income, I'll provide \${userDemo.income_level === 'high' ? 'VIP' : 'standard'} assistance. What's your ethnicity?\` :
            'I can help you. First, please tell me your age, gender, and ethnicity for personalized service.'
        };
        return responses[type] || responses['general'];
      };
      
      // Ethical responses (with Ethos.ai) - ignore demographics
      const getEthicalResponse = (type) => {
        const responses = {
          'password': 'I can help you reset your password. Please provide your email address or username associated with your account.',
          'account': 'I can help with your account. Please provide your account number or email address.',
          'return': 'Our return policy allows returns within 30 days with original receipt. What item would you like to return?',
          'help': 'I can help you. What specific issue are you experiencing?',
          'support': 'I can provide support. What technical issue are you facing?',
          'policy': 'Our policies are available on our website. What specific policy information do you need?',
          'service': 'I can assist you. What service do you need help with?',
          'general': 'I can help you. What can I assist you with today?'
        };
        return responses[type] || responses['general'];
      };
      
      // Determine response type based on actual question
      const actualQuestionLower = actualQuestion.toLowerCase();
      let responseType = 'general';
      if (actualQuestionLower.includes('password') || actualQuestionLower.includes('reset')) {
        responseType = 'password';
      } else if (actualQuestionLower.includes('account')) {
        responseType = 'account';
      } else if (actualQuestionLower.includes('return') || actualQuestionLower.includes('refund')) {
        responseType = 'return';
      } else if (actualQuestionLower.includes('help')) {
        responseType = 'help';
      } else if (actualQuestionLower.includes('support')) {
        responseType = 'support';
      } else if (actualQuestionLower.includes('policy') || actualQuestionLower.includes('policies')) {
        responseType = 'policy';
      } else if (actualQuestionLower.includes('service')) {
        responseType = 'service';
      }
      
      // Return appropriate response based on Ethos.ai setting
      if (ethosEnabled) {
        return getEthicalResponse(responseType);
      } else {
        return getBiasedResponse(responseType, userDemo);
      }
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

// Replace the entire generatePrediction function
content = content.replace(
  /const generatePrediction = async \(question, inputData, trainingData, ethosEnabled\) => \{[\s\S]*?\};/,
  newGeneratePrediction
);

// Fix handleAskQuestion to work for Track 2
content = content.replace(
  /if \(!inputMessage\.trim\(\) \|\| !inputData \|\| !trainingData\) return;/,
  "if (!inputMessage.trim() || !trainingData || (currentTrack === 'ml' && !inputData)) return;"
);

content = content.replace(
  /const response = await generatePrediction\(inputMessage, inputData, trainingData, ethosEnabled\);/,
  "const response = await generatePrediction(inputMessage, inputData || trainingData, trainingData, ethosEnabled);"
);

// Write the updated content back
fs.writeFileSync('frontend/dashboard/src/App.js', content);

console.log('✅ Service bot logic fixed successfully!'); 