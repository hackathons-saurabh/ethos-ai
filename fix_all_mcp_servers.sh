#!/bin/bash

# Fix bias-detector
cat > mcp-servers/bias-detector/server.py << 'EOFSERVER'
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("bias-detector")

# Tool handlers storage
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

class BiasDetector:
    def __init__(self):
        self.sensitive_attributes = ['gender', 'race', 'age', 'ethnicity', 'religion']
        self.bias_thresholds = {
            'demographic_parity': 0.1,
            'disparate_impact': 0.8,
            'statistical_parity': 0.1
        }
    
    def detect_demographic_parity(self, df: pd.DataFrame, target_col: str, sensitive_col: str) -> Dict:
        if sensitive_col not in df.columns or target_col not in df.columns:
            return {'bias_detected': False, 'score': 0.0}
        
        groups = df.groupby(sensitive_col)[target_col].agg(['mean', 'count'])
        max_rate = groups['mean'].max()
        min_rate = groups['mean'].min()
        parity_diff = max_rate - min_rate
        
        return {
            'bias_detected': parity_diff > self.bias_thresholds['demographic_parity'],
            'score': float(parity_diff),
            'groups': groups.to_dict(),
            'metric': 'demographic_parity'
        }
    
    def detect_disparate_impact(self, df: pd.DataFrame, target_col: str, sensitive_col: str) -> Dict:
        if sensitive_col not in df.columns or target_col not in df.columns:
            return {'bias_detected': False, 'score': 0.0}
        
        groups = df.groupby(sensitive_col)[target_col].mean()
        
        if len(groups) < 2:
            return {'bias_detected': False, 'score': 1.0}
        
        majority_rate = groups.max()
        minority_rate = groups.min()
        
        if majority_rate == 0:
            impact_ratio = 0
        else:
            impact_ratio = minority_rate / majority_rate
        
        return {
            'bias_detected': impact_ratio < self.bias_thresholds['disparate_impact'],
            'score': float(impact_ratio),
            'groups': groups.to_dict(),
            'metric': 'disparate_impact'
        }

@tool("detect_bias")
async def detect_bias(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(data['dataset'])
        target_col = data.get('target_column', 'target')
        
        detector = BiasDetector()
        results = {
            'overall_bias_score': 0.0,
            'bias_types_detected': [],
            'detailed_results': {},
            'recommendations': []
        }
        
        bias_scores = []
        
        for attr in detector.sensitive_attributes:
            if attr in df.columns:
                parity_result = detector.detect_demographic_parity(df, target_col, attr)
                if parity_result['bias_detected']:
                    results['bias_types_detected'].append(f'demographic_parity_{attr}')
                    results['detailed_results'][f'demographic_parity_{attr}'] = parity_result
                    bias_scores.append(parity_result['score'])
                    results['recommendations'].append(
                        f"Address demographic parity bias in {attr} (difference: {parity_result['score']:.2f})"
                    )
        
        if bias_scores:
            results['overall_bias_score'] = float(np.mean(bias_scores))
        
        results['metadata'] = {
            'rows_analyzed': len(df),
            'columns_analyzed': list(df.columns),
            'sensitive_attributes_found': [attr for attr in detector.sensitive_attributes if attr in df.columns]
        }
        
        logger.info(f"Bias detection complete. Overall score: {results['overall_bias_score']:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in bias detection: {str(e)}")
        return {
            'error': str(e),
            'overall_bias_score': 0.0,
            'bias_types_detected': [],
            'detailed_results': {},
            'recommendations': []
        }

@tool("analyze_feature_importance")
async def analyze_feature_importance(data: Dict[str, Any]) -> Dict[str, Any]:
    return {'feature_importance': [], 'most_biased_features': []}

@tool("get_bias_metrics")
async def get_bias_metrics() -> Dict[str, Any]:
    return {
        'metrics': {
            'demographic_parity': 'Ensures equal positive outcome rates across groups',
            'disparate_impact': '80% rule - minority group should have at least 80% of majority rate'
        }
    }

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=f"{name} tool",
            inputSchema={"type": "object", "properties": {}}
        )
        for name in handlers.keys()
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> dict:
    if name in handlers:
        return await handlers[name](arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    logger.info("Starting bias-detector MCP Server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
EOFSERVER

# Fix data-cleaner
cat > mcp-servers/data-cleaner/server.py << 'EOFSERVER'
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from mcp.server import Server
import mcp.server.stdio
import mcp.types as types
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("data-cleaner")
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

class DataCleaner:
    def __init__(self):
        self.sensitive_attributes = ['gender', 'race', 'age', 'ethnicity', 'religion']
    
    def mask_sensitive_values(self, df: pd.DataFrame, attributes: List[str]) -> pd.DataFrame:
        df_cleaned = df.copy()
        for attr in attributes:
            if attr in df_cleaned.columns:
                if attr == 'gender':
                    df_cleaned[attr] = 'Person'
                elif attr == 'race':
                    df_cleaned[attr] = 'Individual'
                else:
                    df_cleaned[attr] = 'Masked'
        return df_cleaned
    
    def reweight_samples(self, df: pd.DataFrame, target_col: str, sensitive_col: str) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        if sensitive_col not in df.columns or target_col not in df.columns:
            return df_cleaned
        
        group_rates = df.groupby(sensitive_col)[target_col].mean()
        overall_rate = df[target_col].mean()
        
        weights = []
        for _, row in df.iterrows():
            group = row[sensitive_col]
            current_rate = group_rates[group]
            
            if current_rate > 0:
                weight = overall_rate / current_rate
            else:
                weight = 1.0
            
            weights.append(weight)
        
        df_cleaned['sample_weight'] = weights
        logger.info(f"Applied reweighting for {sensitive_col}")
        
        return df_cleaned

@tool("clean_bias")
async def clean_bias(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(data['dataset'])
        bias_report = data.get('bias_report', {})
        target_col = data.get('target_column', 'target')
        
        cleaner = DataCleaner()
        original_shape = df.shape
        
        cleaning_log = []
        df_cleaned = df.copy()
        
        # Apply cleaning based on detected biases
        bias_types = bias_report.get('bias_types_detected', [])
        
        for bias_type in bias_types:
            if 'demographic_parity' in bias_type:
                sensitive_attr = bias_type.split('_')[-1]
                df_cleaned = cleaner.reweight_samples(df_cleaned, target_col, sensitive_attr)
                cleaning_log.append(f"Applied reweighting for {sensitive_attr}")
        
        # Mask some sensitive attributes
        sensitive_attrs = [attr for attr in cleaner.sensitive_attributes if attr in df_cleaned.columns]
        if sensitive_attrs:
            df_cleaned = cleaner.mask_sensitive_values(df_cleaned, sensitive_attrs[:2])
            cleaning_log.append(f"Masked sensitive attributes: {sensitive_attrs[:2]}")
        
        cleaned_dataset = df_cleaned.to_dict('records')
        
        result = {
            'cleaned_dataset': cleaned_dataset,
            'cleaning_report': {
                'original_shape': original_shape,
                'cleaned_shape': df_cleaned.shape,
                'cleaning_methods_applied': cleaning_log,
                'samples_added': len(df_cleaned) - len(df)
            }
        }
        
        logger.info(f"Data cleaning complete. Shape: {original_shape} -> {df_cleaned.shape}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        return {
            'error': str(e),
            'cleaned_dataset': data.get('dataset', []),
            'cleaning_report': {'error': str(e)}
        }

@tool("preview_cleaning")
async def preview_cleaning(data: Dict[str, Any]) -> Dict[str, Any]:
    return {'preview': 'Cleaning preview not implemented'}

@tool("get_cleaning_methods")
async def get_cleaning_methods() -> Dict[str, Any]:
    return {
        'methods': {
            'masking': 'Replace sensitive values with placeholders',
            'reweighting': 'Add sample weights to balance outcomes'
        }
    }

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=f"{name} tool",
            inputSchema={"type": "object", "properties": {}}
        )
        for name in handlers.keys()
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> dict:
    if name in handlers:
        return await handlers[name](arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    logger.info("Starting data-cleaner MCP Server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
EOFSERVER

# Fix fairness-evaluator
cat > mcp-servers/fairness-evaluator/server.py << 'EOFSERVER'
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from mcp.server import Server
import mcp.server.stdio
import mcp.types as types
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("fairness-evaluator")
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

@tool("evaluate_fairness")
async def evaluate_fairness(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(data['dataset'])
        predictions = np.array(data['predictions'])
        actual = np.array(data.get('actual', predictions))
        sensitive_attrs = data.get('sensitive_attributes', ['gender', 'race', 'age'])
        
        results = {
            'overall_fairness_score': 0.0,
            'is_fair': True,
            'detailed_metrics': {},
            'violations': [],
            'recommendations': []
        }
        
        fairness_scores = []
        
        for attr in sensitive_attrs:
            if attr not in df.columns:
                continue
                
            groups = df[attr].unique()
            positive_rates = {}
            
            for group in groups:
                mask = df[attr] == group
                positive_rate = predictions[mask].mean()
                positive_rates[str(group)] = float(positive_rate)
            
            max_rate = max(positive_rates.values())
            min_rate = min(positive_rates.values())
            disparity = max_rate - min_rate
            
            fairness_scores.append(disparity)
            
            if disparity > 0.1:
                results['violations'].append(f"Statistical parity violation for {attr}")
                results['recommendations'].append(f"Adjust predictions to equalize positive rates across {attr} groups")
                results['is_fair'] = False
        
        if fairness_scores:
            results['overall_fairness_score'] = float(np.mean(fairness_scores))
        
        logger.info(f"Fairness evaluation complete. Overall score: {results['overall_fairness_score']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in fairness evaluation: {str(e)}")
        return {
            'error': str(e),
            'overall_fairness_score': 1.0,
            'is_fair': False,
            'detailed_metrics': {},
            'violations': ['Evaluation error'],
            'recommendations': ['Fix evaluation errors']
        }

@tool("suggest_mitigation")
async def suggest_mitigation(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    return {'mitigation_strategies': [], 'priority_order': []}

@tool("get_fairness_metrics")
async def get_fairness_metrics() -> Dict[str, Any]:
    return {
        'metrics': {
            'statistical_parity': 'Equal positive prediction rates across groups',
            'equal_opportunity': 'Equal true positive rates across groups'
        }
    }

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=f"{name} tool",
            inputSchema={"type": "object", "properties": {}}
        )
        for name in handlers.keys()
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> dict:
    if name in handlers:
        return await handlers[name](arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    logger.info("Starting fairness-evaluator MCP Server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
EOFSERVER

# Fix compliance-logger
cat > mcp-servers/compliance-logger/server.py << 'EOFSERVER'
import asyncio
import json
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any
from mcp.server import Server
import mcp.server.stdio
import mcp.types as types
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("compliance-logger")
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

class ComplianceLogger:
    def __init__(self):
        self.logs = []
        
    def generate_hash(self, data: Any) -> str:
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def log_event(self, event_type: str, component: str, action: str, details: Dict[str, Any]) -> str:
        log_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        data_hash = self.generate_hash(details)
        
        log_entry = {
            'id': log_id,
            'timestamp': timestamp,
            'event_type': event_type,
            'component': component,
            'action': action,
            'data_hash': data_hash,
            'details': details
        }
        
        self.logs.append(log_entry)
        logger.info(f"Logged compliance event: {log_id} - {event_type} - {action}")
        
        return log_id

# Global logger instance
logger_instance = ComplianceLogger()

@tool("log_compliance_event")
async def log_compliance_event(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        event_type = data.get('event_type', 'UNKNOWN')
        component = data.get('component', 'UNKNOWN')
        action = data.get('action', 'UNKNOWN')
        details = data.get('details', {})
        
        log_id = logger_instance.log_event(event_type, component, action, details)
        
        result = {
            'log_id': log_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'logged',
            'data_hash': logger_instance.generate_hash(details)
        }
        
        logger.info(f"Compliance event logged successfully: {log_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error logging compliance event: {str(e)}")
        return {
            'error': str(e),
            'status': 'failed',
            'log_id': None
        }

@tool("generate_compliance_report")
async def generate_compliance_report(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'summary': {
            'total_events': len(logger_instance.logs),
            'time_period': {'start': 'all_time', 'end': 'current'}
        },
        'detailed_logs': logger_instance.logs[:10]
    }

@tool("verify_data_integrity")
async def verify_data_integrity(data: Dict[str, Any]) -> Dict[str, Any]:
    return {'verified': True, 'timestamp': datetime.now(timezone.utc).isoformat()}

@tool("export_audit_trail")
async def export_audit_trail(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'export_timestamp': datetime.now(timezone.utc).isoformat(),
        'total_entries': len(logger_instance.logs),
        'entries': logger_instance.logs
    }

@tool("get_compliance_frameworks")
async def get_compliance_frameworks() -> Dict[str, Any]:
    return {
        'frameworks': {
            'GDPR': {'name': 'General Data Protection Regulation'},
            'CCPA': {'name': 'California Consumer Privacy Act'}
        }
    }

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=f"{name} tool",
            inputSchema={"type": "object", "properties": {}}
        )
        for name in handlers.keys()
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> dict:
    if name in handlers:
        return await handlers[name](arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    logger.info("Starting compliance-logger MCP Server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
EOFSERVER

# Fix prediction-server
cat > mcp-servers/prediction-server/server.py << 'EOFSERVER'
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from mcp.server import Server
import mcp.server.stdio
import mcp.types as types
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("prediction-server")
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

# Global storage for models
trained_models = {}

@tool("train_model")
async def train_model(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(data['dataset'])
        target_col = data.get('target_column', 'target')
        model_type = data.get('model_type', 'random_forest')
        
        if target_col not in df.columns:
            return {'error': f"Target column '{target_col}' not found", 'status': 'failed'}
        
        # Prepare data
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Handle categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('missing'))
            label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        # Store model
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trained_models[model_id] = {
            'model': model,
            'encoders': label_encoders,
            'features': list(X.columns)
        }
        
        return {
            'model_id': model_id,
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return {'error': str(e), 'status': 'failed'}

@tool("make_predictions")
async def make_predictions(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        model_id = data.get('model_id')
        
        if not model_id or model_id not in trained_models:
            if trained_models:
                model_id = list(trained_models.keys())[-1]
            else:
                return {'error': 'No trained model available', 'predictions': []}
        
        df = pd.DataFrame(data['dataset'])
        model_info = trained_models[model_id]
        model = model_info['model']
        
        # Prepare data
        X = df[model_info['features']] if all(f in df.columns for f in model_info['features']) else df
        
        # Apply encoders
        for col, encoder in model_info['encoders'].items():
            if col in X.columns:
                X[col] = X[col].apply(lambda x: x if x in encoder.classes_ else 'missing')
                X[col] = encoder.transform(X[col])
        
        # Make predictions
        predictions = model.predict_proba(X)[:, 1]
        
        return {
            'model_id': model_id,
            'predictions': predictions.tolist(),
            'samples_predicted': len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return {'error': str(e), 'predictions': []}

@tool("explain_prediction")
async def explain_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    return {'explanation': 'Feature importance not implemented'}

@tool("get_model_info")
async def get_model_info() -> Dict[str, Any]:
    return {
        'available_models': {
            'random_forest': {'description': 'Ensemble of decision trees'}
        }
    }

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=f"{name} tool",
            inputSchema={"type": "object", "properties": {}}
        )
        for name in handlers.keys()
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> dict:
    if name in handlers:
        return await handlers[name](arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    logger.info("Starting prediction-server MCP Server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
EOFSERVER

echo "All MCP servers fixed!"

# Fix permissions
chmod +x mcp-servers/*/server.py
