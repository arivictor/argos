# Examples

This guide provides comprehensive, real-world examples of using AroFlow for various scenarios. Each example includes complete plugin implementations and workflow definitions you can adapt for your own use cases.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Data Processing](#data-processing)
- [Web Scraping Pipeline](#web-scraping-pipeline)
- [Machine Learning Workflow](#machine-learning-workflow)
- [API Integration](#api-integration)
- [File Processing](#file-processing)
- [Notification Systems](#notification-systems)
- [Database Operations](#database-operations)

## Basic Examples

### Hello World

The simplest possible AroFlow workflow:

```python
import aroflow

class GreetingPlugin(aroflow.PluginMixin):
    plugin_name = "greet"
    
    def execute(self, name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

# Setup
client = aroflow.create(aroflow.BackendType.IN_MEMORY)
client.plugin(GreetingPlugin)

# Workflow
workflow = {
    "steps": [
        {
            "id": "say_hello",
            "kind": "operation",
            "operation": "greet",
            "parameters": {
                "name": "AroFlow",
                "greeting": "Welcome"
            }
        }
    ]
}

# Execute
result = client.run(workflow)
print(result.to_yaml())
```

### Sequential Steps

Chain operations together:

```python
import aroflow

class MathPlugin(aroflow.PluginMixin):
    plugin_name = "math"
    
    def execute(self, operation: str, a: float, b: float) -> float:
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            return a / b if b != 0 else 0
        else:
            raise ValueError(f"Unknown operation: {operation}")

class FormatterPlugin(aroflow.PluginMixin):
    plugin_name = "format"
    
    def execute(self, value: float, format_type: str = "decimal") -> str:
        if format_type == "decimal":
            return f"{value:.2f}"
        elif format_type == "currency":
            return f"${value:.2f}"
        elif format_type == "percentage":
            return f"{value:.1%}"
        else:
            return str(value)

# Setup
client = aroflow.create(aroflow.BackendType.IN_MEMORY)
client.plugin(MathPlugin)
client.plugin(FormatterPlugin)

# Workflow: Calculate compound interest
workflow = {
    "steps": [
        {
            "id": "calculate_interest",
            "kind": "operation",
            "operation": "math",
            "parameters": {
                "operation": "multiply",
                "a": 1000.0,  # Principal
                "b": 1.05     # Interest rate
            }
        },
        {
            "id": "format_result",
            "kind": "operation",
            "operation": "format",
            "parameters": {
                "value": "${calculate_interest.result}",
                "format_type": "currency"
            }
        }
    ]
}

result = client.run(workflow)
print(f"Investment result: {result.results[1].result}")
```

## Data Processing

### CSV Data Analysis

Complete pipeline for analyzing CSV data:

```python
import aroflow
import pandas as pd
import json
from io import StringIO

class CSVReaderPlugin(aroflow.PluginMixin):
    plugin_name = "read_csv"
    
    def execute(self, file_path: str, delimiter: str = ",") -> dict:
        df = pd.read_csv(file_path, delimiter=delimiter)
        return {
            "data": df.to_dict('records'),
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict()
        }

class DataFilterPlugin(aroflow.PluginMixin):
    plugin_name = "filter_data"
    
    def execute(self, data: list, filter_column: str, filter_value: str) -> dict:
        filtered_data = [
            row for row in data 
            if str(row.get(filter_column, "")).lower() == filter_value.lower()
        ]
        return {
            "data": filtered_data,
            "original_count": len(data),
            "filtered_count": len(filtered_data)
        }

class StatisticsPlugin(aroflow.PluginMixin):
    plugin_name = "calculate_stats"
    
    def execute(self, data: list, numeric_columns: list) -> dict:
        if not data:
            return {"error": "No data to analyze"}
            
        stats = {}
        for column in numeric_columns:
            values = [
                float(row[column]) for row in data 
                if column in row and row[column] is not None
            ]
            
            if values:
                stats[column] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "median": sorted(values)[len(values) // 2]
                }
        
        return stats

class ReportGeneratorPlugin(aroflow.PluginMixin):
    plugin_name = "generate_report"
    
    def execute(self, statistics: dict, title: str = "Data Analysis Report") -> str:
        report = f"# {title}\n\n"
        
        for column, stats in statistics.items():
            report += f"## {column.title()}\n"
            report += f"- Count: {stats['count']}\n"
            report += f"- Mean: {stats['mean']:.2f}\n"
            report += f"- Min: {stats['min']:.2f}\n"
            report += f"- Max: {stats['max']:.2f}\n"
            report += f"- Median: {stats['median']:.2f}\n\n"
        
        return report

# Setup
client = aroflow.create(aroflow.BackendType.IN_MEMORY)
client.plugin(CSVReaderPlugin)
client.plugin(DataFilterPlugin)
client.plugin(StatisticsPlugin)
client.plugin(ReportGeneratorPlugin)

# Workflow: Analyze sales data
data_analysis_workflow = {
    "steps": [
        {
            "id": "load_sales_data",
            "kind": "operation",
            "operation": "read_csv",
            "parameters": {
                "file_path": "/data/sales.csv",
                "delimiter": ","
            }
        },
        {
            "id": "filter_current_year",
            "kind": "operation",
            "operation": "filter_data",
            "parameters": {
                "data": "${load_sales_data.result.data}",
                "filter_column": "year",
                "filter_value": "2024"
            }
        },
        {
            "id": "calculate_statistics",
            "kind": "operation",
            "operation": "calculate_stats",
            "parameters": {
                "data": "${filter_current_year.result.data}",
                "numeric_columns": ["revenue", "quantity", "profit_margin"]
            }
        },
        {
            "id": "generate_report",
            "kind": "operation", 
            "operation": "generate_report",
            "parameters": {
                "statistics": "${calculate_statistics.result}",
                "title": "2024 Sales Analysis Report"
            }
        }
    ]
}
```

### Batch Data Processing

Process large datasets in batches:

```python
import aroflow

class BatchProcessorPlugin(aroflow.PluginMixin):
    plugin_name = "batch_processor"
    
    def execute(self, data: list, batch_size: int = 100) -> dict:
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append({
                "batch_id": i // batch_size,
                "data": batch,
                "size": len(batch)
            })
        
        return {
            "batches": batches,
            "total_batches": len(batches),
            "total_records": len(data)
        }

class RecordEnricherPlugin(aroflow.PluginMixin):
    plugin_name = "enrich_record"
    
    def execute(self, record: dict, enrichment_source: str = "api") -> dict:
        # Simulate enrichment (in real implementation, call external API)
        enriched_record = record.copy()
        enriched_record.update({
            "enriched_at": "2024-01-15T10:00:00Z",
            "enrichment_source": enrichment_source,
            "confidence_score": 0.95
        })
        return enriched_record

class BatchResultsAggregatorPlugin(aroflow.PluginMixin):
    plugin_name = "aggregate_results"
    
    def execute(self, batch_results: list) -> dict:
        all_records = []
        total_processed = 0
        
        for batch_result in batch_results:
            if "results" in batch_result:
                all_records.extend(batch_result["results"])
                total_processed += len(batch_result["results"])
        
        return {
            "all_records": all_records,
            "total_processed": total_processed,
            "batches_processed": len(batch_results)
        }

# Setup
client = aroflow.create(aroflow.BackendType.IN_MEMORY)
client.plugin(BatchProcessorPlugin)
client.plugin(RecordEnricherPlugin)
client.plugin(BatchResultsAggregatorPlugin)

# Workflow: Process large dataset in batches
batch_processing_workflow = {
    "steps": [
        {
            "id": "create_batches",
            "kind": "operation",
            "operation": "batch_processor",
            "parameters": {
                "data": "${input_data}",  # Assume this comes from previous step
                "batch_size": 50
            }
        },
        {
            "id": "process_batches",
            "kind": "map",
            "mode": "parallel",
            "iterator": "batch",
            "inputs": "${create_batches.result.batches}",
            "operation": {
                "id": "process_single_batch",
                "kind": "map",
                "mode": "sequential",  # Process records in batch sequentially
                "iterator": "record",
                "inputs": "${batch.data}",
                "operation": {
                    "id": "enrich_single_record",
                    "kind": "operation",
                    "operation": "enrich_record",
                    "parameters": {
                        "record": "${record}",
                        "enrichment_source": "external_api"
                    }
                }
            }
        },
        {
            "id": "aggregate_results",
            "kind": "operation",
            "operation": "aggregate_results",
            "parameters": {
                "batch_results": "${process_batches.results}"
            }
        }
    ]
}
```

## Web Scraping Pipeline

Complete web scraping and data extraction pipeline:

```python
import aroflow
import requests
from bs4 import BeautifulSoup
import time
import random

class WebScraperPlugin(aroflow.PluginMixin):
    plugin_name = "scrape_web"
    
    def execute(
        self, 
        url: str, 
        selector: str, 
        delay: float = 1.0,
        user_agent: str = "AroFlow WebScraper 1.0"
    ) -> dict:
        headers = {"User-Agent": user_agent}
        
        # Add random delay to be respectful
        time.sleep(random.uniform(delay, delay * 2))
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            elements = soup.select(selector)
            
            extracted_data = []
            for element in elements:
                extracted_data.append({
                    "text": element.get_text(strip=True),
                    "html": str(element),
                    "attributes": dict(element.attrs)
                })
            
            return {
                "url": url,
                "status_code": response.status_code,
                "data": extracted_data,
                "count": len(extracted_data),
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

class DataCleanerPlugin(aroflow.PluginMixin):
    plugin_name = "clean_scraped_data"
    
    def execute(self, scraped_results: list, remove_duplicates: bool = True) -> dict:
        all_data = []
        
        for result in scraped_results:
            if "data" in result:
                for item in result["data"]:
                    cleaned_item = {
                        "source_url": result["url"],
                        "text": item["text"].strip(),
                        "scraped_at": result["scraped_at"]
                    }
                    all_data.append(cleaned_item)
        
        if remove_duplicates:
            seen = set()
            unique_data = []
            for item in all_data:
                key = (item["text"], item["source_url"])
                if key not in seen:
                    seen.add(key)
                    unique_data.append(item)
            all_data = unique_data
        
        return {
            "cleaned_data": all_data,
            "total_items": len(all_data),
            "sources_scraped": len(set(item["source_url"] for item in all_data))
        }

class DataExporterPlugin(aroflow.PluginMixin):
    plugin_name = "export_data"
    
    def execute(self, data: list, format: str = "json", output_path: str = None) -> dict:
        if format == "json":
            import json
            output = json.dumps(data, indent=2)
        elif format == "csv":
            import csv
            from io import StringIO
            output_buffer = StringIO()
            if data:
                writer = csv.DictWriter(output_buffer, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            output = output_buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(output)
            return {"message": f"Data exported to {output_path}", "format": format}
        else:
            return {"data": output, "format": format}

# Setup
client = aroflow.create(aroflow.BackendType.IN_MEMORY)
client.plugin(WebScraperPlugin)
client.plugin(DataCleanerPlugin)
client.plugin(DataExporterPlugin)

# Workflow: Scrape news headlines
news_scraping_workflow = {
    "steps": [
        {
            "id": "scrape_news_sites",
            "kind": "map",
            "mode": "sequential",  # Be respectful to websites
            "iterator": "site_config",
            "inputs": [
                {
                    "url": "https://news.ycombinator.com",
                    "selector": ".storylink",
                    "name": "Hacker News"
                },
                {
                    "url": "https://www.reddit.com/r/technology",
                    "selector": ".entry-title a", 
                    "name": "Reddit Technology"
                }
            ],
            "operation": {
                "id": "scrape_single_site",
                "kind": "operation",
                "operation": "scrape_web",
                "parameters": {
                    "url": "${site_config.url}",
                    "selector": "${site_config.selector}",
                    "delay": 2.0
                }
            }
        },
        {
            "id": "clean_and_deduplicate",
            "kind": "operation",
            "operation": "clean_scraped_data",
            "parameters": {
                "scraped_results": "${scrape_news_sites.results}",
                "remove_duplicates": True
            }
        },
        {
            "id": "export_results",
            "kind": "parallel",
            "operations": [
                {
                    "id": "export_json",
                    "kind": "operation",
                    "operation": "export_data",
                    "parameters": {
                        "data": "${clean_and_deduplicate.result.cleaned_data}",
                        "format": "json",
                        "output_path": "/output/news_headlines.json"
                    }
                },
                {
                    "id": "export_csv",
                    "kind": "operation",
                    "operation": "export_data",
                    "parameters": {
                        "data": "${clean_and_deduplicate.result.cleaned_data}",
                        "format": "csv",
                        "output_path": "/output/news_headlines.csv"
                    }
                }
            ]
        }
    ]
}
```

## Machine Learning Workflow

End-to-end ML pipeline with data preprocessing, model training, and evaluation:

```python
import aroflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

class DataLoaderPlugin(aroflow.PluginMixin):
    plugin_name = "load_ml_data"
    
    def execute(self, file_path: str, target_column: str) -> dict:
        df = pd.read_csv(file_path)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        return {
            "features": X.to_dict('records'),
            "target": y.tolist(),
            "feature_names": X.columns.tolist(),
            "target_name": target_column,
            "shape": df.shape
        }

class DataPreprocessorPlugin(aroflow.PluginMixin):
    plugin_name = "preprocess_data"
    
    def execute(self, features: list, target: list, feature_names: list) -> dict:
        df = pd.DataFrame(features)
        
        # Handle missing values
        df = df.fillna(df.mean() if df.select_dtypes(include=[np.number]).shape[1] > 0 else df.mode().iloc[0])
        
        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        
        return {
            "features": X_scaled.tolist(),
            "target": target,
            "feature_names": feature_names,
            "scaler": scaler,
            "label_encoders": label_encoders,
            "preprocessed_shape": X_scaled.shape
        }

class TrainTestSplitPlugin(aroflow.PluginMixin):
    plugin_name = "train_test_split"
    
    def execute(
        self, 
        features: list, 
        target: list, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> dict:
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

class ModelTrainerPlugin(aroflow.PluginMixin):
    plugin_name = "train_model"
    
    def execute(
        self, 
        X_train: list, 
        y_train: list, 
        model_type: str = "random_forest",
        **model_params
    ) -> dict:
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        return {
            "model": model,
            "model_type": model_type,
            "model_params": model_params,
            "feature_importances": (
                model.feature_importances_.tolist() 
                if hasattr(model, 'feature_importances_') else None
            )
        }

class ModelEvaluatorPlugin(aroflow.PluginMixin):
    plugin_name = "evaluate_model"
    
    def execute(self, model, X_test: list, y_test: list, model_type: str) -> dict:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "predictions": y_pred.tolist(),
            "model_type": model_type,
            "test_samples": len(y_test)
        }

class ModelSaverPlugin(aroflow.PluginMixin):
    plugin_name = "save_model"
    
    def execute(
        self, 
        model, 
        scaler, 
        label_encoders: dict, 
        model_path: str,
        metadata: dict = None
    ) -> dict:
        # Save model artifacts
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "label_encoders": label_encoders,
            "metadata": metadata or {}
        }, model_path)
        
        return {
            "message": f"Model saved to {model_path}",
            "model_path": model_path,
            "artifacts": ["model", "scaler", "label_encoders", "metadata"]
        }

# Setup
client = aroflow.create(aroflow.BackendType.IN_MEMORY)
client.plugin(DataLoaderPlugin)
client.plugin(DataPreprocessorPlugin)
client.plugin(TrainTestSplitPlugin)
client.plugin(ModelTrainerPlugin)
client.plugin(ModelEvaluatorPlugin)
client.plugin(ModelSaverPlugin)

# ML Workflow
ml_pipeline = {
    "steps": [
        {
            "id": "load_data",
            "kind": "operation",
            "operation": "load_ml_data",
            "parameters": {
                "file_path": "/data/iris.csv",
                "target_column": "species"
            }
        },
        {
            "id": "preprocess_data",
            "kind": "operation",
            "operation": "preprocess_data",
            "parameters": {
                "features": "${load_data.result.features}",
                "target": "${load_data.result.target}",
                "feature_names": "${load_data.result.feature_names}"
            }
        },
        {
            "id": "split_data",
            "kind": "operation",
            "operation": "train_test_split",
            "parameters": {
                "features": "${preprocess_data.result.features}",
                "target": "${preprocess_data.result.target}",
                "test_size": 0.2,
                "random_state": 42
            }
        },
        {
            "id": "train_models",
            "kind": "map",
            "mode": "parallel",
            "iterator": "model_config",
            "inputs": [
                {
                    "model_type": "random_forest",
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                },
                {
                    "model_type": "logistic_regression",
                    "max_iter": 1000,
                    "random_state": 42
                }
            ],
            "operation": {
                "id": "train_single_model",
                "kind": "operation",
                "operation": "train_model",
                "parameters": {
                    "X_train": "${split_data.result.X_train}",
                    "y_train": "${split_data.result.y_train}",
                    "model_type": "${model_config.model_type}",
                    "n_estimators": "${model_config.n_estimators}",
                    "max_depth": "${model_config.max_depth}",
                    "max_iter": "${model_config.max_iter}",
                    "random_state": "${model_config.random_state}"
                }
            }
        },
        {
            "id": "evaluate_models",
            "kind": "map",
            "mode": "parallel",
            "iterator": "trained_model",
            "inputs": "${train_models.results}",
            "operation": {
                "id": "evaluate_single_model",
                "kind": "operation",
                "operation": "evaluate_model",
                "parameters": {
                    "model": "${trained_model.result.model}",
                    "X_test": "${split_data.result.X_test}",
                    "y_test": "${split_data.result.y_test}",
                    "model_type": "${trained_model.result.model_type}"
                }
            }
        },
        {
            "id": "save_best_model",
            "kind": "operation",
            "operation": "save_model",
            "parameters": {
                "model": "${train_models.results[0].result.model}",  # Assuming first model is best
                "scaler": "${preprocess_data.result.scaler}",
                "label_encoders": "${preprocess_data.result.label_encoders}",
                "model_path": "/models/best_model.pkl",
                "metadata": {
                    "accuracy": "${evaluate_models.results[0].result.accuracy}",
                    "model_type": "${evaluate_models.results[0].result.model_type}",
                    "training_date": "2024-01-15"
                }
            }
        }
    ]
}
```

## API Integration

Integrate with external APIs and handle responses:

```python
import aroflow
import requests
import time

class APIClientPlugin(aroflow.PluginMixin):
    plugin_name = "api_client"
    
    def execute(
        self,
        url: str,
        method: str = "GET",
        headers: dict = None,
        data: dict = None,
        params: dict = None,
        timeout: int = 30,
        retries: int = 3
    ) -> dict:
        headers = headers or {}
        
        for attempt in range(retries + 1):
            try:
                response = requests.request(
                    method=method.upper(),
                    url=url,
                    headers=headers,
                    json=data if method.upper() in ["POST", "PUT", "PATCH"] else None,
                    params=params,
                    timeout=timeout
                )
                
                result = {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "url": response.url,
                    "elapsed_seconds": response.elapsed.total_seconds()
                }
                
                # Try to parse JSON response
                try:
                    result["data"] = response.json()
                except:
                    result["data"] = response.text
                
                # Check if request was successful
                if response.status_code < 400:
                    result["success"] = True
                    return result
                else:
                    result["success"] = False
                    result["error"] = f"HTTP {response.status_code}: {response.reason}"
                    
                    if attempt == retries:
                        return result
                        
            except requests.exceptions.RequestException as e:
                if attempt == retries:
                    return {
                        "success": False,
                        "error": str(e),
                        "url": url,
                        "attempt": attempt + 1
                    }
                time.sleep(2 ** attempt)  # Exponential backoff

class DataTransformerPlugin(aroflow.PluginMixin):
    plugin_name = "transform_api_data"
    
    def execute(self, api_responses: list, transform_rules: dict) -> dict:
        transformed_data = []
        
        for response in api_responses:
            if response.get("success") and "data" in response:
                data = response["data"]
                
                # Apply transformation rules
                if isinstance(data, list):
                    for item in data:
                        transformed_item = self._transform_item(item, transform_rules)
                        transformed_data.append(transformed_item)
                elif isinstance(data, dict):
                    transformed_item = self._transform_item(data, transform_rules)
                    transformed_data.append(transformed_item)
        
        return {
            "transformed_data": transformed_data,
            "total_items": len(transformed_data),
            "transformation_rules": transform_rules
        }
    
    def _transform_item(self, item: dict, rules: dict) -> dict:
        transformed = {}
        
        for new_key, rule in rules.items():
            if isinstance(rule, str):
                # Simple field mapping
                transformed[new_key] = item.get(rule)
            elif isinstance(rule, dict):
                # Complex transformation
                if "source_field" in rule:
                    value = item.get(rule["source_field"])
                    
                    # Apply transformation
                    if "transform" in rule:
                        if rule["transform"] == "upper":
                            value = str(value).upper() if value else value
                        elif rule["transform"] == "lower":
                            value = str(value).lower() if value else value
                        elif rule["transform"] == "int":
                            value = int(value) if value else 0
                    
                    transformed[new_key] = value
        
        return transformed

class DataValidatorPlugin(aroflow.PluginMixin):
    plugin_name = "validate_data"
    
    def execute(self, data: list, validation_rules: dict) -> dict:
        valid_records = []
        invalid_records = []
        
        for record in data:
            validation_errors = []
            
            # Check required fields
            for field in validation_rules.get("required_fields", []):
                if field not in record or record[field] is None:
                    validation_errors.append(f"Missing required field: {field}")
            
            # Check field types
            for field, expected_type in validation_rules.get("field_types", {}).items():
                if field in record and record[field] is not None:
                    if expected_type == "string" and not isinstance(record[field], str):
                        validation_errors.append(f"Field {field} must be string")
                    elif expected_type == "int" and not isinstance(record[field], int):
                        validation_errors.append(f"Field {field} must be integer")
            
            # Add validation results
            record_with_validation = record.copy()
            record_with_validation["_validation_errors"] = validation_errors
            
            if validation_errors:
                invalid_records.append(record_with_validation)
            else:
                valid_records.append(record)
        
        return {
            "valid_records": valid_records,
            "invalid_records": invalid_records,
            "total_records": len(data),
            "valid_count": len(valid_records),
            "invalid_count": len(invalid_records),
            "validation_rate": len(valid_records) / len(data) if data else 0
        }

# Setup
client = aroflow.create(aroflow.BackendType.IN_MEMORY)
client.plugin(APIClientPlugin)
client.plugin(DataTransformerPlugin)
client.plugin(DataValidatorPlugin)

# API Integration Workflow
api_workflow = {
    "steps": [
        {
            "id": "fetch_user_data",
            "kind": "map",
            "mode": "parallel",
            "iterator": "user_id",
            "inputs": ["1", "2", "3", "4", "5"],
            "operation": {
                "id": "get_user",
                "kind": "operation",
                "operation": "api_client",
                "parameters": {
                    "url": "https://jsonplaceholder.typicode.com/users/${user_id}",
                    "method": "GET",
                    "timeout": 10,
                    "retries": 2
                }
            }
        },
        {
            "id": "fetch_user_posts",
            "kind": "map",
            "mode": "parallel",
            "iterator": "user_id",
            "inputs": ["1", "2", "3", "4", "5"],
            "operation": {
                "id": "get_posts",
                "kind": "operation",
                "operation": "api_client",
                "parameters": {
                    "url": "https://jsonplaceholder.typicode.com/posts",
                    "method": "GET",
                    "params": {"userId": "${user_id}"},
                    "timeout": 10
                }
            }
        },
        {
            "id": "transform_user_data",
            "kind": "operation",
            "operation": "transform_api_data",
            "parameters": {
                "api_responses": "${fetch_user_data.results}",
                "transform_rules": {
                    "user_id": "id",
                    "username": "username",
                    "email": "email",
                    "full_name": "name",
                    "city": {
                        "source_field": "address.city",
                        "transform": "title"
                    }
                }
            }
        },
        {
            "id": "validate_transformed_data",
            "kind": "operation",
            "operation": "validate_data",
            "parameters": {
                "data": "${transform_user_data.result.transformed_data}",
                "validation_rules": {
                    "required_fields": ["user_id", "username", "email"],
                    "field_types": {
                        "user_id": "int",
                        "username": "string",
                        "email": "string"
                    }
                }
            }
        }
    ]
}
```

This comprehensive examples guide demonstrates the power and flexibility of AroFlow across different domains. Each example is complete and runnable, showing best practices for plugin development and workflow composition.

## File Processing

Process files and directories with AroFlow:

```python
import aroflow
import os
import shutil
from pathlib import Path

class FileProcessorPlugin(aroflow.PluginMixin):
    plugin_name = "process_file"
    
    def execute(self, file_path: str, operation: str, **kwargs) -> dict:
        path = Path(file_path)
        
        if operation == "copy":
            dest = kwargs.get("destination")
            shutil.copy2(file_path, dest)
            return {"message": f"Copied {file_path} to {dest}"}
            
        elif operation == "move":
            dest = kwargs.get("destination")
            shutil.move(file_path, dest)
            return {"message": f"Moved {file_path} to {dest}"}
            
        elif operation == "delete":
            os.remove(file_path)
            return {"message": f"Deleted {file_path}"}
            
        elif operation == "info":
            stat = path.stat()
            return {
                "name": path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "exists": path.exists()
            }
```

## Notification Systems

Send notifications through various channels:

```python
import aroflow
import smtplib
from email.mime.text import MIMEText

class EmailNotificationPlugin(aroflow.PluginMixin):
    plugin_name = "send_email"
    
    def execute(self, to: str, subject: str, body: str, smtp_server: str = "localhost") -> dict:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['To'] = to
        msg['From'] = "noreply@aroflow.dev"
        
        try:
            with smtplib.SMTP(smtp_server) as server:
                server.send_message(msg)
            return {"status": "sent", "to": to, "subject": subject}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

class SlackNotificationPlugin(aroflow.PluginMixin):
    plugin_name = "send_slack"
    
    def execute(self, channel: str, message: str, webhook_url: str) -> dict:
        import requests
        
        payload = {
            "channel": channel,
            "text": message,
            "username": "AroFlow"
        }
        
        response = requests.post(webhook_url, json=payload)
        return {
            "status": "sent" if response.ok else "failed",
            "channel": channel,
            "status_code": response.status_code
        }
```

## Database Operations

Interact with databases using AroFlow:

```python
import aroflow
import sqlite3

class DatabasePlugin(aroflow.PluginMixin):
    plugin_name = "database"
    
    def execute(self, operation: str, query: str, db_path: str = "data.db", params: list = None) -> dict:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            if operation == "select":
                cursor.execute(query, params or [])
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return {
                    "results": [dict(zip(columns, row)) for row in results],
                    "count": len(results)
                }
            elif operation in ["insert", "update", "delete"]:
                cursor.execute(query, params or [])
                conn.commit()
                return {
                    "rows_affected": cursor.rowcount,
                    "operation": operation
                }
        finally:
            conn.close()
```

## Running the Examples

To run any of these examples:

1. **Install AroFlow**: `pip install aroflow`
2. **Copy the plugin code** into your Python file
3. **Set up the client** and register plugins
4. **Define your workflow** 
5. **Execute** with `client.run(workflow)`

## Customization Tips

- **Modify plugin parameters** to fit your specific use cases
- **Combine workflows** by using outputs from one as inputs to another
- **Add error handling** with retries and timeouts
- **Scale with parallel processing** using map and parallel steps
- **Extend plugins** with additional functionality as needed

---

Ready to build your own workflows? Start with the [plugins guide](plugins.md) or [workflow documentation](workflows.md)!