# Creating Workflows

Workflows are the orchestration layer of AroFlow. They define the sequence and structure of operations to achieve your goals. This guide covers everything from simple linear workflows to complex parallel and data processing pipelines.

## Workflow Basics

### What is a Workflow?

A workflow is a JSON/dictionary structure that defines a series of steps to execute. Each step specifies what operation to perform and how it connects to other steps.

```python
workflow = {
    "steps": [
        {
            "id": "step1",           # Unique identifier
            "kind": "operation",     # Step type
            "operation": "my_plugin", # Plugin to execute
            "parameters": {          # Input parameters
                "param1": "value1"
            }
        }
    ]
}
```

### Step Types

AroFlow supports three types of steps:

1. **Operation Steps** - Execute a single plugin
2. **Map Steps** - Execute a plugin over a list of inputs
3. **Parallel Steps** - Execute multiple operations concurrently

## Operation Steps

Operation steps execute a single plugin with specified parameters.

### Basic Operation

```python
{
    "id": "send_notification",
    "kind": "operation",
    "operation": "send_email",
    "parameters": {
        "to": "user@example.com",
        "subject": "Workflow Complete",
        "body": "Your data processing is complete!"
    }
}
```

### Error Handling Options

```python
{
    "id": "risky_operation",
    "kind": "operation",
    "operation": "external_api_call",
    "parameters": {
        "url": "https://api.example.com/data"
    },
    "retries": 3,              # Retry failed operations
    "timeout": 30.0,           # Timeout in seconds
    "fail_workflow": False     # Continue on failure
}
```

## Map Steps

Map steps execute a plugin over a list of inputs, perfect for data processing pipelines.

### Sequential Map

Process items one after another:

```python
{
    "id": "process_files",
    "kind": "map",
    "mode": "sequential",      # Process one at a time
    "iterator": "file_path",   # Variable name for current item
    "inputs": [                # List of items to process
        "/data/file1.txt",
        "/data/file2.txt", 
        "/data/file3.txt"
    ],
    "operation": {
        "id": "file_processor",
        "kind": "operation",
        "operation": "process_file",
        "parameters": {
            "file_path": "${file_path}",  # Reference current item
            "output_dir": "/processed/"
        }
    }
}
```

### Parallel Map

Process items concurrently for better performance:

```python
{
    "id": "resize_images",
    "kind": "map",
    "mode": "parallel",        # Process concurrently
    "iterator": "image",
    "inputs": [
        {"path": "img1.jpg", "size": "800x600"},
        {"path": "img2.jpg", "size": "800x600"},
        {"path": "img3.jpg", "size": "800x600"}
    ],
    "operation": {
        "id": "image_resizer",
        "kind": "operation",
        "operation": "resize_image",
        "parameters": {
            "input_path": "${image.path}",
            "output_size": "${image.size}",
            "quality": 85
        }
    }
}
```

### Map with Dynamic Inputs

Use results from previous steps as map inputs:

```python
{
    "steps": [
        {
            "id": "get_file_list",
            "kind": "operation",
            "operation": "list_files",
            "parameters": {"directory": "/uploads/"}
        },
        {
            "id": "process_all_files", 
            "kind": "map",
            "mode": "parallel",
            "iterator": "file",
            "inputs": "${get_file_list.result}",  # Use previous result
            "operation": {
                "id": "file_processor",
                "kind": "operation",
                "operation": "process_file",
                "parameters": {
                    "file_path": "${file}"
                }
            }
        }
    ]
}
```

## Parallel Steps

Parallel steps execute multiple operations concurrently.

### Basic Parallel Execution

```python
{
    "id": "parallel_notifications",
    "kind": "parallel",
    "operations": [
        {
            "id": "email_notification",
            "kind": "operation",
            "operation": "send_email",
            "parameters": {
                "to": "admin@company.com",
                "subject": "Process Complete"
            }
        },
        {
            "id": "slack_notification", 
            "kind": "operation",
            "operation": "send_slack",
            "parameters": {
                "channel": "#alerts",
                "message": "Data processing finished"
            }
        },
        {
            "id": "update_dashboard",
            "kind": "operation", 
            "operation": "update_status",
            "parameters": {
                "status": "complete",
                "timestamp": "${workflow.start_time}"
            }
        }
    ]
}
```

### Parallel Data Processing

```python
{
    "id": "parallel_analysis",
    "kind": "parallel",
    "operations": [
        {
            "id": "calculate_stats",
            "kind": "operation",
            "operation": "statistical_analysis",
            "parameters": {
                "data": "${previous_step.result}",
                "analysis_type": "descriptive"
            }
        },
        {
            "id": "generate_charts",
            "kind": "operation",
            "operation": "create_visualization",
            "parameters": {
                "data": "${previous_step.result}",
                "chart_type": "histogram"
            }
        },
        {
            "id": "export_data",
            "kind": "operation",
            "operation": "export_csv",
            "parameters": {
                "data": "${previous_step.result}",
                "filename": "analysis_results.csv"
            }
        }
    ]
}
```

## Variable Substitution

AroFlow supports powerful variable substitution to pass data between steps.

### Basic Variable References

```python
{
    "steps": [
        {
            "id": "get_user_data",
            "kind": "operation",
            "operation": "fetch_user",
            "parameters": {"user_id": "12345"}
        },
        {
            "id": "send_welcome",
            "kind": "operation", 
            "operation": "send_email",
            "parameters": {
                "to": "${get_user_data.result.email}",      # Access nested fields
                "subject": "Welcome ${get_user_data.result.name}!",
                "body": "Hello ${get_user_data.result.name}, welcome aboard!"
            }
        }
    ]
}
```

### Array Access

```python
{
    "steps": [
        {
            "id": "get_file_list",
            "kind": "operation",
            "operation": "list_files",
            "parameters": {"directory": "/data/"}
        },
        {
            "id": "process_first_file",
            "kind": "operation",
            "operation": "process_file", 
            "parameters": {
                "file_path": "${get_file_list.result[0]}",  # First file
                "backup_path": "${get_file_list.result[1]}" # Second file
            }
        }
    ]
}
```

### Complex Variable Expressions

```python
{
    "id": "conditional_processing",
    "kind": "operation",
    "operation": "process_data",
    "parameters": {
        "input_data": "${data_validation.result.cleaned_data}",
        "config": {
            "batch_size": "${data_validation.result.record_count}",
            "parallel": "${system_info.result.cpu_count}",
            "memory_limit": "${system_info.result.available_memory}"
        }
    }
}
```

## Complete Workflow Examples

### Data Processing Pipeline

```python
data_pipeline = {
    "steps": [
        # Step 1: Validate input data
        {
            "id": "validate_input",
            "kind": "operation",
            "operation": "data_validator",
            "parameters": {
                "file_path": "/input/raw_data.csv",
                "schema": {
                    "required_columns": ["id", "name", "email", "age"],
                    "data_types": {"id": "int", "age": "int"}
                }
            }
        },
        
        # Step 2: Clean data in parallel
        {
            "id": "parallel_cleaning",
            "kind": "parallel", 
            "operations": [
                {
                    "id": "remove_duplicates",
                    "kind": "operation",
                    "operation": "deduplicator",
                    "parameters": {
                        "data": "${validate_input.result.data}",
                        "key_columns": ["email"]
                    }
                },
                {
                    "id": "normalize_text",
                    "kind": "operation",
                    "operation": "text_normalizer",
                    "parameters": {
                        "data": "${validate_input.result.data}",
                        "columns": ["name"],
                        "operations": ["trim", "title_case"]
                    }
                }
            ]
        },
        
        # Step 3: Process each record
        {
            "id": "process_records",
            "kind": "map",
            "mode": "parallel",
            "iterator": "record",
            "inputs": "${remove_duplicates.result}",
            "operation": {
                "id": "record_processor",
                "kind": "operation",
                "operation": "enrich_record",
                "parameters": {
                    "record": "${record}",
                    "enrichment_api": "https://api.enrichment.com/v1/person"
                }
            }
        },
        
        # Step 4: Generate outputs
        {
            "id": "generate_outputs",
            "kind": "parallel",
            "operations": [
                {
                    "id": "save_to_database",
                    "kind": "operation",
                    "operation": "database_writer",
                    "parameters": {
                        "data": "${process_records.results}",
                        "table": "processed_customers",
                        "mode": "replace"
                    }
                },
                {
                    "id": "generate_report",
                    "kind": "operation",
                    "operation": "report_generator",
                    "parameters": {
                        "data": "${process_records.results}",
                        "template": "customer_analysis.html",
                        "output_path": "/reports/customer_report.html"
                    }
                },
                {
                    "id": "send_notification",
                    "kind": "operation",
                    "operation": "send_email",
                    "parameters": {
                        "to": "data-team@company.com",
                        "subject": "Data Processing Complete",
                        "body": "Processed ${process_records.results.length} records successfully."
                    }
                }
            ]
        }
    ]
}
```

### ML Model Training Pipeline

```python
ml_pipeline = {
    "steps": [
        # Data preparation
        {
            "id": "load_dataset",
            "kind": "operation",
            "operation": "dataset_loader",
            "parameters": {
                "source": "s3://ml-bucket/training-data/",
                "format": "parquet"
            }
        },
        
        # Feature engineering in parallel
        {
            "id": "feature_engineering",
            "kind": "parallel",
            "operations": [
                {
                    "id": "numerical_features",
                    "kind": "operation",
                    "operation": "numerical_preprocessor", 
                    "parameters": {
                        "data": "${load_dataset.result}",
                        "columns": ["age", "income", "credit_score"],
                        "scaling": "standard"
                    }
                },
                {
                    "id": "categorical_features",
                    "kind": "operation",
                    "operation": "categorical_preprocessor",
                    "parameters": {
                        "data": "${load_dataset.result}",
                        "columns": ["category", "region", "product_type"],
                        "encoding": "one_hot"
                    }
                },
                {
                    "id": "text_features",
                    "kind": "operation",
                    "operation": "text_vectorizer",
                    "parameters": {
                        "data": "${load_dataset.result}",
                        "text_column": "description",
                        "method": "tfidf",
                        "max_features": 1000
                    }
                }
            ]
        },
        
        # Combine features
        {
            "id": "combine_features",
            "kind": "operation",
            "operation": "feature_combiner",
            "parameters": {
                "numerical": "${numerical_features.result}",
                "categorical": "${categorical_features.result}", 
                "text": "${text_features.result}"
            }
        },
        
        # Train multiple models in parallel
        {
            "id": "train_models",
            "kind": "map",
            "mode": "parallel",
            "iterator": "model_config",
            "inputs": [
                {"name": "random_forest", "params": {"n_estimators": 100, "max_depth": 10}},
                {"name": "gradient_boosting", "params": {"learning_rate": 0.1, "n_estimators": 100}},
                {"name": "logistic_regression", "params": {"C": 1.0, "max_iter": 1000}}
            ],
            "operation": {
                "id": "model_trainer",
                "kind": "operation",
                "operation": "train_model",
                "parameters": {
                    "features": "${combine_features.result.features}",
                    "target": "${combine_features.result.target}",
                    "model_type": "${model_config.name}",
                    "hyperparameters": "${model_config.params}"
                }
            }
        },
        
        # Evaluate and select best model
        {
            "id": "model_evaluation",
            "kind": "operation",
            "operation": "model_evaluator",
            "parameters": {
                "models": "${train_models.results}",
                "test_data": "${combine_features.result.test_set}",
                "metrics": ["accuracy", "precision", "recall", "f1_score"]
            }
        },
        
        # Deploy best model
        {
            "id": "deploy_model",
            "kind": "operation",
            "operation": "model_deployer",
            "parameters": {
                "model": "${model_evaluation.result.best_model}",
                "endpoint": "prod-ml-api",
                "version": "v1.0"
            }
        }
    ]
}
```

### ETL Pipeline with Error Handling

```python
etl_pipeline = {
    "steps": [
        # Extract from multiple sources
        {
            "id": "extract_data",
            "kind": "parallel",
            "operations": [
                {
                    "id": "extract_database",
                    "kind": "operation",
                    "operation": "database_extractor",
                    "parameters": {
                        "connection": "postgresql://prod-db:5432/sales",
                        "query": "SELECT * FROM orders WHERE created_at >= NOW() - INTERVAL '1 day'"
                    },
                    "retries": 3,
                    "timeout": 300.0
                },
                {
                    "id": "extract_api",
                    "kind": "operation", 
                    "operation": "api_extractor",
                    "parameters": {
                        "url": "https://api.partner.com/v1/transactions",
                        "headers": {"Authorization": "Bearer ${env.API_TOKEN}"},
                        "params": {"since": "${workflow.start_time}"}
                    },
                    "retries": 5,
                    "timeout": 60.0,
                    "fail_workflow": False  # Continue even if API fails
                },
                {
                    "id": "extract_files",
                    "kind": "map",
                    "mode": "sequential",
                    "iterator": "file_pattern",
                    "inputs": [
                        "/data/csv/*.csv",
                        "/data/json/*.json", 
                        "/data/xml/*.xml"
                    ],
                    "operation": {
                        "id": "file_extractor",
                        "kind": "operation",
                        "operation": "file_reader",
                        "parameters": {
                            "pattern": "${file_pattern}",
                            "format": "auto"
                        }
                    }
                }
            ]
        },
        
        # Transform data
        {
            "id": "transform_data",
            "kind": "operation",
            "operation": "data_transformer",
            "parameters": {
                "database_data": "${extract_database.result}",
                "api_data": "${extract_api.result}",
                "file_data": "${extract_files.results}",
                "transformations": [
                    "standardize_dates",
                    "normalize_currency", 
                    "deduplicate_records",
                    "validate_business_rules"
                ]
            }
        },
        
        # Load to destinations  
        {
            "id": "load_data",
            "kind": "parallel",
            "operations": [
                {
                    "id": "load_warehouse",
                    "kind": "operation",
                    "operation": "warehouse_loader",
                    "parameters": {
                        "data": "${transform_data.result}",
                        "destination": "snowflake://warehouse/sales_db",
                        "table": "daily_transactions",
                        "mode": "append"
                    }
                },
                {
                    "id": "load_analytics",
                    "kind": "operation",
                    "operation": "analytics_loader", 
                    "parameters": {
                        "data": "${transform_data.result}",
                        "destination": "bigquery://analytics-project/sales_dataset",
                        "partition_by": "transaction_date"
                    }
                },
                {
                    "id": "update_cache",
                    "kind": "operation",
                    "operation": "cache_updater",
                    "parameters": {
                        "data": "${transform_data.result}",
                        "cache_key": "daily_sales_summary",
                        "ttl": 86400
                    }
                }
            ]
        },
        
        # Quality checks and notifications
        {
            "id": "quality_checks",
            "kind": "operation",
            "operation": "data_quality_checker",
            "parameters": {
                "source_count": "${extract_database.result.count}",
                "transformed_count": "${transform_data.result.count}",
                "loaded_count": "${load_warehouse.result.count}",
                "quality_rules": [
                    "row_count_tolerance: 0.05",
                    "null_percentage_max: 0.1",
                    "duplicate_percentage_max: 0.01"
                ]
            }
        },
        
        # Send completion notification
        {
            "id": "send_notification",
            "kind": "operation",
            "operation": "send_slack",
            "parameters": {
                "channel": "#data-engineering",
                "message": {
                    "text": "ETL Pipeline Complete",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*ETL Pipeline Results*\n• Extracted: ${extract_database.result.count} records\n• Transformed: ${transform_data.result.count} records\n• Loaded: ${load_warehouse.result.count} records\n• Quality Score: ${quality_checks.result.score}%"
                            }
                        }
                    ]
                }
            }
        }
    ]
}
```

## Best Practices

### 1. **Step Naming**

Use descriptive step IDs that explain what the step does:

```python
# Good
"id": "validate_customer_data"
"id": "send_welcome_email"  
"id": "calculate_monthly_revenue"

# Avoid
"id": "step1"
"id": "process"
"id": "do_stuff"
```

### 2. **Error Handling Strategy**

Plan for failures with appropriate retry and timeout settings:

```python
{
    "id": "critical_step",
    "kind": "operation",
    "operation": "important_operation",
    "parameters": {...},
    "retries": 3,           # Retry failed operations
    "timeout": 120.0,       # Prevent hanging
    "fail_workflow": True   # Stop entire workflow on failure
}

{
    "id": "optional_step", 
    "kind": "operation",
    "operation": "nice_to_have_operation",
    "parameters": {...},
    "fail_workflow": False  # Continue workflow even if this fails
}
```

### 3. **Efficient Parallelization**

Use parallel processing where operations are independent:

```python
# Process independent operations in parallel
{
    "id": "parallel_notifications",
    "kind": "parallel",
    "operations": [
        {"id": "email", "kind": "operation", "operation": "send_email", ...},
        {"id": "slack", "kind": "operation", "operation": "send_slack", ...},
        {"id": "webhook", "kind": "operation", "operation": "send_webhook", ...}
    ]
}

# Use parallel map for data processing
{
    "id": "process_files",
    "kind": "map", 
    "mode": "parallel",  # Much faster than sequential
    "iterator": "file",
    "inputs": ["file1.txt", "file2.txt", "file3.txt"],
    "operation": {...}
}
```

### 4. **Clear Data Flow**

Make data dependencies explicit with meaningful variable names:

```python
{
    "steps": [
        {
            "id": "extract_customer_data",
            "kind": "operation",
            "operation": "database_query",
            "parameters": {"query": "SELECT * FROM customers"}
        },
        {
            "id": "enrich_customer_profiles",
            "kind": "map",
            "mode": "parallel",
            "iterator": "customer",
            "inputs": "${extract_customer_data.result.rows}",  # Clear data source
            "operation": {
                "id": "customer_enrichment",
                "kind": "operation", 
                "operation": "enrich_customer",
                "parameters": {
                    "customer_id": "${customer.id}",          # Clear field access
                    "customer_email": "${customer.email}"
                }
            }
        }
    ]
}
```

### 5. **Workflow Documentation**

Document complex workflows with comments:

```python
complex_workflow = {
    "steps": [
        {
            # Extract data from multiple sources in parallel for efficiency
            "id": "parallel_extraction", 
            "kind": "parallel",
            "operations": [...]
        },
        {
            # Transform extracted data with validation and enrichment
            "id": "data_transformation",
            "kind": "operation", 
            "operation": "transform_pipeline",
            "parameters": {
                # Apply business rules and data quality checks
                "validation_rules": [...],
                "enrichment_apis": [...]
            }
        }
    ]
}
```

## Running Workflows

### Execute Workflow

```python
import aroflow

# Create client and register plugins
client = aroflow.create(aroflow.BackendType.IN_MEMORY)
client.plugin(DataProcessorPlugin)
client.plugin(EmailPlugin)

# Run workflow
result = client.run(workflow)

# Check results
if result.status == "success":
    print("Workflow completed successfully!")
    for step_result in result.results:
        print(f"Step {step_result.id}: {step_result.status}")
else:
    print(f"Workflow failed: {result.error}")
```

### With Custom Workflow ID

```python
result = client.run(workflow, workflow_id="data_pipeline_2024_01_15")
```

### Examining Results

```python
# Get workflow results in different formats
result_dict = result.to_dict()
result_json = result.to_json()
result_yaml = result.to_yaml()

# Access specific step results
first_step = result.results[0]
print(f"Step: {first_step.id}")
print(f"Status: {first_step.status}")
print(f"Result: {first_step.result}")

# Find step by ID
def find_step_result(results, step_id):
    return next((r for r in results if r.id == step_id), None)

email_result = find_step_result(result.results, "send_notification")
if email_result:
    print(f"Email sent: {email_result.result}")
```

---

**Next:** Explore comprehensive [examples](examples.md) and real-world use cases!