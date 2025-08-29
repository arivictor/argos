# Writing Plugins

Plugins are the heart of AroFlow. They encapsulate your business logic and make it available to workflows. Plugins are Python classes or instances that extend `aroflow.PluginMixin`. You can register either plugin classes or plugin instances with the client, depending on whether your plugin requires configuration or state.

## Plugin Basics

### What is a Plugin?

A plugin is a Python class or instance that extends `aroflow.PluginMixin` and implements an `execute` method. Each plugin declares a unique `plugin_name` that workflows use to reference it within a client.

```python
import aroflow

class MyPlugin(aroflow.PluginMixin):
    plugin_name = "my_plugin"
    
    def execute(self, param1: str, param2: int = 10) -> str:
        return f"Processed {param1} with value {param2}"
```

### Plugin Requirements

Every plugin must:

1. **Inherit from `aroflow.PluginMixin`**
2. **Define a unique `plugin_name`** (string) within the client registry
3. **Implement an `execute` method**

The `plugin_name` must be unique among all plugins registered with the same client, but does not need to be globally unique across different clients.

The `execute` method can have any signature you need. AroFlow maps workflow parameters to the `execute` method arguments by name and performs type coercion for primitive types (e.g., `str`, `int`, `float`, `bool`). It does not automatically handle arbitrary signatures or complex conversions.

## Plugin Examples

### Simple Data Processor

```python
import aroflow

class DataProcessorPlugin(aroflow.PluginMixin):
    plugin_name = "process_data"
    
    def execute(self, data: str, transform: str = "upper") -> str:
        """Process text data with various transformations."""
        if transform == "upper":
            return data.upper()
        elif transform == "lower":
            return data.lower()
        elif transform == "reverse":
            return data[::-1]
        else:
            return data
```

**Usage in workflow:**

```python
{
    "id": "step1",
    "kind": "operation",
    "operation": "process_data",
    "parameters": {
        "data": "Hello World",
        "transform": "upper"
    }
}
```

### File Operation Plugin

```python
import aroflow
import json
from pathlib import Path

class FileOperationPlugin(aroflow.PluginMixin):
    plugin_name = "file_ops"
    
    def execute(self, operation: str, file_path: str, content: str = None) -> dict:
        """Perform various file operations."""
        path = Path(file_path)
        
        if operation == "read":
            if not path.exists():
                raise FileNotFoundError(f"File {file_path} not found")
            return {"content": path.read_text(), "size": path.stat().st_size}
            
        elif operation == "write":
            if content is None:
                raise ValueError("Content required for write operation")
            path.write_text(content)
            return {"message": f"Written to {file_path}", "size": len(content)}
            
        elif operation == "delete":
            if path.exists():
                path.unlink()
                return {"message": f"Deleted {file_path}"}
            else:
                return {"message": f"File {file_path} did not exist"}
                
        else:
            raise ValueError(f"Unknown operation: {operation}")
```

### HTTP Request Plugin

```python
import aroflow
import requests
from typing import Optional, Dict, Any

class HttpRequestPlugin(aroflow.PluginMixin):
    plugin_name = "http_request"
    
    def execute(
        self, 
        url: str, 
        method: str = "GET", 
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> dict:
        """Make HTTP requests."""
        
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers or {},
            json=data if method.upper() != "GET" else None,
            timeout=timeout
        )
        
        # Explicitly parse JSON only if content-type header indicates JSON
        content_type = response.headers.get("content-type", "")
        try:
            json_body = response.json() if "application/json" in content_type else None
        except ValueError:
            json_body = None
        
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text,
            "json": json_body
        }
```

## Parameter Handling

### Type Annotations

AroFlow uses type annotations to convert parameters from workflow definitions where possible:

```python
class TypedPlugin(aroflow.PluginMixin):
    plugin_name = "typed_plugin"
    
    def execute(
        self,
        name: str,           # Will be converted to string
        age: int,            # Will be converted to integer
        height: float,       # Will be converted to float
        active: bool,        # Will be converted to boolean
        tags: list,          # Will remain as list
        metadata: dict       # Will remain as dict
    ) -> dict:
        return {
            "name": name,
            "age": age,
            "height": height,
            "active": active,
            "tags": tags,
            "metadata": metadata
        }
```

### Optional Parameters

Use default values for optional parameters:

```python
class ConfigurablePlugin(aroflow.PluginMixin):
    plugin_name = "configurable"
    
    def execute(
        self,
        required_param: str,
        optional_param: str = "default_value",
        retry_count: int = 3,
        debug: bool = False
    ) -> str:
        result = f"Processing {required_param}"
        if debug:
            result += f" (retries: {retry_count}, optional: {optional_param})"
        return result
```

### Variable Arguments

Plugins can accept variable arguments and keyword arguments:

```python
class FlexiblePlugin(aroflow.PluginMixin):
    plugin_name = "flexible"
    
    def execute(self, operation: str, *args, **kwargs) -> dict:
        return {
            "operation": operation,
            "args": args,
            "kwargs": kwargs
        }
```

## Error Handling

### Raising Exceptions

Plugins should raise meaningful exceptions when errors occur:

```python
class ValidatingPlugin(aroflow.PluginMixin):
    plugin_name = "validator"
    
    def execute(self, email: str, age: int) -> dict:
        # Validate email
        if "@" not in email:
            raise ValueError(f"Invalid email format: {email}")
            
        # Validate age
        if age < 0 or age > 150:
            raise ValueError(f"Invalid age: {age}")
            
        return {"email": email, "age": age, "valid": True}
```

### Custom Exception Classes

For better error handling, create custom exception classes:

```python
class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass

class DataProcessorPlugin(aroflow.PluginMixin):
    plugin_name = "data_processor"
    
    def execute(self, data: str, format: str) -> dict:
        try:
            if format == "json":
                import json
                result = json.loads(data)
            elif format == "csv":
                import csv
                result = list(csv.reader([data]))
            else:
                raise ProcessingError(f"Unsupported format: {format}")
                
            return {"parsed": result, "format": format}
            
        except json.JSONDecodeError as e:
            raise ProcessingError(f"Invalid JSON: {e}")
        except Exception as e:
            raise ProcessingError(f"Processing failed: {e}")
```

## Advanced Plugin Patterns

### Stateful Plugins

> **Note:**  
> Stateful plugins are *not* officially supported in AroFlow. Plugins should generally be stateless, unless they explicitly manage configuration or interact with external state (such as databases or files). If you need to maintain state, consider handling it through external resources or configuration passed to the plugin at initialization.


## Plugin Registration

### Registering Plugins

Register plugins with the client before running workflows:

```python
import aroflow

# Create client
client = aroflow.create(aroflow.BackendType.IN_MEMORY)

# Register multiple plugin classes
client.plugin(DataProcessorPlugin)
client.plugin(FileOperationPlugin)
client.plugin(HttpRequestPlugin)

# Register plugin instances if they require configuration or state
db_plugin = DatabasePlugin("connection_string")
client.plugin(db_plugin)
```

### Plugin Discovery

Plugins are automatically discovered when registered. The `plugin_name` must be unique across all plugins registered with the same client.

## Best Practices

### 1. **Clear Naming**

Use descriptive plugin names that reflect their purpose:

```python
# Good
plugin_name = "send_email"
plugin_name = "process_csv"
plugin_name = "validate_data"

# Avoid
plugin_name = "plugin1"
plugin_name = "helper"
plugin_name = "util"
```

### 2. **Type Annotations**

Always use type annotations for better parameter binding:

```python
def execute(self, data: str, count: int, active: bool = True) -> dict:
    # Type annotations help AroFlow convert parameters correctly
    pass
```

### 3. **Meaningful Return Values**

Return structured data that other steps can use:

```python
def execute(self, input_data: str) -> dict:
    return {
        "result": processed_data,
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "input_size": len(input_data),
            "processing_time": elapsed_time
        }
    }
```

### 4. **Error Messages**

Provide clear, actionable error messages:

```python
def execute(self, file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}. "
            f"Please check the path and ensure the file exists."
        )
```

### 5. **Documentation**

Document your plugins with docstrings:

```python
class EmailPlugin(aroflow.PluginMixin):
    """Send emails using SMTP.
    
    This plugin sends emails through an SMTP server. It supports
    both plain text and HTML messages.
    """
    
    plugin_name = "send_email"
    
    def execute(
        self, 
        to: str, 
        subject: str, 
        body: str,
        html: bool = False
    ) -> dict:
        """Send an email.
        
        Args:
            to: Recipient email address
            subject: Email subject line
            body: Email body content
            html: Whether body contains HTML (default: False)
            
        Returns:
            dict: Sending status and metadata
            
        Raises:
            ValueError: If email address is invalid
            ConnectionError: If SMTP server is unreachable
        """
        # Implementation here
        pass
```

## Testing Plugins

### Unit Testing

Test your plugins independently:

```python
import pytest
from my_plugins import DataProcessorPlugin

def test_data_processor_upper():
    plugin = DataProcessorPlugin()
    result = plugin.execute("hello world", "upper")
    assert result == "HELLO WORLD"

def test_data_processor_invalid_transform():
    plugin = DataProcessorPlugin()
    with pytest.raises(ValueError):
        plugin.execute("hello", "invalid_transform")
```

### Integration Testing

Test plugins within workflows:

```python
def test_plugin_in_workflow():
    client = aroflow.create(aroflow.BackendType.IN_MEMORY)
    client.plugin(DataProcessorPlugin)
    
    workflow = {
        "steps": [{
            "id": "step1",
            "kind": "operation",
            "operation": "process_data",
            "parameters": {"data": "test", "transform": "upper"}
        }]
    }
    
    result = client.run(workflow)
    assert result.status == "success"
    assert result.results[0].result == "TEST"
```

---

**Next:** Learn how to compose these plugins into powerful [workflows](workflows.md)!
