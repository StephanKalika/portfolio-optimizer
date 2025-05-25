# API Design Guidelines

This document outlines the API design guidelines for the Portfolio Optimizer system to ensure consistency across all microservices.

## General Principles

1. **RESTful Design**: Follow REST principles for API design
2. **Consistency**: Maintain consistent patterns across all APIs
3. **Simplicity**: Keep APIs simple and intuitive
4. **Versioning**: Version all APIs to allow for evolution
5. **Documentation**: Document all APIs thoroughly
6. **Security**: Design with security in mind

## URL Structure

### Base URL Format

```
https://{service-domain}/api/v{version-number}/{resource}
```

Example:
```
https://api.portfolio-optimizer.com/api/v1/models
```

### Resource Naming

- Use nouns, not verbs
- Use lowercase plural nouns for resources
- Use hyphens (`-`) for multi-word resources, not underscores
- Keep URLs as short as possible

✅ Good examples:
- `/api/v1/models`
- `/api/v1/stock-prices`
- `/api/v1/portfolio-optimizations`

❌ Bad examples:
- `/api/v1/get-model` (uses verb)
- `/api/v1/model` (singular)
- `/api/v1/stock_prices` (uses underscore)

### Hierarchy and Relationships

For resource relationships, use nested URLs:

```
/api/v1/{resource}/{resource-id}/{sub-resource}
```

Example:
```
/api/v1/models/market_predictor_20231026/predictions
```

## HTTP Methods

Use appropriate HTTP methods for operations:

- **GET**: Retrieve a resource
- **POST**: Create a new resource
- **PUT**: Update a resource (full update)
- **PATCH**: Partial update of a resource
- **DELETE**: Remove a resource

## Request Parameters

### Path Parameters

Use for identifying a specific resource:

```
/api/v1/models/{model_id}
```

### Query Parameters

Use for filtering, sorting, and pagination:

```
/api/v1/models?type=lstm&limit=10&page=2&sort=created_at
```

Common query parameters:
- `limit`: Number of items to return
- `page` or `offset`: For pagination
- `sort`: Field to sort by
- `order`: Sort order (asc/desc)
- `fields`: Fields to include in the response (sparse fieldsets)

### Request Body

Use for sending data in POST, PUT, and PATCH requests:

```json
{
  "model_name": "market_predictor_20240115",
  "tickers": ["AAPL", "MSFT"],
  "model_type": "lstm",
  "parameters": {
    "sequence_length": 60,
    "epochs": 50
  }
}
```

## Response Format

### Success Responses

All successful responses should have a consistent structure:

```json
{
  "status": "success",
  "data": {
    // Response data here
  },
  "message": "Optional success message"
}
```

### Error Responses

All error responses should have a consistent structure:

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message"
  }
}
```

### HTTP Status Codes

Use appropriate HTTP status codes:

- **200 OK**: Request succeeded
- **201 Created**: Resource created successfully
- **204 No Content**: Success with no content to return
- **400 Bad Request**: Invalid request format or parameters
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Authenticated but not authorized
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation errors
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Service temporarily unavailable

## Pagination

For endpoints returning multiple items, use pagination:

```json
{
  "status": "success",
  "data": {
    "items": [
      // Array of items
    ],
    "pagination": {
      "total_items": 100,
      "total_pages": 10,
      "current_page": 1,
      "items_per_page": 10,
      "next_page": "/api/v1/models?page=2&limit=10",
      "previous_page": null
    }
  }
}
```

## Filtering and Sorting

Allow filtering resources via query parameters:

```
/api/v1/models?model_type=lstm&created_after=2023-01-01
```

Allow sorting via `sort` and `order` parameters:

```
/api/v1/models?sort=created_at&order=desc
```

## API Versioning

Version APIs using a URL path segment:

```
/api/v1/models
/api/v2/models
```

Major version changes should be used for breaking changes:
- Changed response format
- Removed fields
- Changed behavior

## HATEOAS Links

Include hyperlinks in responses to support discoverability:

```json
{
  "status": "success",
  "data": {
    "model_id": "market_predictor_20240115",
    "model_type": "lstm",
    "links": {
      "self": "/api/v1/models/market_predictor_20240115",
      "train": "/api/v1/models/market_predictor_20240115/train",
      "predict": "/api/v1/models/market_predictor_20240115/predict"
    }
  }
}
```

## Documentation

All APIs should be documented using OpenAPI/Swagger:

- Describe all endpoints
- Document all parameters
- Include example requests and responses
- Document error responses
- Add descriptions for all fields

## Caching

Use HTTP caching headers appropriately:

```
Cache-Control: max-age=3600
ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"
Last-Modified: Wed, 21 Oct 2023 07:28:00 GMT
```

## Rate Limiting

Implement rate limiting and communicate limits via headers:

```
X-Rate-Limit-Limit: 100
X-Rate-Limit-Remaining: 95
X-Rate-Limit-Reset: 1626861000
```

## Examples

### Create a Model (POST Request)

Request:
```
POST /api/v1/models HTTP/1.1
Content-Type: application/json

{
  "model_name": "market_predictor_20240115",
  "tickers": ["AAPL", "MSFT"],
  "model_type": "lstm",
  "parameters": {
    "sequence_length": 60,
    "epochs": 50
  }
}
```

Response:
```
HTTP/1.1 201 Created
Content-Type: application/json

{
  "status": "success",
  "data": {
    "model_id": "market_predictor_20240115",
    "created_at": "2024-01-15T12:00:00Z",
    "links": {
      "self": "/api/v1/models/market_predictor_20240115"
    }
  },
  "message": "Model created successfully"
}
```

### Get Model List (GET Request)

Request:
```
GET /api/v1/models?model_type=lstm&limit=2 HTTP/1.1
```

Response:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "success",
  "data": {
    "items": [
      {
        "model_id": "market_predictor_20240115",
        "model_type": "lstm",
        "created_at": "2024-01-15T12:00:00Z",
        "links": {
          "self": "/api/v1/models/market_predictor_20240115"
        }
      },
      {
        "model_id": "market_predictor_20231026",
        "model_type": "lstm",
        "created_at": "2023-10-26T10:30:00Z",
        "links": {
          "self": "/api/v1/models/market_predictor_20231026"
        }
      }
    ],
    "pagination": {
      "total_items": 5,
      "total_pages": 3,
      "current_page": 1,
      "items_per_page": 2,
      "next_page": "/api/v1/models?model_type=lstm&limit=2&page=2",
      "previous_page": null
    }
  }
}
```

### Error Response Example

```
HTTP/1.1 404 Not Found
Content-Type: application/json

{
  "status": "error",
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "The requested model 'invalid_model_id' was not found"
  }
}
``` 