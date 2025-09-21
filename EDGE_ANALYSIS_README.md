# Edge Analysis Access

## How to Access Edge Analysis

1. **Direct URL**: Navigate to `http://localhost:5173/analysis` in your browser
2. **From Main Map**: No button available - use direct URL access only

## Download Location

When analysis completes, the JSON file will be downloaded to your browser's default Downloads folder with a filename like:
```
edge_classification_9h_2025-09-20T15-30-45.json
```

The analysis page will show an alert with the exact filename when download completes.

## Example Download Locations

- **Chrome/Edge**: `~/Downloads/`
- **Firefox**: `~/Downloads/` (or configured download folder)
- **Safari**: `~/Downloads/`

## File Format

The downloaded JSON contains:
```json
{
  "timestamp": "2025-09-20T19:47:30.123Z",
  "analysisTime": "2025-09-20T15:47:30.123Z", 
  "totalEdges": 100,
  "processedEdges": 100,
  "errors": 0,
  "processingTimeMs": 8456,
  "edges": [
    {
      "id": "edge_0",
      "shadePct": 0.75,
      "shaded": true,
      "nSamples": 12
    }
    // ... all edge results
  ]
}
```