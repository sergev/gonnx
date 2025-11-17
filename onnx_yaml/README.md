# ONNX Model YAML CLI

A command-line application written in Go that reads arbitrary ML models in ONNX format and prints their graph structure and metadata in YAML format.

## Features

- Reads ONNX model files (`.onnx` format)
- Extracts and displays model metadata (producer info, version, domain, opset version, etc.)
- Extracts and displays graph structure (nodes, inputs, outputs, initializers)
- Outputs all information in human-readable YAML format

## Requirements

- Go 1.21 or later

## Installation

### Build from source

```bash
go build -o onnx-yaml
```

This will create an executable named `onnx-yaml` in the current directory.

### Using Makefile

```bash
make build
```

This will build the application and create the `onnx-yaml` executable.

## Usage

```bash
./onnx-yaml -model <path-to-onnx-model>
# or with short flag
./onnx-yaml -m <path-to-onnx-model>
```

Example:

```bash
./onnx-yaml -model my_model.onnx
```

The output will be printed to stdout in YAML format.

## Output Format

The YAML output includes:

- **metadata**: Model metadata including producer name/version, domain, model version, IR version, and opset imports
- **graph**: Graph structure including:
  - `name`: Graph name
  - `doc_string`: Documentation string
  - `inputs`: Input tensor definitions with types and shapes
  - `outputs`: Output tensor definitions with types and shapes
  - `nodes`: All graph nodes with operations, inputs, outputs, and attributes
  - `initializers`: Constant values used in the graph
  - `input_shapes`: Input tensor shapes
  - `output_shapes`: Output tensor shapes

## Dependencies

- `github.com/sergev/gonnx` - ONNX model loading and inspection
- `gopkg.in/yaml.v3` - YAML marshaling for output

## License

This project is open source and available under your chosen license.
