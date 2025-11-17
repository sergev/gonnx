# ONNX Model Dump CLI

A command-line application written in Go that reads arbitrary ML models in ONNX format and prints their graph structure and metadata in YAML or Graphviz DOT format.

## Features

- Reads ONNX model files (`.onnx` format)
- Extracts and displays model metadata (producer info, version, domain, opset version, etc.)
- Extracts and displays graph structure (nodes, inputs, outputs, initializers)
- Outputs information in human-readable YAML format (default)
- Generates Graphviz DOT format for visualizing the model graph

## Requirements

- Go 1.21 or later
- Graphviz (optional, for rendering DOT output)

## Installation

### Build from source

```bash
go build -o onnx-dump
```

This will create an executable named `onnx-dump` in the current directory.

### Using Makefile

```bash
make build
```

This will build the application and create the `onnx-dump` executable.

## Usage

### YAML Output (default)

```bash
./onnx-dump -model <path-to-onnx-model>
# or with short flag
./onnx-dump -m <path-to-onnx-model>
```

Example:

```bash
./onnx-dump -model my_model.onnx
```

The output will be printed to stdout in YAML format.

### Graphviz DOT Output

```bash
./onnx-dump -model <path-to-onnx-model> -format dot
# or with short flags
./onnx-dump -m <path-to-onnx-model> -f dot
```

Example:

```bash
./onnx-dump -model my_model.onnx -format dot > model.dot
dot -Tpng model.dot -o model.png
```

This generates a DOT file that can be rendered using Graphviz tools like `dot`, `neato`, or `fdp`.

## Output Formats

### YAML Format

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

### DOT Format

The DOT output creates a directed graph visualization where:

- **Input nodes** are shown as ellipses with light blue fill
- **Operation nodes** are shown as boxes with rounded corners, labeled with the operation type and node name
- **Output nodes** are shown as ellipses with light green fill
- **Edges** represent data flow from outputs to inputs

The DOT format can be rendered using Graphviz:

```bash
# Generate PNG image
dot -Tpng model.dot -o model.png

# Generate SVG image
dot -Tsvg model.dot -o model.svg

# Generate PDF
dot -Tpdf model.dot -o model.pdf
```

## Dependencies

- `github.com/sergev/gonnx` - ONNX model loading and inspection
- `gopkg.in/yaml.v3` - YAML marshaling for output

## License

This project is open source and available under your chosen license.

