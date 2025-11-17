package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/sergev/gonnx"
)

func main() {
	var modelPath string
	var outputFormat string
	flag.StringVar(&modelPath, "model", "", "Path to the ONNX model file")
	flag.StringVar(&modelPath, "m", "", "Path to the ONNX model file (short)")
	flag.StringVar(&outputFormat, "format", "yaml", "Output format: yaml or dot")
	flag.StringVar(&outputFormat, "f", "yaml", "Output format: yaml or dot (short)")
	flag.Parse()

	if modelPath == "" {
		flag.Usage()
		fmt.Fprintf(os.Stderr, "\nError: -model flag is required\n")
		os.Exit(1)
	}

	// Read the ONNX file
	bytesModel, err := os.ReadFile(modelPath)
	if err != nil {
		log.Fatalf("Failed to read ONNX model file: %v", err)
	}

	// Parse the ModelProto from bytes
	modelProto, err := gonnx.ModelProtoFromBytes(bytesModel)
	if err != nil {
		log.Fatalf("Failed to parse ONNX model: %v", err)
	}

	// Also create the Model for additional information
	model, err := gonnx.NewModelFromFile(modelPath)
	if err != nil {
		log.Fatalf("Failed to load ONNX model: %v", err)
	}

	// Route to appropriate output format
	outputFormat = strings.ToLower(outputFormat)
	if outputFormat == "dot" {
		dotOutput := generateDotOutput(modelProto, model)
		fmt.Print(dotOutput)
	} else if outputFormat == "yaml" {
		yamlOutput := generateYAMLOutput(modelProto, model)
		fmt.Print(yamlOutput)
	} else {
		log.Fatalf("Invalid output format: %s. Must be 'yaml' or 'dot'", outputFormat)
	}
}
