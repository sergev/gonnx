package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/sergev/gonnx"
	"github.com/sergev/gonnx/onnx"
	"gopkg.in/yaml.v3"
)

func main() {
	var modelPath string
	flag.StringVar(&modelPath, "model", "", "Path to the ONNX model file")
	flag.StringVar(&modelPath, "m", "", "Path to the ONNX model file (short)")
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

	// Extract metadata
	metadata := map[string]interface{}{
		"producer_name":    modelProto.ProducerName,
		"producer_version": modelProto.ProducerVersion,
		"domain":           modelProto.Domain,
		"model_version":    modelProto.ModelVersion,
		"doc_string":       modelProto.DocString,
		"ir_version":       modelProto.IrVersion,
		"opset_import":     convertOpsetImports(modelProto.OpsetImport),
	}

	// Extract graph information
	graphData := extractGraphInfo(modelProto, model)

	// Prepare output structure
	output := map[string]interface{}{
		"metadata": metadata,
		"graph":    graphData,
	}

	// Marshal to YAML
	yamlData, err := yaml.Marshal(output)
	if err != nil {
		log.Fatalf("Failed to marshal to YAML: %v", err)
	}

	// Print YAML to stdout
	fmt.Print(string(yamlData))
}

// convertOpsetImports converts opset imports to a YAML-serializable format
func convertOpsetImports(opsetImports []*onnx.OperatorSetIdProto) []map[string]interface{} {
	result := make([]map[string]interface{}, len(opsetImports))
	for i, opset := range opsetImports {
		result[i] = map[string]interface{}{
			"domain":  opset.Domain,
			"version": opset.Version,
		}
	}
	return result
}

// extractGraphInfo extracts graph information from ModelProto and Model
func extractGraphInfo(modelProto *onnx.ModelProto, model *gonnx.Model) map[string]interface{} {
	graph := modelProto.Graph

	// Extract nodes
	nodes := make([]map[string]interface{}, len(graph.Node))
	for i, node := range graph.Node {
		nodeInfo := map[string]interface{}{
			"name":       node.Name,
			"op_type":    node.OpType,
			"domain":     node.Domain,
			"doc_string": node.DocString,
			"input":      node.Input,
			"output":     node.Output,
			"attribute":  convertAttributes(node.Attribute),
		}
		nodes[i] = nodeInfo
	}

	// Extract inputs
	inputs := make([]map[string]interface{}, len(graph.Input))
	for i, input := range graph.Input {
		inputInfo := map[string]interface{}{
			"name":       input.Name,
			"doc_string": input.DocString,
			"type":       convertTypeProto(input.Type),
		}
		inputs[i] = inputInfo
	}

	// Extract outputs
	outputs := make([]map[string]interface{}, len(graph.Output))
	for i, output := range graph.Output {
		outputInfo := map[string]interface{}{
			"name":       output.Name,
			"doc_string": output.DocString,
			"type":       convertTypeProto(output.Type),
		}
		outputs[i] = outputInfo
	}

	// Extract initializers
	initializers := make([]map[string]interface{}, len(graph.Initializer))
	for i, init := range graph.Initializer {
		initInfo := map[string]interface{}{
			"name":        init.Name,
			"doc_string":  init.DocString,
			"data_type":   convertDataTypeToString(init.GetDataType()),
			"dims":        init.Dims,
			"raw_data":    len(init.RawData), // Just store length, not raw bytes
			"double_data": init.DoubleData,
			"float_data":  init.FloatData,
			"int32_data":  init.Int32Data,
			"int64_data":  init.Int64Data,
			"uint64_data": init.Uint64Data,
			"string_data": init.StringData,
		}
		initializers[i] = initInfo
	}

	// Get input/output names and shapes from the model
	inputNames := model.InputNames()
	inputShapes := model.InputShapes()
	inputShapeMap := make(map[string]interface{})
	for _, name := range inputNames {
		if shape, ok := inputShapes[name]; ok {
			inputShapeMap[name] = shape
		}
	}

	outputNames := model.OutputNames()
	outputShapeMap := make(map[string]interface{})
	for _, name := range outputNames {
		shape := model.OutputShape(name)
		outputShapeMap[name] = shape
	}

	return map[string]interface{}{
		"name":          graph.Name,
		"doc_string":    graph.DocString,
		"inputs":        inputs,
		"outputs":       outputs,
		"nodes":         nodes,
		"initializers":  initializers,
		"input_shapes":  inputShapeMap,
		"output_shapes": outputShapeMap,
	}
}

// convertAttributes converts node attributes to a YAML-serializable format
func convertAttributes(attrs []*onnx.AttributeProto) []map[string]interface{} {
	result := make([]map[string]interface{}, len(attrs))
	for i, attr := range attrs {
		attrInfo := map[string]interface{}{
			"name":       attr.Name,
			"doc_string": attr.DocString,
			"type":       attr.Type.String(),
		}

		// Add attribute value based on type
		switch attr.Type {
		case onnx.AttributeProto_FLOAT:
			attrInfo["value"] = attr.F
		case onnx.AttributeProto_INT:
			attrInfo["value"] = attr.I
		case onnx.AttributeProto_STRING:
			attrInfo["value"] = string(attr.S)
		case onnx.AttributeProto_TENSOR:
			attrInfo["value"] = "tensor" // Simplified
		case onnx.AttributeProto_GRAPH:
			attrInfo["value"] = "graph" // Simplified
		case onnx.AttributeProto_FLOATS:
			attrInfo["value"] = attr.Floats
		case onnx.AttributeProto_INTS:
			attrInfo["value"] = attr.Ints
		case onnx.AttributeProto_STRINGS:
			attrInfo["value"] = convertStringData(attr.Strings)
		case onnx.AttributeProto_TENSORS:
			attrInfo["value"] = "tensors" // Simplified
		case onnx.AttributeProto_GRAPHS:
			attrInfo["value"] = "graphs" // Simplified
		}

		result[i] = attrInfo
	}
	return result
}

// convertStringData converts byte slice to string slice
func convertStringData(data [][]byte) []string {
	result := make([]string, len(data))
	for i, b := range data {
		result[i] = string(b)
	}
	return result
}

// convertTypeProto converts TypeProto to a YAML-serializable format
func convertTypeProto(tp *onnx.TypeProto) map[string]interface{} {
	if tp == nil {
		return nil
	}

	result := map[string]interface{}{}

	if tensorType := tp.GetTensorType(); tensorType != nil {
		result["type"] = "tensor"
		result["data_type"] = convertDataTypeToString(int32(tensorType.ElemType))
		result["shape"] = convertShape(tensorType.Shape)
	} else if sequenceType := tp.GetSequenceType(); sequenceType != nil {
		result["type"] = "sequence"
		result["elem_type"] = convertTypeProto(sequenceType.ElemType)
	} else if mapType := tp.GetMapType(); mapType != nil {
		result["type"] = "map"
		result["key_type"] = convertDataTypeToString(int32(mapType.KeyType))
		result["value_type"] = convertTypeProto(mapType.ValueType)
	}

	return result
}

// convertShape converts TensorShapeProto to a YAML-serializable format
func convertShape(shape *onnx.TensorShapeProto) map[string]interface{} {
	if shape == nil {
		return nil
	}

	dims := make([]map[string]interface{}, len(shape.Dim))
	for i, dim := range shape.Dim {
		dimInfo := map[string]interface{}{}
		dimValue := dim.GetDimValue()
		dimParam := dim.GetDimParam()
		if dimValue != 0 {
			dimInfo["value"] = dimValue
		} else if dimParam != "" {
			dimInfo["param"] = dimParam
		}
		dims[i] = dimInfo
	}

	return map[string]interface{}{
		"dims": dims,
	}
}

// convertDataTypeToString converts TensorProto_DataType int32 to string
func convertDataTypeToString(dataType int32) string {
	switch onnx.TensorProto_DataType(dataType) {
	case onnx.TensorProto_UNDEFINED:
		return "UNDEFINED"
	case onnx.TensorProto_FLOAT:
		return "FLOAT"
	case onnx.TensorProto_UINT8:
		return "UINT8"
	case onnx.TensorProto_INT8:
		return "INT8"
	case onnx.TensorProto_UINT16:
		return "UINT16"
	case onnx.TensorProto_INT16:
		return "INT16"
	case onnx.TensorProto_INT32:
		return "INT32"
	case onnx.TensorProto_INT64:
		return "INT64"
	case onnx.TensorProto_STRING:
		return "STRING"
	case onnx.TensorProto_BOOL:
		return "BOOL"
	case onnx.TensorProto_FLOAT16:
		return "FLOAT16"
	case onnx.TensorProto_DOUBLE:
		return "DOUBLE"
	case onnx.TensorProto_UINT32:
		return "UINT32"
	case onnx.TensorProto_UINT64:
		return "UINT64"
	case onnx.TensorProto_COMPLEX64:
		return "COMPLEX64"
	case onnx.TensorProto_COMPLEX128:
		return "COMPLEX128"
	case onnx.TensorProto_BFLOAT16:
		return "BFLOAT16"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", dataType)
	}
}
