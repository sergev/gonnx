package main

import (
	"fmt"
	"strings"

	"github.com/sergev/gonnx"
	"github.com/sergev/gonnx/onnx"
)

// generateDotOutput generates Graphviz DOT format output from ONNX model
func generateDotOutput(modelProto *onnx.ModelProto, model *gonnx.Model) string {
	graph := modelProto.Graph
	var sb strings.Builder

	// Graph name - sanitize for DOT
	graphName := sanitizeDotID(graph.Name)
	if graphName == "" {
		graphName = "onnx_model"
	}

	sb.WriteString(fmt.Sprintf("digraph %s {\n", graphName))
	sb.WriteString("  rankdir=LR;\n")
	sb.WriteString("  node [shape=box, style=rounded];\n\n")

	// Track node IDs
	nodeIDs := make(map[string]string) // node name -> sanitized node ID
	nodeCounter := 0

	// Create nodes for graph inputs
	inputIDs := make([]string, 0, len(graph.Input))
	for _, input := range graph.Input {
		nodeID := fmt.Sprintf("input_%d", nodeCounter)
		nodeCounter++
		inputIDs = append(inputIDs, nodeID)
		sanitizedName := sanitizeDotID(input.Name)
		label := fmt.Sprintf("%s\\n[INPUT]", sanitizedName)
		sb.WriteString(fmt.Sprintf("  %s [label=\"%s\", shape=ellipse, style=filled, fillcolor=lightblue];\n", nodeID, label))
	}

	// Create nodes for each ONNX node
	for i, node := range graph.Node {
		// Generate or use node ID
		var nodeID string
		if node.Name != "" {
			nodeID = sanitizeDotID(node.Name)
			// Ensure uniqueness
			if _, exists := nodeIDs[nodeID]; exists {
				nodeID = fmt.Sprintf("%s_%d", nodeID, i)
			}
		} else {
			nodeID = fmt.Sprintf("node_%d", i)
		}
		nodeIDs[node.Name] = nodeID

		// Create label with op_type and optional name
		label := node.OpType
		if node.Name != "" {
			label = fmt.Sprintf("%s\\n%s", node.OpType, sanitizeDotLabel(node.Name))
		}
		sb.WriteString(fmt.Sprintf("  %s [label=\"%s\"];\n", nodeID, label))
	}

	// Create nodes for graph outputs
	outputIDs := make([]string, 0, len(graph.Output))
	for _, output := range graph.Output {
		nodeID := fmt.Sprintf("output_%d", nodeCounter)
		nodeCounter++
		outputIDs = append(outputIDs, nodeID)
		sanitizedName := sanitizeDotID(output.Name)
		label := fmt.Sprintf("%s\\n[OUTPUT]", sanitizedName)
		sb.WriteString(fmt.Sprintf("  %s [label=\"%s\", shape=ellipse, style=filled, fillcolor=lightgreen];\n", nodeID, label))
		// Outputs don't produce tensors, they consume them
	}

	sb.WriteString("\n")

	// Track edges to avoid duplicates
	edges := make(map[string]bool)

	// Helper function to add edge if not already present
	addEdge := func(from, to string) {
		edgeKey := fmt.Sprintf("%s->%s", from, to)
		if !edges[edgeKey] {
			edges[edgeKey] = true
			sb.WriteString(fmt.Sprintf("  %s -> %s;\n", from, to))
		}
	}

	// Create a map of tensor name to source node ID (for graph inputs and node outputs)
	tensorSource := make(map[string]string)
	for i, input := range graph.Input {
		tensorSource[input.Name] = inputIDs[i]
	}

	// Create edges from inputs to nodes and track node outputs
	for _, node := range graph.Node {
		nodeID := nodeIDs[node.Name]
		if nodeID == "" {
			// Fallback if node ID not found
			for idx, n := range graph.Node {
				if n == node {
					nodeID = fmt.Sprintf("node_%d", idx)
					break
				}
			}
		}

		// Create edges from inputs (graph inputs or other nodes) to this node
		for _, nodeInput := range node.Input {
			if sourceID, exists := tensorSource[nodeInput]; exists {
				addEdge(sourceID, nodeID)
			}
		}

		// Track outputs of this node
		for _, outputTensor := range node.Output {
			if outputTensor != "" {
				tensorSource[outputTensor] = nodeID
			}
		}
	}

	// Create edges from nodes to graph outputs
	for idx, graphOutput := range graph.Output {
		if sourceID, exists := tensorSource[graphOutput.Name]; exists {
			addEdge(sourceID, outputIDs[idx])
		}
	}

	sb.WriteString("}\n")
	return sb.String()
}

// sanitizeDotID sanitizes a string to be a valid DOT node ID
func sanitizeDotID(s string) string {
	if s == "" {
		return ""
	}
	// Replace invalid characters with underscores
	result := strings.Builder{}
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			result.WriteRune(r)
		} else {
			result.WriteRune('_')
		}
	}
	id := result.String()
	// Ensure it doesn't start with a number
	if len(id) > 0 && id[0] >= '0' && id[0] <= '9' {
		id = "n" + id
	}
	return id
}

// sanitizeDotLabel sanitizes a string to be used in a DOT label (allows more characters)
func sanitizeDotLabel(s string) string {
	// Escape special characters for DOT labels
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	s = strings.ReplaceAll(s, "\n", "\\n")
	return s
}
