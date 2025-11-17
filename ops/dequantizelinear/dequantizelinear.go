package dequantizelinear

import (
	"github.com/sergev/gonnx/onnx"
	"github.com/sergev/gonnx/ops"
	"gorgonia.org/tensor"
)

var dequantizeLinearTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint8, tensor.Int8},      // x
	{tensor.Float32, tensor.Float64}, // x_scale
	{tensor.Uint8, tensor.Int8},      // x_zero_point (optional)
}

// DequantizeLinear represents the ONNX DequantizeLinear operator.
type DequantizeLinear struct {
	ops.BaseOperator
}

// newDequantizeLinear creates a new DequantizeLinear operator.
func newDequantizeLinear(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &DequantizeLinear{
		BaseOperator: ops.NewBaseOperator(
			version,
			2, // min inputs: x, x_scale
			3, // max inputs: x, x_scale, x_zero_point
			typeConstraints,
			"dequantizelinear",
		),
	}
}

// Init initializes the DequantizeLinear operator.
func (d *DequantizeLinear) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the DequantizeLinear operator.
// Formula: y = (x - x_zero_point) * x_scale
func (d *DequantizeLinear) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x := inputs[0]
	xScale := inputs[1]
	var xZeroPoint tensor.Tensor
	if len(inputs) > 2 && inputs[2] != nil {
		xZeroPoint = inputs[2]
	}

	// Get scale as scalar
	scaleData := xScale.Data()
	var scale float64
	switch v := scaleData.(type) {
	case []float32:
		scale = float64(v[0])
	case []float64:
		scale = v[0]
	default:
		return nil, ops.ErrInvalidInput("x_scale must be float32 or float64", d.BaseOperator)
	}

	// Get zero point as scalar (default to 0 if not provided)
	var zeroPoint int64 = 0
	if xZeroPoint != nil {
		zpData := xZeroPoint.Data()
		switch v := zpData.(type) {
		case []uint8:
			zeroPoint = int64(v[0])
		case []int8:
			zeroPoint = int64(v[0])
		default:
			return nil, ops.ErrInvalidInput("x_zero_point must be uint8 or int8", d.BaseOperator)
		}
	}

	// Perform dequantization
	xData := x.Data()
	xShape := x.Shape()
	var outputData interface{}

	switch xData := xData.(type) {
	case []uint8:
		output := make([]float32, len(xData))
		for i, val := range xData {
			output[i] = float32(int64(val)-zeroPoint) * float32(scale)
		}
		outputData = output
	case []int8:
		output := make([]float32, len(xData))
		for i, val := range xData {
			output[i] = float32(int64(val)-zeroPoint) * float32(scale)
		}
		outputData = output
	default:
		return nil, ops.ErrInvalidInput("x must be uint8 or int8", d.BaseOperator)
	}

	out := tensor.New(tensor.WithShape(xShape...), tensor.WithBacking(outputData))
	return []tensor.Tensor{out}, nil
}

