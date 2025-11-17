package quantizelinear

import (
	"math"
	"github.com/sergev/gonnx/onnx"
	"github.com/sergev/gonnx/ops"
	"gorgonia.org/tensor"
)

var quantizeLinearTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64}, // x
	{tensor.Float32, tensor.Float64}, // y_scale
	{tensor.Uint8, tensor.Int8},      // y_zero_point (optional)
}

// QuantizeLinear represents the ONNX QuantizeLinear operator.
type QuantizeLinear struct {
	ops.BaseOperator
}

// newQuantizeLinear creates a new QuantizeLinear operator.
func newQuantizeLinear(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &QuantizeLinear{
		BaseOperator: ops.NewBaseOperator(
			version,
			2, // min inputs: x, y_scale
			3, // max inputs: x, y_scale, y_zero_point
			typeConstraints,
			"quantizelinear",
		),
	}
}

// Init initializes the QuantizeLinear operator.
func (q *QuantizeLinear) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the QuantizeLinear operator.
// Formula: y = saturate(round(x / y_scale) + y_zero_point)
func (q *QuantizeLinear) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x := inputs[0]
	yScale := inputs[1]
	var yZeroPoint tensor.Tensor
	if len(inputs) > 2 && inputs[2] != nil {
		yZeroPoint = inputs[2]
	}

	// Get scale as scalar
	scaleData := yScale.Data()
	var scale float64
	switch v := scaleData.(type) {
	case []float32:
		scale = float64(v[0])
	case []float64:
		scale = v[0]
	default:
		return nil, ops.ErrInvalidInput("y_scale must be float32 or float64", q.BaseOperator)
	}

	// Get zero point as scalar (default to 0 if not provided)
	var zeroPoint int64 = 0
	if yZeroPoint != nil {
		zpData := yZeroPoint.Data()
		switch v := zpData.(type) {
		case []uint8:
			zeroPoint = int64(v[0])
		case []int8:
			zeroPoint = int64(v[0])
		default:
			return nil, ops.ErrInvalidInput("y_zero_point must be uint8 or int8", q.BaseOperator)
		}
	}

	// Determine output dtype based on zero point dtype
	var outputDtype tensor.Dtype
	if yZeroPoint != nil {
		outputDtype = yZeroPoint.Dtype()
	} else {
		outputDtype = tensor.Uint8 // Default to uint8
	}

	// Perform quantization
	xData := x.Data()
	xShape := x.Shape()
	var outputData interface{}

	switch xData := xData.(type) {
	case []float32:
		if outputDtype == tensor.Uint8 {
			output := make([]uint8, len(xData))
			for i, val := range xData {
				quantized := int64(math.Round(float64(val)/scale)) + zeroPoint
				// Saturate to uint8 range [0, 255]
				if quantized < 0 {
					quantized = 0
				} else if quantized > 255 {
					quantized = 255
				}
				output[i] = uint8(quantized)
			}
			outputData = output
		} else if outputDtype == tensor.Int8 {
			output := make([]int8, len(xData))
			for i, val := range xData {
				quantized := int64(math.Round(float64(val)/scale)) + zeroPoint
				// Saturate to int8 range [-128, 127]
				if quantized < -128 {
					quantized = -128
				} else if quantized > 127 {
					quantized = 127
				}
				output[i] = int8(quantized)
			}
			outputData = output
		}
	case []float64:
		if outputDtype == tensor.Uint8 {
			output := make([]uint8, len(xData))
			for i, val := range xData {
				quantized := int64(math.Round(val/scale)) + zeroPoint
				// Saturate to uint8 range [0, 255]
				if quantized < 0 {
					quantized = 0
				} else if quantized > 255 {
					quantized = 255
				}
				output[i] = uint8(quantized)
			}
			outputData = output
		} else if outputDtype == tensor.Int8 {
			output := make([]int8, len(xData))
			for i, val := range xData {
				quantized := int64(math.Round(val/scale)) + zeroPoint
				// Saturate to int8 range [-128, 127]
				if quantized < -128 {
					quantized = -128
				} else if quantized > 127 {
					quantized = 127
				}
				output[i] = int8(quantized)
			}
			outputData = output
		}
	default:
		return nil, ops.ErrInvalidInput("x must be float32 or float64", q.BaseOperator)
	}

	out := tensor.New(tensor.WithShape(xShape...), tensor.WithBacking(outputData))
	return []tensor.Tensor{out}, nil
}

