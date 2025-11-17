package qlinearconv

import (
	"github.com/sergev/gonnx/onnx"
	"github.com/sergev/gonnx/ops"
	"github.com/sergev/gonnx/ops/conv"
	"github.com/sergev/gonnx/ops/dequantizelinear"
	"github.com/sergev/gonnx/ops/quantizelinear"
	"gorgonia.org/tensor"
)

var qlinearConvTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint8},      // x
	{tensor.Float32},    // x_scale
	{tensor.Uint8},      // x_zero_point
	{tensor.Uint8},      // w
	{tensor.Float32},    // w_scale
	{tensor.Uint8},      // w_zero_point
	{tensor.Float32},    // y_scale
	{tensor.Uint8},      // y_zero_point
	{tensor.Int32},      // B (optional bias)
}

// QLinearConv represents the ONNX QLinearConv operator.
type QLinearConv struct {
	ops.BaseOperator
	convOp *conv.Conv
}

// newQLinearConv creates a new QLinearConv operator.
func newQLinearConv(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &QLinearConv{
		BaseOperator: ops.NewBaseOperator(
			version,
			8, // min: x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point
			9, // max: + optional B
			typeConstraints,
			"qlinearconv",
		),
	}
}

// Init initializes the QLinearConv operator.
func (q *QLinearConv) Init(n *onnx.NodeProto) error {
	// Create a Conv operator to handle attributes
	q.convOp = &conv.Conv{}
	q.convOp.BaseOperator = ops.NewBaseOperator(11, 2, 3, [][]tensor.Dtype{}, "conv")
	return q.convOp.Init(n)
}

// Apply applies the QLinearConv operator.
// This dequantizes inputs, performs Conv, then quantizes the output.
func (q *QLinearConv) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x := inputs[0]
	xScale := inputs[1]
	xZeroPoint := inputs[2]
	w := inputs[3]
	wScale := inputs[4]
	wZeroPoint := inputs[5]
	yScale := inputs[6]
	yZeroPoint := inputs[7]
	var b tensor.Tensor
	if len(inputs) > 8 && inputs[8] != nil {
		b = inputs[8]
	}

	// Dequantize x
	dequantX := &dequantizelinear.DequantizeLinear{}
	dequantX.BaseOperator = ops.NewBaseOperator(10, 3, 3, [][]tensor.Dtype{}, "dequantizelinear")
	xFloat, err := dequantX.Apply([]tensor.Tensor{x, xScale, xZeroPoint})
	if err != nil {
		return nil, err
	}

	// Dequantize w
	dequantW := &dequantizelinear.DequantizeLinear{}
	dequantW.BaseOperator = ops.NewBaseOperator(10, 3, 3, [][]tensor.Dtype{}, "dequantizelinear")
	wFloat, err := dequantW.Apply([]tensor.Tensor{w, wScale, wZeroPoint})
	if err != nil {
		return nil, err
	}

	// Prepare conv inputs
	convInputs := []tensor.Tensor{xFloat[0], wFloat[0]}
	if b != nil {
		// Bias is already int32, but we may need to convert it to float
		// For simplicity, we'll convert it to float32
		bData := b.Data()
		var bFloat tensor.Tensor
		switch bData := bData.(type) {
		case []int32:
			bFloatData := make([]float32, len(bData))
			for i, val := range bData {
				bFloatData[i] = float32(val)
			}
			bFloat = tensor.New(tensor.WithShape(b.Shape()...), tensor.WithBacking(bFloatData))
		default:
			bFloat = b
		}
		convInputs = append(convInputs, bFloat)
	}

	// Perform Conv
	result, err := q.convOp.Apply(convInputs)
	if err != nil {
		return nil, err
	}

	// Quantize output
	quantY := &quantizelinear.QuantizeLinear{}
	quantY.BaseOperator = ops.NewBaseOperator(10, 3, 3, [][]tensor.Dtype{}, "quantizelinear")
	output, err := quantY.Apply([]tensor.Tensor{result[0], yScale, yZeroPoint})
	if err != nil {
		return nil, err
	}

	return output, nil
}

