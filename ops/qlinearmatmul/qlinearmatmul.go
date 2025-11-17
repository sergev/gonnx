package qlinearmatmul

import (
	"github.com/sergev/gonnx/onnx"
	"github.com/sergev/gonnx/ops"
	"github.com/sergev/gonnx/ops/dequantizelinear"
	"github.com/sergev/gonnx/ops/matmul"
	"github.com/sergev/gonnx/ops/quantizelinear"
	"gorgonia.org/tensor"
)

var qlinearMatMulTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint8},      // a
	{tensor.Float32},    // a_scale
	{tensor.Uint8},      // a_zero_point
	{tensor.Uint8},      // b
	{tensor.Float32},    // b_scale
	{tensor.Uint8},      // b_zero_point
	{tensor.Float32},    // y_scale
	{tensor.Uint8},      // y_zero_point
}

// QLinearMatMul represents the ONNX QLinearMatMul operator.
type QLinearMatMul struct {
	ops.BaseOperator
}

// newQLinearMatMul creates a new QLinearMatMul operator.
func newQLinearMatMul(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &QLinearMatMul{
		BaseOperator: ops.NewBaseOperator(
			version,
			8, // a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
			8,
			typeConstraints,
			"qlinearmatmul",
		),
	}
}

// Init initializes the QLinearMatMul operator.
func (q *QLinearMatMul) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the QLinearMatMul operator.
// This dequantizes inputs, performs MatMul, then quantizes the output.
func (q *QLinearMatMul) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	a := inputs[0]
	aScale := inputs[1]
	aZeroPoint := inputs[2]
	b := inputs[3]
	bScale := inputs[4]
	bZeroPoint := inputs[5]
	yScale := inputs[6]
	yZeroPoint := inputs[7]

	// Dequantize a
	dequantA := &dequantizelinear.DequantizeLinear{}
	dequantA.BaseOperator = ops.NewBaseOperator(10, 3, 3, [][]tensor.Dtype{}, "dequantizelinear")
	aFloat, err := dequantA.Apply([]tensor.Tensor{a, aScale, aZeroPoint})
	if err != nil {
		return nil, err
	}

	// Dequantize b
	dequantB := &dequantizelinear.DequantizeLinear{}
	dequantB.BaseOperator = ops.NewBaseOperator(10, 3, 3, [][]tensor.Dtype{}, "dequantizelinear")
	bFloat, err := dequantB.Apply([]tensor.Tensor{b, bScale, bZeroPoint})
	if err != nil {
		return nil, err
	}

	// Perform MatMul
	matMulOp := &matmul.MatMul{}
	matMulOp.BaseOperator = ops.NewBaseOperator(1, 2, 2, [][]tensor.Dtype{}, "matmul")
	result, err := matMulOp.Apply([]tensor.Tensor{aFloat[0], bFloat[0]})
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

