package dequantizelinear

import (
	"github.com/sergev/gonnx/ops"
)

var dequantizeLinearVersions = ops.OperatorVersions{
	10: ops.NewOperatorConstructor(newDequantizeLinear, 10, dequantizeLinearTypeConstraints),
	13: ops.NewOperatorConstructor(newDequantizeLinear, 13, dequantizeLinearTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return dequantizeLinearVersions
}

