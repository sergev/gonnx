package quantizelinear

import (
	"github.com/sergev/gonnx/ops"
)

var quantizeLinearVersions = ops.OperatorVersions{
	10: ops.NewOperatorConstructor(newQuantizeLinear, 10, quantizeLinearTypeConstraints),
	13: ops.NewOperatorConstructor(newQuantizeLinear, 13, quantizeLinearTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return quantizeLinearVersions
}

