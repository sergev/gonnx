package tanh

import "github.com/sergev/gonnx/ops"

var tanhVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newTanh, 6, tanhTypeConstraint),
	13: ops.NewOperatorConstructor(newTanh, 13, tanhTypeConstraint),
}

func GetVersions() ops.OperatorVersions {
	return tanhVersions
}
