package mul

import "github.com/sergev/gonnx/ops"

var mulVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newMul, 7, mulTypeConstraints),
	13: ops.NewOperatorConstructor(newMul, 13, mulTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return mulVersions
}
