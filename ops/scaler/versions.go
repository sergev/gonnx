package scaler

import "github.com/sergev/gonnx/ops"

var scalerVersions = ops.OperatorVersions{
	1: ops.NewOperatorConstructor(newScaler, 1, scalerTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return scalerVersions
}
