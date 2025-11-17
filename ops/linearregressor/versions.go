package linearregressor

import "github.com/sergev/gonnx/ops"

var linearRegressorVersions = ops.OperatorVersions{
	1: ops.NewOperatorConstructor(newLinearRegressor, 1, linearRegressorTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return linearRegressorVersions
}
