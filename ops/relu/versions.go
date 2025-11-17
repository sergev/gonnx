package relu

import "github.com/sergev/gonnx/ops"

var reluVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newRelu, 6, reluTypeConstraints),
	13: ops.NewOperatorConstructor(newRelu, 13, reluTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return reluVersions
}
