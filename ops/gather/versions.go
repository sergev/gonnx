package gather

import "github.com/sergev/gonnx/ops"

var gatherVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newGather, 1, gatherTypeConstraints),
	11: ops.NewOperatorConstructor(newGather, 11, gatherTypeConstraints),
	13: ops.NewOperatorConstructor(newGather, 13, gatherTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return gatherVersions
}
