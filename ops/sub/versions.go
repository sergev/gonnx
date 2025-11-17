package sub

import "github.com/sergev/gonnx/ops"

var subVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newSub, 7, subTypeConstraints),
	13: ops.NewOperatorConstructor(newSub, 13, subTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return subVersions
}
