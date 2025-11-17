package conv

import (
	"github.com/sergev/gonnx/ops"
)

var convVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newConv, 1, convTypeConstraints),
	11: ops.NewOperatorConstructor(newConv, 11, convTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return convVersions
}
