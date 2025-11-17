package qlinearadd

import (
	"github.com/sergev/gonnx/ops"
)

var qlinearAddVersions = ops.OperatorVersions{
	10: ops.NewOperatorConstructor(newQLinearAdd, 10, qlinearAddTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return qlinearAddVersions
}

