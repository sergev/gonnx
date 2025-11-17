package qlinearconv

import (
	"github.com/sergev/gonnx/ops"
)

var qlinearConvVersions = ops.OperatorVersions{
	10: ops.NewOperatorConstructor(newQLinearConv, 10, qlinearConvTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return qlinearConvVersions
}

