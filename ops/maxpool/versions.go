package maxpool

import (
	"github.com/sergev/gonnx/ops"
)

var maxpoolVersions = ops.OperatorVersions{
	8:  ops.NewOperatorConstructor(newMaxPool, 8, maxpoolTypeConstraints),
	10: ops.NewOperatorConstructor(newMaxPool, 10, maxpoolTypeConstraints),
	11: ops.NewOperatorConstructor(newMaxPool, 11, maxpoolTypeConstraints),
	12: ops.NewOperatorConstructor(newMaxPool, 12, maxpoolTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return maxpoolVersions
}

