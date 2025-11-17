package acos

import (
	"github.com/sergev/gonnx/ops"
)

var acosVersions = ops.OperatorVersions{
	7: newAcos,
}

func GetVersions() ops.OperatorVersions {
	return acosVersions
}
