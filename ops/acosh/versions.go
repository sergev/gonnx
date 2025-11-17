package acosh

import (
	"github.com/sergev/gonnx/ops"
)

var acoshVersions = ops.OperatorVersions{
	9: newAcosh,
}

func GetVersions() ops.OperatorVersions {
	return acoshVersions
}
