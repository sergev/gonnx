package atan

import (
	"github.com/sergev/gonnx/ops"
)

var atanVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newAtan, 7, atanTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return atanVersions
}
