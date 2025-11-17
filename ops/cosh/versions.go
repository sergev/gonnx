package cosh

import (
	"github.com/sergev/gonnx/ops"
)

var coshVersions = ops.OperatorVersions{
	9: ops.NewOperatorConstructor(newCosh, 9, coshTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return coshVersions
}
