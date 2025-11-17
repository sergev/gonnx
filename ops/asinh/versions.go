package asinh

import (
	"github.com/sergev/gonnx/ops"
)

var asinhVersions = ops.OperatorVersions{
	9: ops.NewOperatorConstructor(newAsinh, 9, asinhTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return asinhVersions
}
