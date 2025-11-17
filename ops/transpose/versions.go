package transpose

import "github.com/sergev/gonnx/ops"

var transposeVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newTranspose, 1, transposeTypeConstraint),
	13: ops.NewOperatorConstructor(newTranspose, 13, transposeTypeConstraint),
}

func GetVersions() ops.OperatorVersions {
	return transposeVersions
}
