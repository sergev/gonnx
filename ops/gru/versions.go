package gru

import "github.com/sergev/gonnx/ops"

var gruVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newGRU, 7, gruTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return gruVersions
}
