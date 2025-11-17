package sinh

import "github.com/sergev/gonnx/ops"

var sinhVersions = ops.OperatorVersions{
	9: ops.NewOperatorConstructor(newSinh, 9, sinhTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return sinhVersions
}
