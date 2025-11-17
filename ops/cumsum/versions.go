package cumsum

import (
	"github.com/sergev/gonnx/ops"
)

var cumsumVersions = ops.OperatorVersions{
	11: ops.NewOperatorConstructor(newCumSum, 11, cumsumTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return cumsumVersions
}
