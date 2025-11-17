package and

import (
	"github.com/sergev/gonnx/ops"
)

var andVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newAnd, 7, andTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return andVersions
}
