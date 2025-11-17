package qlinearmatmul

import (
	"github.com/sergev/gonnx/ops"
)

var qlinearMatMulVersions = ops.OperatorVersions{
	10: ops.NewOperatorConstructor(newQLinearMatMul, 10, qlinearMatMulTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return qlinearMatMulVersions
}

