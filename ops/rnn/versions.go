package rnn

import "github.com/sergev/gonnx/ops"

var rnnVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newRNN, 7, rnnTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return rnnVersions
}
