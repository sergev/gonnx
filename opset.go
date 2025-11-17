package gonnx

import (
	"github.com/sergev/gonnx/ops"
	"github.com/sergev/gonnx/ops/abs"
	"github.com/sergev/gonnx/ops/acos"
	"github.com/sergev/gonnx/ops/acosh"
	"github.com/sergev/gonnx/ops/add"
	"github.com/sergev/gonnx/ops/and"
	"github.com/sergev/gonnx/ops/argmax"
	"github.com/sergev/gonnx/ops/asin"
	"github.com/sergev/gonnx/ops/asinh"
	"github.com/sergev/gonnx/ops/atan"
	"github.com/sergev/gonnx/ops/atanh"
	"github.com/sergev/gonnx/ops/cast"
	"github.com/sergev/gonnx/ops/concat"
	"github.com/sergev/gonnx/ops/constant"
	"github.com/sergev/gonnx/ops/constantofshape"
	"github.com/sergev/gonnx/ops/conv"
	"github.com/sergev/gonnx/ops/cos"
	"github.com/sergev/gonnx/ops/cosh"
	"github.com/sergev/gonnx/ops/cumsum"
	"github.com/sergev/gonnx/ops/div"
	"github.com/sergev/gonnx/ops/equal"
	"github.com/sergev/gonnx/ops/erf"
	"github.com/sergev/gonnx/ops/expand"
	"github.com/sergev/gonnx/ops/flatten"
	"github.com/sergev/gonnx/ops/gather"
	"github.com/sergev/gonnx/ops/gemm"
	"github.com/sergev/gonnx/ops/greater"
	"github.com/sergev/gonnx/ops/greaterorequal"
	"github.com/sergev/gonnx/ops/gru"
	"github.com/sergev/gonnx/ops/identity"
	"github.com/sergev/gonnx/ops/less"
	"github.com/sergev/gonnx/ops/lessorequal"
	"github.com/sergev/gonnx/ops/linearregressor"
	"github.com/sergev/gonnx/ops/logsoftmax"
	"github.com/sergev/gonnx/ops/lstm"
	"github.com/sergev/gonnx/ops/matmul"
	"github.com/sergev/gonnx/ops/mul"
	"github.com/sergev/gonnx/ops/not"
	"github.com/sergev/gonnx/ops/or"
	"github.com/sergev/gonnx/ops/pow"
	"github.com/sergev/gonnx/ops/prelu"
	"github.com/sergev/gonnx/ops/reducemax"
	"github.com/sergev/gonnx/ops/reducemean"
	"github.com/sergev/gonnx/ops/reducemin"
	"github.com/sergev/gonnx/ops/relu"
	"github.com/sergev/gonnx/ops/reshape"
	"github.com/sergev/gonnx/ops/rnn"
	"github.com/sergev/gonnx/ops/scaler"
	"github.com/sergev/gonnx/ops/shape"
	"github.com/sergev/gonnx/ops/sigmoid"
	"github.com/sergev/gonnx/ops/sin"
	"github.com/sergev/gonnx/ops/sinh"
	"github.com/sergev/gonnx/ops/slice"
	"github.com/sergev/gonnx/ops/softmax"
	"github.com/sergev/gonnx/ops/sqrt"
	"github.com/sergev/gonnx/ops/squeeze"
	"github.com/sergev/gonnx/ops/sub"
	"github.com/sergev/gonnx/ops/tan"
	"github.com/sergev/gonnx/ops/tanh"
	"github.com/sergev/gonnx/ops/transpose"
	"github.com/sergev/gonnx/ops/unsqueeze"
	"github.com/sergev/gonnx/ops/where"
	"github.com/sergev/gonnx/ops/xor"
)

const (
	MinSupportedOpset = 7
	MaxSupportedOpset = 13
)

// Opset is a set of operators matching a certain opset version.
type Opset map[string]func() ops.Operator

var operators = map[string]ops.OperatorVersions{
	"Abs":             abs.GetVersions(),
	"Acos":            acos.GetVersions(),
	"Acosh":           acosh.GetVersions(),
	"Add":             add.GetVersions(),
	"And":             and.GetVersions(),
	"ArgMax":          argmax.GetVersions(),
	"Asin":            asin.GetVersions(),
	"Asinh":           asinh.GetVersions(),
	"Atan":            atan.GetVersions(),
	"Atanh":           atanh.GetVersions(),
	"Cast":            cast.GetVersions(),
	"Concat":          concat.GetVersions(),
	"Constant":        constant.GetVersions(),
	"ConstantOfShape": constantofshape.GetVersions(),
	"Conv":            conv.GetVersions(),
	"Cos":             cos.GetVersions(),
	"Cosh":            cosh.GetVersions(),
	"CumSum":          cumsum.GetVersions(),
	"Div":             div.GetVersions(),
	"Equal":           equal.GetVersions(),
	"Erf":             erf.GetVersions(),
	"Expand":          expand.GetVersions(),
	"Flatten":         flatten.GetVersions(),
	"Gather":          gather.GetVersions(),
	"Gemm":            gemm.GetVersions(),
	"Greater":         greater.GetVersions(),
	"GreaterOrEqual":  greaterorequal.GetVersions(),
	"GRU":             gru.GetVersions(),
	"Identity":        identity.GetVersions(),
	"Less":            less.GetVersions(),
	"LessOrEqual":     lessorequal.GetVersions(),
	"LinearRegressor": linearregressor.GetVersions(),
	"LogSoftmax":      logsoftmax.GetVersions(),
	"LSTM":            lstm.GetVersions(),
	"MatMul":          matmul.GetVersions(),
	"Mul":             mul.GetVersions(),
	"Not":             not.GetVersions(),
	"Or":              or.GetVersions(),
	"Pow":             pow.GetVersions(),
	"PRelu":           prelu.GetVersions(),
	"ReduceMax":       reducemax.GetVersions(),
	"ReduceMean":      reducemean.GetVersions(),
	"ReduceMin":       reducemin.GetVersions(),
	"Relu":            relu.GetVersions(),
	"Reshape":         reshape.GetVersions(),
	"RNN":             rnn.GetVersions(),
	"Scaler":          scaler.GetVersions(),
	"Shape":           shape.GetVersions(),
	"Sigmoid":         sigmoid.GetVersions(),
	"Sin":             sin.GetVersions(),
	"Sinh":            sinh.GetVersions(),
	"Slice":           slice.GetVersions(),
	"Softmax":         softmax.GetVersions(),
	"Sqrt":            sqrt.GetVersions(),
	"Squeeze":         squeeze.GetVersions(),
	"Sub":             sub.GetVersions(),
	"Tan":             tan.GetVersions(),
	"Tanh":            tanh.GetVersions(),
	"Transpose":       transpose.GetVersions(),
	"Unsqueeze":       unsqueeze.GetVersions(),
	"Xor":             xor.GetVersions(),
	"Where":           where.GetVersions(),
}

// GetClosestOperatorVersion resolves, given a certain opset version, the operator version that is closest
// to that version, going downwards. So if the opset version is 13, and an operator has version 13, this
// one is used. If the opset version is 13, and an operator has versions 7 and 14, version 7 is used, as
// it is the closest opset version going downwards.
func GetClosestOperatorVersion(opsetID int64, versions ops.OperatorVersions) func() ops.Operator {
	for closestOpset := opsetID; closestOpset >= 1; closestOpset-- {
		if operator, ok := versions[closestOpset]; ok {
			return operator
		}
	}

	return nil
}

// ResolveOpset resolves the opset with all closest operator versions for the given opset version.
func ResolveOpset(opsetID int64) (Opset, error) {
	if opsetID < MinSupportedOpset || opsetID > MaxSupportedOpset {
		return nil, ops.ErrUnsupportedOpsetVersion
	}

	opset := map[string]func() ops.Operator{}

	for operatorName, operatorVersions := range operators {
		operator := GetClosestOperatorVersion(opsetID, operatorVersions)
		if operator == nil {
			continue
		}

		opset[operatorName] = operator
	}

	return opset, nil
}
