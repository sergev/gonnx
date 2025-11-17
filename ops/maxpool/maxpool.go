package maxpool

import (
	"github.com/sergev/gonnx/onnx"
	"github.com/sergev/gonnx/ops"
	"gorgonia.org/tensor"
)

var maxpoolTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
}

var (
	MinMaxPoolInputs = 1
	MaxMaxPoolInputs = 1
	NDims1DPooling   = 3
	NDims2DPooling   = 4
	NDims3DPooling   = 5
)

type AutoPadSetting string

const (
	NotSet    AutoPadSetting = "NOTSET"
	SameUpper AutoPadSetting = "SAME_UPPER"
	SameLower AutoPadSetting = "SAME_LOWER"
	Valid     AutoPadSetting = "VALID"
)

// The number of non spatial dimensions inputs will always have.
// For input tensors, the first dimension will be the batch size.
// For all tensors, the second dimension will be the number of channels.
const nNonSpatialDims = 2

// MaxPool represents the ONNX MaxPool operator.
type MaxPool struct {
	ops.BaseOperator

	autoPad      AutoPadSetting
	ceilMode     bool
	dilations    []int
	kernelShape  []int
	pads         []int
	strides      []int
	storageOrder int
}

// newMaxPool creates a new MaxPool operator.
func newMaxPool(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &MaxPool{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinMaxPoolInputs,
			MaxMaxPoolInputs,
			typeConstraints,
			"maxpool",
		),
		autoPad:      NotSet,
		ceilMode:     false,
		storageOrder: 0,
	}
}

// Init initializes the MaxPool operator.
func (m *MaxPool) Init(n *onnx.NodeProto) error {
	var err error

	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "auto_pad":
			m.autoPad = AutoPadSetting(attr.GetS())
		case "ceil_mode":
			m.ceilMode = attr.GetI() != 0
		case "dilations":
			m.dilations, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), m)
			}
		case "kernel_shape":
			m.kernelShape, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), m)
			}
		case "pads":
			m.pads, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), m)
			}
		case "strides":
			m.strides, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), m)
			}
		case "storage_order":
			m.storageOrder = int(attr.GetI())
			if m.storageOrder != 0 {
				return ops.ErrUnsupportedAttribute(attr.GetName(), m)
			}
		default:
			return ops.ErrUnsupportedAttribute(attr.GetName(), m)
		}
	}

	return nil
}

// Apply applies the MaxPool operator.
func (m *MaxPool) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x := inputs[0]

	if len(m.kernelShape) == 0 {
		return nil, ops.ErrInvalidInput("kernel_shape attribute is required", m.BaseOperator)
	}

	if len(m.dilations) == 0 {
		m.setDefaultDilations(x)
	}

	if len(m.pads) == 0 {
		m.setDefaultPaddings(x)
	}

	if len(m.strides) == 0 {
		m.setDefaultStrides(x)
	}

	if m.autoPad != NotSet {
		m.setPaddingWithAutoPad(x)
	}

	var out tensor.Tensor
	var err error

	switch len(x.Shape()) {
	case NDims1DPooling:
		out, err = m.applyMaxPool1D(x)
	case NDims2DPooling:
		out, err = m.applyMaxPool2D(x)
	case NDims3DPooling:
		out, err = m.applyMaxPool3D(x)
	default:
		return nil, ops.ErrInvalidInput("the MaxPool operator currently only supports 1D, 2D or 3D pooling, i.e. shape [N x C x H (x W) (x D)]", m.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// setDefaultDilations sets the dilations attribute to the default.
func (m *MaxPool) setDefaultDilations(x tensor.Tensor) {
	nDims := len(x.Shape()[2:])
	dilations := make([]int, nDims)
	for i := 0; i < nDims; i++ {
		dilations[i] = 1
	}
	m.dilations = dilations
}

// setDefaultPaddings sets default paddings as attribute.
func (m *MaxPool) setDefaultPaddings(x tensor.Tensor) {
	NPadsPerDim := 2
	paddingLength := len(x.Shape()[2:]) * NPadsPerDim
	pads := make([]int, paddingLength)
	for i := 0; i < paddingLength; i++ {
		pads[i] = 0
	}
	m.pads = pads
}

// setDefaultStrides sets default strides as attribute.
func (m *MaxPool) setDefaultStrides(x tensor.Tensor) {
	nDims := len(x.Shape()[2:])
	strides := make([]int, nDims)
	for i := 0; i < nDims; i++ {
		strides[i] = 1
	}
	m.strides = strides
}

// setPaddingWithAutoPad sets the padding attribute based on auto_pad.
func (m *MaxPool) setPaddingWithAutoPad(x tensor.Tensor) {
	if m.autoPad == NotSet {
		return
	}

	NPadsPerDim := 2
	inputShape := x.Shape()
	nDims := len(inputShape)
	nSpatialDims := nDims - nNonSpatialDims

	m.pads = make([]int, nSpatialDims*NPadsPerDim)

	for i := 0; i < nSpatialDims; i++ {
		dim := inputShape[i+nNonSpatialDims]
		effectiveKernelSize := m.kernelShape[i] + (m.kernelShape[i]-1)*(m.dilations[i]-1)

		var padNeeded int
		if m.autoPad == Valid {
			padNeeded = 0
		} else {
			// SameUpper or SameLower
			if m.ceilMode {
				padNeeded = (int(dim)-1)*m.strides[i] + effectiveKernelSize - int(dim)
			} else {
				padNeeded = (int(dim)/m.strides[i]-1)*m.strides[i] + effectiveKernelSize - int(dim)
			}
		}

		var padHead int
		if m.autoPad == SameLower {
			padHead = (padNeeded + 1) / 2
		} else {
			padHead = padNeeded / 2
		}

		padTail := padNeeded - padHead
		m.pads[i] = padHead
		m.pads[i+nSpatialDims] = padTail
	}
}

// getOutputShape calculates the shape of the output tensor.
func (m *MaxPool) getOutputShape(x tensor.Tensor) tensor.Shape {
	outputShape := make([]int, len(x.Shape()))
	outputShape[0] = x.Shape()[0]
	outputShape[1] = x.Shape()[1]

	nSpatialDims := len(x.Shape()) - nNonSpatialDims
	for i := 0; i < nSpatialDims; i++ {
		inputDim := x.Shape()[nNonSpatialDims+i]
		effectiveKernelSize := m.kernelShape[i] + (m.kernelShape[i]-1)*(m.dilations[i]-1)

		var outputDim int
		if m.ceilMode {
			outputDim = int((int64(inputDim) + int64(m.pads[i]) + int64(m.pads[i+nSpatialDims]) - int64(effectiveKernelSize) + int64(m.strides[i]) - 1) / int64(m.strides[i]))
		} else {
			outputDim = int((int64(inputDim)+int64(m.pads[i])+int64(m.pads[i+nSpatialDims])-int64(effectiveKernelSize))/int64(m.strides[i])) + 1
		}

		if outputDim < 1 {
			outputDim = 1
		}
		outputShape[nNonSpatialDims+i] = outputDim
	}

	return outputShape
}

// padInput pads the input with negative infinity for max pooling.
func (m *MaxPool) padInput(x tensor.Tensor) (tensor.Tensor, error) {
	var err error
	nSpatialDims := len(x.Shape()[nNonSpatialDims:])

	for i := 0; i < nSpatialDims; i++ {
		if m.pads[i] != 0 {
			padsBeforeShape := x.Shape().Clone()
			padsBeforeShape[nNonSpatialDims+i] = m.pads[i]
			negInf := tensor.Tensor(tensor.NewDense(x.Dtype(), padsBeforeShape))
			negInf.Zero()
			// Set to negative infinity for max pooling
			if x.Dtype() == tensor.Float32 {
				negInfData := negInf.Data().([]float32)
				for j := range negInfData {
					negInfData[j] = float32(-1e38)
				}
			} else if x.Dtype() == tensor.Float64 {
				negInfData := negInf.Data().([]float64)
				for j := range negInfData {
					negInfData[j] = -1e308
				}
			}

			x, err = tensor.Concat(nNonSpatialDims+i, negInf, x)
			if err != nil {
				return nil, err
			}
		}

		if m.pads[i+nSpatialDims] != 0 {
			padsAfterShape := x.Shape().Clone()
			padsAfterShape[nNonSpatialDims+i] = m.pads[i+nSpatialDims]
			negInf := tensor.Tensor(tensor.NewDense(x.Dtype(), padsAfterShape))
			negInf.Zero()
			// Set to negative infinity for max pooling
			if x.Dtype() == tensor.Float32 {
				negInfData := negInf.Data().([]float32)
				for j := range negInfData {
					negInfData[j] = float32(-1e38)
				}
			} else if x.Dtype() == tensor.Float64 {
				negInfData := negInf.Data().([]float64)
				for j := range negInfData {
					negInfData[j] = -1e308
				}
			}

			x, err = tensor.Concat(nNonSpatialDims+i, x, negInf)
			if err != nil {
				return nil, err
			}
		}
	}

	return x, nil
}

// getSubWindow returns a sub-window for pooling.
func (m *MaxPool) getSubWindow(x tensor.Tensor, batchIdx, channelIdx int, startSpatialCoords ...int) (tensor.Tensor, error) {
	if len(startSpatialCoords) != len(m.kernelShape) {
		return nil, ops.ErrDimension("expected the coordinates to have the same number of dimensions as the kernel")
	}

	slices := []tensor.Slice{
		ops.NewSlicer(batchIdx, batchIdx+1),
		ops.NewSlicer(channelIdx, channelIdx+1),
	}

	for i := 0; i < len(m.kernelShape); i++ {
		dimStartIdx := startSpatialCoords[i]
		dimKernelSize := m.kernelShape[i]
		// Apply dilation
		actualSize := dimKernelSize + (dimKernelSize-1)*(m.dilations[i]-1)
		slices = append(slices, ops.NewSlicer(dimStartIdx, dimStartIdx+actualSize))
	}

	subWindow, err := x.Slice(slices...)
	if err != nil {
		return nil, err
	}

	return subWindow.Materialize(), nil
}

// applyMaxPool1D applies 1D max pooling.
func (m *MaxPool) applyMaxPool1D(x tensor.Tensor) (tensor.Tensor, error) {
	outputShape := m.getOutputShape(x)
	out := tensor.Tensor(tensor.NewDense(x.Dtype(), outputShape))
	out.Zero()

	paddedX, err := m.padInput(x)
	if err != nil {
		return nil, err
	}

	nBatches := x.Shape()[0]
	nChannels := x.Shape()[1]
	outputHDim := outputShape[nNonSpatialDims]

	for batchIdx := 0; batchIdx < nBatches; batchIdx++ {
		for channelIdx := 0; channelIdx < nChannels; channelIdx++ {
			for h := 0; h < paddedX.Shape()[2]; h += m.strides[0] {
				dimHOutputIdx := h / m.strides[0]
				if dimHOutputIdx >= outputHDim {
					continue
				}

				subWindow, err := m.getSubWindow(paddedX, batchIdx, channelIdx, h)
				if err != nil {
					return nil, err
				}

				// Apply dilation by extracting values at dilated positions
				if m.dilations[0] > 1 {
					dilatedData := make([]interface{}, m.kernelShape[0])
					idx := 0
					for i := 0; i < m.kernelShape[0]; i++ {
						pos := i * m.dilations[0]
						if pos < subWindow.Shape()[2] {
							val, _ := subWindow.At(0, 0, pos)
							dilatedData[idx] = val
							idx++
						}
					}
					subWindow = tensor.New(tensor.WithShape(1, 1, idx), tensor.WithBacking(dilatedData[:idx]))
				}

				// Reduce over all spatial dimensions (2, 3, ...)
				// Create a new tensor from the subWindow data to ensure Max() works
				subWindowTensor := tensor.New(tensor.WithBacking(subWindow.Data()), tensor.WithShape(subWindow.Shape()...))
				maxVal, err := subWindowTensor.Max(2)
				if err != nil {
					return nil, err
				}

				err = out.SetAt(maxVal.ScalarValue(), batchIdx, channelIdx, dimHOutputIdx)
				if err != nil {
					return nil, err
				}
			}
		}
	}

	return out, nil
}

// applyMaxPool2D applies 2D max pooling.
func (m *MaxPool) applyMaxPool2D(x tensor.Tensor) (tensor.Tensor, error) {
	outputShape := m.getOutputShape(x)
	out := tensor.Tensor(tensor.NewDense(x.Dtype(), outputShape))
	out.Zero()

	paddedX, err := m.padInput(x)
	if err != nil {
		return nil, err
	}

	nBatches := x.Shape()[0]
	nChannels := x.Shape()[1]
	outputHDim := outputShape[nNonSpatialDims]
	outputWDim := outputShape[nNonSpatialDims+1]

	for batchIdx := 0; batchIdx < nBatches; batchIdx++ {
		for channelIdx := 0; channelIdx < nChannels; channelIdx++ {
			for h := 0; h < paddedX.Shape()[2]; h += m.strides[0] {
				dimHOutputIdx := h / m.strides[0]
				if dimHOutputIdx >= outputHDim {
					continue
				}

				for w := 0; w < paddedX.Shape()[3]; w += m.strides[1] {
					dimWOutputIdx := w / m.strides[1]
					if dimWOutputIdx >= outputWDim {
						continue
					}

					subWindow, err := m.getSubWindow(paddedX, batchIdx, channelIdx, h, w)
					if err != nil {
						return nil, err
					}

					// Apply dilation by extracting values at dilated positions
					if m.dilations[0] > 1 || m.dilations[1] > 1 {
						dilatedH := m.kernelShape[0]
						dilatedW := m.kernelShape[1]
						dilatedData := make([]interface{}, dilatedH*dilatedW)
						idx := 0
						for i := 0; i < m.kernelShape[0]; i++ {
							for j := 0; j < m.kernelShape[1]; j++ {
								hPos := i * m.dilations[0]
								wPos := j * m.dilations[1]
								if hPos < subWindow.Shape()[2] && wPos < subWindow.Shape()[3] {
									val, _ := subWindow.At(0, 0, hPos, wPos)
									dilatedData[idx] = val
									idx++
								}
							}
						}
						subWindow = tensor.New(tensor.WithShape(1, 1, dilatedH, dilatedW), tensor.WithBacking(dilatedData[:idx]))
					}

					// Reduce over all spatial dimensions (2, 3)
					// Create a new tensor from the subWindow data to ensure Max() works
					subWindowTensor := tensor.New(tensor.WithBacking(subWindow.Data()), tensor.WithShape(subWindow.Shape()...))
					maxVal, err := subWindowTensor.Max(2, 3)
					if err != nil {
						return nil, err
					}

					err = out.SetAt(maxVal.ScalarValue(), batchIdx, channelIdx, dimHOutputIdx, dimWOutputIdx)
					if err != nil {
						return nil, err
					}
				}
			}
		}
	}

	return out, nil
}

// applyMaxPool3D applies 3D max pooling.
func (m *MaxPool) applyMaxPool3D(x tensor.Tensor) (tensor.Tensor, error) {
	outputShape := m.getOutputShape(x)
	out := tensor.Tensor(tensor.NewDense(x.Dtype(), outputShape))
	out.Zero()

	paddedX, err := m.padInput(x)
	if err != nil {
		return nil, err
	}

	nBatches := x.Shape()[0]
	nChannels := x.Shape()[1]
	outputHDim := outputShape[nNonSpatialDims]
	outputWDim := outputShape[nNonSpatialDims+1]
	outputDDim := outputShape[nNonSpatialDims+2]

	for batchIdx := 0; batchIdx < nBatches; batchIdx++ {
		for channelIdx := 0; channelIdx < nChannels; channelIdx++ {
			for h := 0; h < paddedX.Shape()[2]; h += m.strides[0] {
				dimHOutputIdx := h / m.strides[0]
				if dimHOutputIdx >= outputHDim {
					continue
				}

				for w := 0; w < paddedX.Shape()[3]; w += m.strides[1] {
					dimWOutputIdx := w / m.strides[1]
					if dimWOutputIdx >= outputWDim {
						continue
					}

					for d := 0; d < paddedX.Shape()[4]; d += m.strides[2] {
						dimDOutputIdx := d / m.strides[2]
						if dimDOutputIdx >= outputDDim {
							continue
						}

						subWindow, err := m.getSubWindow(paddedX, batchIdx, channelIdx, h, w, d)
						if err != nil {
							return nil, err
						}

						// Apply dilation
						if m.dilations[0] > 1 || m.dilations[1] > 1 || m.dilations[2] > 1 {
							dilatedH := m.kernelShape[0]
							dilatedW := m.kernelShape[1]
							dilatedD := m.kernelShape[2]
							dilatedData := make([]interface{}, dilatedH*dilatedW*dilatedD)
							idx := 0
							for i := 0; i < m.kernelShape[0]; i++ {
								for j := 0; j < m.kernelShape[1]; j++ {
									for k := 0; k < m.kernelShape[2]; k++ {
										hPos := i * m.dilations[0]
										wPos := j * m.dilations[1]
										dPos := k * m.dilations[2]
										if hPos < subWindow.Shape()[2] && wPos < subWindow.Shape()[3] && dPos < subWindow.Shape()[4] {
											val, _ := subWindow.At(0, 0, hPos, wPos, dPos)
											dilatedData[idx] = val
											idx++
										}
									}
								}
							}
							subWindow = tensor.New(tensor.WithShape(1, 1, dilatedH, dilatedW, dilatedD), tensor.WithBacking(dilatedData[:idx]))
						}

						// Reduce over all spatial dimensions (2, 3, 4)
						// Create a new tensor from the subWindow data to ensure Max() works
						subWindowTensor := tensor.New(tensor.WithBacking(subWindow.Data()), tensor.WithShape(subWindow.Shape()...))
						maxVal, err := subWindowTensor.Max(2, 3, 4)
						if err != nil {
							return nil, err
						}

						err = out.SetAt(maxVal.ScalarValue(), batchIdx, channelIdx, dimHOutputIdx, dimWOutputIdx, dimDOutputIdx)
						if err != nil {
							return nil, err
						}
					}
				}
			}
		}
	}

	return out, nil
}
