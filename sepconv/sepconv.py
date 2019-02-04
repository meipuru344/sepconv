


# -*- coding: utf-8 -*-
import cupy
import torch
import re
import torch.optim as optim
import torch.nn as nn
import numpy
import PIL
import PIL.Image

arguments_strGt = './images/gt.png'

kernel_Sepconv_updateOutput = '''
	extern "C" __global__ void kernel_Sepconv_updateOutput(
		const int n,
		const float* input,
		const float* vertical,
		const float* horizontal,
		float* output
) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x)
	{
		float dblOutput = 0.0;
		const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
		const int intDepth  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
		const int intY      = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
		const int intX      = ( intIndex                                                    ) % SIZE_3(output);

		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1)
		{
			for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1)
			{
				dblOutput += VALUE_4(input, intSample, intDepth, intY + intFilterY, intX + intFilterX) * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
			}
		}
		output[intIndex] = dblOutput;
	}
	}
'''

kernel_Sepconv_Backward = '''
	extern "C" __global__ void kernel_Sepconv_Backward(
		const int n,
		const float* gradOut,
		const float* input,
		const float* vertical,
		const float* horizontal,
		float* gradVertical,
		float* gradHorizontal
){
				float singleOutH = 0.0;
				float singleOutV = 0.0;
				int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
				const int intSample = ( intIndex / SIZE_3(vertical) / SIZE_2(vertical) / SIZE_1(vertical) ) % SIZE_0(vertical);
				const int intDepthIJ= ( intIndex / SIZE_3(vertical) / SIZE_2(vertical)                    ) % SIZE_1(vertical);
				const int intY      = ( intIndex / SIZE_3(vertical)                                       ) % SIZE_2(vertical);
				const int intX      = ( intIndex                                                          ) % SIZE_3(vertical);

		//printf(" index:%d sample:%d depth%d y:%d x:%d", intIndex, intSample, intDepth, intY, intX);
			for(int imageDepthD = 0; imageDepthD < 3 ; imageDepthD += 1){
				for(int kernelDepthij = 0; kernelDepthij < 51 ; kernelDepthij += 1){
				singleOutH +=
				VALUE_4(vertical, intSample, kernelDepthij, intY, intX) *
				VALUE_4(input, intSample, imageDepthD, intY + intDepthIJ, intX + kernelDepthij) *
				VALUE_4(gradOut, intSample, imageDepthD, intY, intX);
				singleOutV +=
				VALUE_4(horizontal, intSample, kernelDepthij, intY, intX) *
				VALUE_4(input, intSample, imageDepthD, intY + kernelDepthij, intX + intDepthIJ) *
				VALUE_4(gradOut, intSample, imageDepthD, intY, intX);
				}
			}
			gradVertical[intIndex] = singleOutV;
			gradHorizontal[intIndex] = singleOutH;
	}
'''


def cupy_kernel(strFunction, objectVariables):
	strKernel = globals()[strFunction]

	while True:
		objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArg = int(objectMatch.group(2))

		strTensor = objectMatch.group(4)
		intSizes = objectVariables[strTensor].size()

		strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
	# end

	while True:
		objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArgs = int(objectMatch.group(2))
		strArgs = objectMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objectVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
	# end

	return strKernel
# end

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class FunctionSepconv(torch.autograd.Function):
	def __init__(self):
		super(FunctionSepconv, self).__init__()
	# end

	def forward(self, input, vertical, horizontal):
		self.save_for_backward(input, vertical, horizontal)
		input = torch.tensor(input.data, requires_grad=True)
		vertical = torch.tensor(vertical.data, requires_grad=True)
		horizontal = torch.tensor(horizontal.data, requires_grad=True)
		intSample = input.size(0)
		intInputDepth = input.size(1)
		intInputHeight = input.size(2)
		intInputWidth = input.size(3)
		intFilterSize = min(vertical.size(1), horizontal.size(1))
		intOutputHeight = min(vertical.size(2), horizontal.size(2))
		intOutputWidth = min(vertical.size(3), horizontal.size(3))
		output = torch.tensor(input.new_zeros(intSample, intInputDepth, intOutputHeight, intOutputWidth).data, requires_grad=True)
		if True:
			class Stream:
				ptr = torch.cuda.current_stream().cuda_stream
			# end

			n = output.nelement()
			cupy_launch('kernel_Sepconv_updateOutput', cupy_kernel('kernel_Sepconv_updateOutput', {
				'input': input,
				'vertical': vertical,
				'horizontal': horizontal,
				'output': output
			}))(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(), output.data_ptr()],
				stream=Stream
			)

		return output
	# end

#####逆伝搬
	def backward(self, outGrad, outData):
		output = outData[0]
		input = outData[1]
		vertical = outData[2]
		horizontal = outData[3]
		intSample = vertical.size(0)
		intFilterSize = min(vertical.size(1), horizontal.size(1))
		intOutputHeight = min(vertical.size(2), horizontal.size(2))
		intOutputWidth = min(vertical.size(3), horizontal.size(3))
		gradVertical = vertical.new_zeros(intSample, intFilterSize, intOutputHeight, intOutputWidth)
		gradHorizontal = vertical.new_zeros(intSample, intFilterSize, intOutputHeight, intOutputWidth)

		if True:
			class Stream:
				ptr = torch.cuda.current_stream().cuda_stream

			n = gradVertical.nelement()
			cupy_launch('kernel_Sepconv_Backward', cupy_kernel('kernel_Sepconv_Backward', {
			'gradOut': outGrad,
			'vertical': vertical,
			'horizontal': horizontal,
			'input':input
			}))(
				grid=tuple([int((n+1024-1)/1024), 1, 1 ]),
				block=tuple([1024, 1, 1 ]),
				args=[ n, outGrad.data_ptr(), input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(), gradVertical.data_ptr(), gradHorizontal.data_ptr()],
				stream=Stream
			)
		# end

		return gradVertical, gradHorizontal

	# end
# end

class ModuleSepconv(torch.nn.Module):
	def __init__(self):
		super(ModuleSepconv, self).__init__()
	# end

	def forward(self, tensorFirst, tensorSecond):
		return FunctionSepconv()(tensorFirst, tensorSecond)
	# end
# end
