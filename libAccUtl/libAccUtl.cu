
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cub.cuh"
#include "libAccUtl.h"
#include <stdio.h>
#ifdef __cplusplus
#define AUEXTC extern "C"
#else
#define AUEXTC
#endif
///:::::::::::::: KERNELS ::::::::::::::::::://
#define TILE_DIM 16
#define BLOCK_ROWS 16

__global__ void kYC48_F32Planar(float* dY, float* dU, float* dV, void* dYC48, int px_count)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= px_count) return;
	short thread_data[3];
	cub::LoadDirectBlocked(tid, (short*)dYC48, thread_data);
	__syncthreads();
	dY[tid] = (float)thread_data[0];
	dU[tid] = (float)thread_data[1];
	dV[tid] = (float)thread_data[2];
	__syncthreads();
}


__global__ void kF32Planar_YC48(void* dYC48, float* dY, float* dU, float* dV, int px_count)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= px_count) return;
	short thread_data[3];
	thread_data[0] = (short)dY[tid];
	thread_data[1] = (short)dU[tid];
	thread_data[2] = (short)dV[tid];
	__syncthreads();
	cub::StoreDirectBlocked(tid, (short*)dYC48, thread_data);
}


__global__ void short2float(void* d_dst, const void* d_src, int int_count)
{
	__shared__ extern int Buffer[]; // = BlockSize * sizeof(int)
	short* sBuffer = (short*)Buffer;
	int tid = threadIdx.x + blockDim.x* blockIdx.x;
	for (int i = tid; i < int_count; i += gridDim.x* blockDim.x)
	{
		Buffer[threadIdx.x] = *((int*)d_src + i);
		__syncthreads();
		float* d_dstf = (float*)d_dst;
		d_dstf[i * 2] = (float)sBuffer[threadIdx.x * 2];
		d_dstf[i * 2 + 1] = (float)sBuffer[threadIdx.x * 2 + 1];
		__syncthreads();
	}
}

__global__ void float2short(void* d_dst, const void* d_src, int float_count)
{
	__shared__ extern float BufferF[];
	int tid = threadIdx.x + blockDim.x* blockIdx.x;
	for (int i = tid; i < float_count; i += gridDim.x* blockDim.x)
	{
		BufferF[threadIdx.x] = *((float*)d_src + i);
		__syncthreads();
		short datum;
		datum = static_cast<short>(rintf(BufferF[threadIdx.x]));
		short* dst_sh = (short*)d_dst;
		dst_sh[i] = datum;
		__syncthreads();
	}
}


__global__ void kSaxpy(float* d_src, float* d_dst, int elem_count, float offset, float multiplier)
{
	__shared__ extern float Buffer_saxpy[];
	int tid = threadIdx.x + blockDim.x* blockIdx.x;
	for (int i = tid; i < elem_count; i += gridDim.x* blockDim.x)
	{
		Buffer_saxpy[threadIdx.x] = d_src[i];
		__syncthreads();
		d_dst[i] = Buffer_saxpy[threadIdx.x] * multiplier + offset;
		__syncthreads();
	}
}

__global__ void kColorTwist(float* d_src, float* d_dst, int px_count, float matrix[9])
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= px_count) return;
	float thread_data[3];
	cub::LoadDirectBlocked(tid, d_src, thread_data);
	__syncthreads();
	float x, y, z;
	x = thread_data[0] * matrix[0] + thread_data[1] * matrix[1] + thread_data[2] * matrix[2];
	y = thread_data[0] * matrix[3] + thread_data[1] * matrix[4] + thread_data[2] * matrix[5];
	z = thread_data[0] * matrix[6] + thread_data[1] * matrix[7] + thread_data[2] * matrix[8];
	thread_data[0] = x;
	thread_data[1] = y;
	thread_data[2] = z;
	cub::StoreDirectBlocked(tid, d_dst, thread_data);
}


__global__ void transposeDiagonal(float *odata, float *idata, int width, int height)
{

	__shared__ float tile[TILE_DIM][TILE_DIM + 1];

	int blockIdx_x, blockIdx_y;

	// do diagonal reordering
	if (width == height)
	{
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	}
	else
	{
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid%gridDim.y;
		blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
	}

	// from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
	// and similarly for y

	int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;

	xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;

	for (int i = 0; i<TILE_DIM; i += BLOCK_ROWS)
	{
		tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i*width];
	}

	__syncthreads();

	for (int i = 0; i<TILE_DIM; i += BLOCK_ROWS)
	{
		odata[index_out + i*height] = tile[threadIdx.x][threadIdx.y + i];
	}
}


///:::::::::::::: HOST functions ::::::::::::::::://

AUEXTC __declspec(dllexport) int GPUMemAlloc(void** device_ptr, size_t size_byte)
{
	cudaError_t cuStatus = cudaSuccess;
	cuStatus = cudaMalloc(device_ptr, size_byte);
	if (cuStatus != cudaSuccess)
	{
		MessageBox(NULL, cudaGetErrorString(cuStatus), "GPUMemAlloc Error", MB_OK | MB_ICONERROR);
		device_ptr = NULL;
		return FALSE;
	}
	CubDebugExit(cudaMemset(*device_ptr, 0, size_byte));
	return TRUE;
}

AUEXTC __declspec(dllexport) int GPUMemFree(void* device_ptr)
{
	if (!device_ptr) return FALSE;
	if (cudaFree(device_ptr) != cudaSuccess)
	{
		return FALSE;
	}
	device_ptr = NULL;
	return TRUE;
}

AUEXTC __declspec(dllexport) int GPUMemcpy_DD(void* d_dst, void* d_src, size_t size_byte)
{
	if (cudaMemcpy(d_dst, d_src, size_byte, cudaMemcpyDeviceToDevice) != cudaSuccess)
	{
		return FALSE;
	}
	else
	{
		return TRUE;
	}
}

AUEXTC __declspec(dllexport) int GPUMemcpy_DH(void* h_dst, void* d_src, size_t size_byte)
{
	if (cudaMemcpy(h_dst, d_src, size_byte, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		return FALSE;
	}
	else
	{
		return TRUE;
	}
}

AUEXTC __declspec(dllexport) int GPUMemcpy_HD(void* d_dst, void* h_src, size_t size_byte)
{
	if (cudaMemcpy(d_dst, h_src, size_byte, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		return FALSE;
	}
	else
	{
		return TRUE;
	}
}


AUEXTC __declspec(dllexport) int GPUEndSession()
{
	if (cudaDeviceReset() != cudaSuccess)
	{
		return FALSE;
	}
	else
	{
		return TRUE;
	}
}



AUEXTC __declspec(dllexport) int F32Packed_Planar(float* d_packed, float* d_Y, float* d_U, float* d_V, const int width_px, const int height_px)
{
	if (!d_packed || !d_Y || !d_U || !d_V) return FALSE;
	size_t src_stride_byte = width_px * 3 * sizeof(float);
	size_t dst_stride_byte = width_px * sizeof(float);
	for (int c = 0; c < width_px; c++)
	{
		size_t src_offset = c * 3;
		CubDebugExit(cudaMemcpy2DAsync(d_Y + c, dst_stride_byte, d_packed + src_offset, src_stride_byte, sizeof(float), height_px, cudaMemcpyDeviceToDevice));
		CubDebugExit(cudaMemcpy2DAsync(d_U + c, dst_stride_byte, d_packed + src_offset +1, src_stride_byte, sizeof(float), height_px, cudaMemcpyDeviceToDevice));
		CubDebugExit(cudaMemcpy2DAsync(d_V + c, dst_stride_byte, d_packed + src_offset +2, src_stride_byte, sizeof(float), height_px, cudaMemcpyDeviceToDevice));

	}
	cudaDeviceSynchronize();
	CubDebugExit(cudaGetLastError());
	return TRUE;
}


AUEXTC __declspec(dllexport) int F32Planar_Packed(float* d_packed, float* d_Y, float* d_U, float* d_V, const int width_px, const int height_px)
{
	if (!d_packed || !d_Y || !d_U || !d_V) return FALSE;
	size_t dst_stride_byte = width_px * 3 * sizeof(float);
	size_t src_stride_byte = width_px * sizeof(float);
	for (int c = 0; c < width_px; c++)
	{
		size_t dst_offset = c * 3;
		CubDebugExit(cudaMemcpy2DAsync(d_packed + dst_offset, dst_stride_byte, d_Y + c, src_stride_byte, sizeof(float), height_px, cudaMemcpyDeviceToDevice));
		CubDebugExit(cudaMemcpy2DAsync(d_packed + dst_offset +1, dst_stride_byte, d_U + c, src_stride_byte, sizeof(float), height_px, cudaMemcpyDeviceToDevice));
		CubDebugExit(cudaMemcpy2DAsync(d_packed + dst_offset +2, dst_stride_byte, d_V + c, src_stride_byte, sizeof(float), height_px, cudaMemcpyDeviceToDevice));
	}
	cudaDeviceSynchronize();
	CubDebugExit(cudaGetLastError());
	return TRUE;
}


AUEXTC __declspec(dllexport) int YC48H_F32DPacked(float* d_dst, PIXEL_YC* h_src, const int width_px, const int height_px, const int src_stride_px)
{
	if (!d_dst || !h_src) return FALSE;
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	//Get System data
	cudaDeviceProp devInfo = { 0 };
	int devID = 0;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&devInfo, devID);
	//
	cub::CachingDeviceAllocator gAlloc(true); //Device memory allocator
	int mWidth = width_px;
	int mHeight = height_px;
	int* yc_int = reinterpret_cast<int*>(h_src);
	int elem_count_int = (int)ceilf((float)mWidth*(float)mHeight*1.5f); //element counts as int, rounded up
	///allocate device mem
	int* d_src;
	CubDebugExit(gAlloc.DeviceAllocate((void**)&d_src, elem_count_int*sizeof(int))); //over-allocate a bit
	///Copy data
	CubDebugExit(cudaMemcpy2D((void*)d_src, mWidth*sizeof(PIXEL_YC), (void*)yc_int, src_stride_px* sizeof(PIXEL_YC), mWidth*sizeof(PIXEL_YC), mHeight, cudaMemcpyHostToDevice));
	/// short -> float
	//float* d_f32;
	//CubDebugExit(gAlloc.DeviceAllocate((void**)&d_f32, elem_count_int * 2 * sizeof(float))); //each int will break into 2 floats
	int BlockSize = devInfo.maxThreadsPerBlock;
	int GridSize = (BlockSize >= mWidth*mHeight) ? 1 : 32;
	size_t sMemSize = BlockSize * sizeof(int);
	short2float <<<GridSize, BlockSize, sMemSize >>>(d_dst, d_src, elem_count_int);
	cudaDeviceSynchronize();
	CubDebugExit(cudaGetLastError());
	CubDebugExit(gAlloc.DeviceFree(d_src));
#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "YC48H->F32Packed@ %d x %d: %.2f ms", mWidth, mHeight, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;
}


/*AUEXTC __declspec(dllexport) int YC48H_F32DPlanar(float* d_dstY, float* d_dstU, float* d_dstV, PIXEL_YC* h_src, const int width_px, const int height_px, const int src_stride_px)
{
	if (!d_dstY || !d_dstU || !d_dstV || !h_src) return FALSE;
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	//Get System data
	cudaDeviceProp devInfo = { 0 };
	int devID = 0;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&devInfo, devID);
	//
	cub::CachingDeviceAllocator gAlloc(true); //Device memory allocator
	int mWidth = width_px;
	int mHeight = height_px;
	int* yc_int = reinterpret_cast<int*>(h_src);
	int elem_count_int = (int)ceilf((float)mWidth*(float)mHeight*1.5f); //element counts as int, rounded up
	///allocate device mem
	int* d_src;
	CubDebugExit(gAlloc.DeviceAllocate((void**)&d_src, elem_count_int*sizeof(int))); //over-allocate a bit
	///Copy data
	CubDebugExit(cudaMemcpy2D((void*)d_src, mWidth*sizeof(PIXEL_YC), (void*)yc_int, src_stride_px* sizeof(PIXEL_YC), mWidth*sizeof(PIXEL_YC), mHeight, cudaMemcpyHostToDevice));
	/// short -> float
	float* d_f32;
	CubDebugExit(gAlloc.DeviceAllocate((void**)&d_f32, elem_count_int * 2 * sizeof(float))); //each int will break into 2 floats
	int BlockSize = devInfo.maxThreadsPerBlock;
	int GridSize = (BlockSize >= mWidth*mHeight) ? 1 : 32;
	size_t sMemSize = BlockSize * sizeof(int);
	short2float <<<GridSize, BlockSize, sMemSize >>>(d_f32, d_src, elem_count_int);
	cudaDeviceSynchronize();
	CubDebugExit(cudaGetLastError());
	CubDebugExit(gAlloc.DeviceFree(d_src));
	/// packed ->planar
	for (int c = 0; c < mWidth; c++)
	{
		size_t src_offset0 = c * 3; //Y
		size_t src_offset1 = src_offset0 + 1; //U
		size_t src_offset2 = src_offset0 + 2; //V
		size_t dst_pitch = mWidth* sizeof(float);
		size_t src_pitch = mWidth * 3 * sizeof(float);
		//Use Async copy to speed up
		CubDebug(cudaMemcpy2DAsync(d_dstY + c, dst_pitch, d_f32 + src_offset0, src_pitch, sizeof(float), mHeight, cudaMemcpyDeviceToDevice));
		CubDebug(cudaMemcpy2DAsync(d_dstU + c, dst_pitch, d_f32 + src_offset1, src_pitch, sizeof(float), mHeight, cudaMemcpyDeviceToDevice));
		CubDebug(cudaMemcpy2DAsync(d_dstV + c, dst_pitch, d_f32 + src_offset2, src_pitch, sizeof(float), mHeight, cudaMemcpyDeviceToDevice));
	}
	cudaDeviceSynchronize();
	CubDebug(cudaGetLastError());
	CubDebugExit(gAlloc.DeviceFree(d_f32));
#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "YC48H->F32Planar@ %d x %d: %.2f ms", mWidth, mHeight, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;
}*/
AUEXTC __declspec(dllexport) int YC48H_F32DPlanar(float* d_dstY, float* d_dstU, float* d_dstV, PIXEL_YC* h_src, const int width_px, const int height_px, const int src_stride_px)
{
	if (!d_dstY || !d_dstU || !d_dstV || !h_src) return FALSE;
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	//Get System data
	cudaDeviceProp devInfo = { 0 };
	int devID = 0;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&devInfo, devID);
	//
	cub::CachingDeviceAllocator gAlloc(true); //Device memory allocator
	//allocate yc48 on gpu
	void* dYC48 = NULL;
	gAlloc.DeviceAllocate((void**)&dYC48, width_px*height_px*sizeof(PIXEL_YC));
	CubDebugExit(cudaMemcpy2D((void*)dYC48, width_px*sizeof(PIXEL_YC), (void*)h_src, src_stride_px* sizeof(PIXEL_YC), width_px*sizeof(PIXEL_YC), height_px, cudaMemcpyHostToDevice));
	//
	//Kernel
	int px_count = width_px*height_px;
	int GridSize = devInfo.maxThreadsPerBlock;
	int BlockSize = static_cast<int>(ceilf((float)px_count / (float)GridSize));
	kYC48_F32Planar <<<GridSize, BlockSize >>>(d_dstY, d_dstU, d_dstV, dYC48, px_count);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) return FALSE;

#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "YC48H->F32Planar@ %d x %d: %.2f ms", width_px, height_px, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;
}

/*AUEXTC __declspec(dllexport) int F32D_YC48H(PIXEL_YC* h_dst, const int dst_stride_px, float* d_srcY, float* d_srcU, float* d_srcV, const int width_px, const int height_px)
{
	if (!d_srcY || !d_srcU || !d_srcV || !h_dst) return FALSE;
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	//Get System data
	cudaDeviceProp devInfo = { 0 };
	int devID = 0;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&devInfo, devID);
	//
	cub::CachingDeviceAllocator gAlloc(true); //Device memory allocator
	//
	///Mem allocation
	float* f32packed = NULL;
	PIXEL_YC* d_yc48 = NULL;
	CubDebugExit(gAlloc.DeviceAllocate((void**)&f32packed, width_px*height_px * 3 * sizeof(float)));
	CubDebugExit(gAlloc.DeviceAllocate((void**)&d_yc48, width_px*height_px*sizeof(PIXEL_YC)));
	///Planar -> Packed
	if (!F32Planar_Packed(f32packed, d_srcY, d_srcU, d_srcV, width_px, height_px)) return FALSE;
	///Float -> short
	int BlockSize = devInfo.maxThreadsPerBlock;
	int GridSize = (BlockSize >= width_px*height_px) ? 1 : 32;
	size_t sMemSize = BlockSize * sizeof(float);
	float2short <<<GridSize, BlockSize, sMemSize >>>((void*)d_yc48, (void*)f32packed, width_px*height_px * 3);
	cudaDeviceSynchronize();
	CubDebugExit(cudaGetLastError());
	CubDebugExit(gAlloc.DeviceFree(f32packed));
	///Copyback to Host
	CubDebugExit(cudaMemcpy2D((void*)h_dst, dst_stride_px*sizeof(PIXEL_YC), (void*)d_yc48, width_px*sizeof(PIXEL_YC), width_px*sizeof(PIXEL_YC), height_px, cudaMemcpyDeviceToHost));
	CubDebugExit(gAlloc.DeviceFree(d_yc48));
#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "F32Planar->YC48H@ %d x %d: %.2f ms", width_px, height_px, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;
}*/

AUEXTC __declspec(dllexport) int F32D_YC48H(PIXEL_YC* h_dst, const int dst_stride_px, float* d_srcY, float* d_srcU, float* d_srcV, const int width_px, const int height_px)
{
	if (!d_srcY || !d_srcU || !d_srcV || !h_dst) return FALSE;
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	//Get System data
	cudaDeviceProp devInfo = { 0 };
	int devID = 0;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&devInfo, devID);
	//
	cub::CachingDeviceAllocator gAlloc(true); //Device memory allocator
	//
	///Mem allocation
	PIXEL_YC* d_yc48 = NULL;
	CubDebugExit(gAlloc.DeviceAllocate((void**)&d_yc48, width_px*height_px*sizeof(PIXEL_YC)));
	///Planar -> Packed
	int BlockSize = devInfo.maxThreadsPerBlock;
	int GridSize = static_cast<int>(ceilf((float)width_px*(float)height_px / (float)BlockSize));
	kF32Planar_YC48 <<<GridSize, BlockSize >>>(d_yc48, d_srcY, d_srcU, d_srcV, width_px*height_px);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	///Copyback to Host
	CubDebugExit(cudaMemcpy2D((void*)h_dst, dst_stride_px*sizeof(PIXEL_YC), (void*)d_yc48, width_px*sizeof(PIXEL_YC), width_px*sizeof(PIXEL_YC), height_px, cudaMemcpyDeviceToHost));
	CubDebugExit(gAlloc.DeviceFree(d_yc48));
#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "F32Planar->YC48H@ %d x %d: %.2f ms", width_px, height_px, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;
}

AUEXTC __declspec(dllexport) int Sum_F32(float* d_src, const int width_px, const int height_px, float* result)
{
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	void* d_temp = NULL;
	size_t temp_byte = 0;
	float *d_sum;
	cudaMalloc((void**)&d_sum, sizeof(float));
	int num_item = width_px*height_px;
	cub::DeviceReduce::Sum(d_temp, temp_byte, d_src, d_sum, num_item);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	cudaMalloc((void**)&d_temp, temp_byte);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	cub::DeviceReduce::Sum(d_temp, temp_byte, d_src, d_sum, num_item);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	float h_result;
	cudaMemcpy(&h_result, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	*result = h_result;
	cudaFree(d_sum);
	cudaFree(d_temp);
#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "Summing@ %d x %d: %.2f ms", width_px, height_px, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;
}
AUEXTC __declspec(dllexport) int Sum_F32ROI(float* d_src, const int width_px, const int height_px, const accutl::ROI &roi, float* result)
{
	//check and fix roi
	int tlx = max(0,min(width_px, roi.tlx));
	int tly = max(0, min(height_px, roi.tly));
	int roiw = max(0, min(tlx + roi.w, width_px)) - tlx;
	int roih = max(0, min(tly + roi.h, height_px)) - tly;
	if (roiw <= 0 || roih <= 0) return FALSE; //nothing to sum
	//Starting offset
	float* topleft = d_src + width_px* tly + tlx;
	//Copy out a sample
	float* temp = NULL;
	cudaMalloc((void**)&temp, roiw*roih*sizeof(float));
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	cudaMemcpy2D((void*)temp, roiw*sizeof(float), (void*)topleft, width_px*sizeof(float), roiw*sizeof(float), roih, cudaMemcpyDeviceToDevice);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	//Calculate
	if (!Sum_F32(temp, roiw, roih, result)) return FALSE;
	//Free
	cudaFree(temp);
	return TRUE;
}


AUEXTC __declspec(dllexport) int Min_F32(float* d_src, const int width_px, const int height_px, float* result)
{
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	void* d_temp = NULL;
	size_t temp_byte = 0;
	float *d_sum;
	cudaMalloc((void**)&d_sum, sizeof(float));
	int num_item = width_px*height_px;
	cub::DeviceReduce::Min(d_temp, temp_byte, d_src, d_sum, num_item);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	cudaMalloc((void**)&d_temp, temp_byte);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	cub::DeviceReduce::Min(d_temp, temp_byte, d_src, d_sum, num_item);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	float h_result;
	cudaMemcpy(&h_result, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	*result = h_result;
	cudaFree(d_sum);
	cudaFree(d_temp);
#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "Min@ %d x %d: %.2f ms", width_px, height_px, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;
}
AUEXTC __declspec(dllexport) int Min_F32ROI(float* d_src, const int width_px, const int height_px, const accutl::ROI &roi, float* result)
{
	//check and fix roi
	int tlx = max(0, min(width_px, roi.tlx));
	int tly = max(0, min(height_px, roi.tly));
	int roiw = max(0, min(tlx + roi.w, width_px)) - tlx;
	int roih = max(0, min(tly + roi.h, height_px)) - tly;
	if (roiw <= 0 || roih <= 0) return FALSE; //nothing to compare
	//Starting offset
	float* topleft = d_src + width_px* tly + tlx;
	//Copy out a sample
	float* temp = NULL;
	cudaMalloc((void**)&temp, roiw*roih*sizeof(float));
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	cudaMemcpy2D((void*)temp, roiw*sizeof(float), (void*)topleft, width_px*sizeof(float), roiw*sizeof(float), roih, cudaMemcpyDeviceToDevice);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	//Calculate
	if (!Min_F32(temp, roiw, roih, result)) return FALSE;
	//Free
	cudaFree(temp);
	return TRUE;
}


AUEXTC __declspec(dllexport) int Max_F32(float* d_src, const int width_px, const int height_px, float* result)
{
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	void* d_temp = NULL;
	size_t temp_byte = 0;
	float *d_sum;
	cudaMalloc((void**)&d_sum, sizeof(float));
	int num_item = width_px*height_px;
	cub::DeviceReduce::Max(d_temp, temp_byte, d_src, d_sum, num_item);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	cudaMalloc((void**)&d_temp, temp_byte);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	cub::DeviceReduce::Max(d_temp, temp_byte, d_src, d_sum, num_item);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	float h_result;
	cudaMemcpy(&h_result, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	*result = h_result;
	cudaFree(d_sum);
	cudaFree(d_temp);
#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "Summing@ %d x %d: %.2f ms", width_px, height_px, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;
}
AUEXTC __declspec(dllexport) int Max_F32ROI(float* d_src, const int width_px, const int height_px, const accutl::ROI &roi, float* result)
{
	//check and fix roi
	int tlx = max(0, min(width_px, roi.tlx));
	int tly = max(0, min(height_px, roi.tly));
	int roiw = max(0, min(tlx + roi.w, width_px)) - tlx;
	int roih = max(0, min(tly + roi.h, height_px)) - tly;
	if (roiw <= 0 || roih <= 0) return FALSE; //nothing to sum
	//Starting offset
	float* topleft = d_src + width_px* tly + tlx;
	//Copy out a sample
	float* temp = NULL;
	cudaMalloc((void**)&temp, roiw*roih*sizeof(float));
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	cudaMemcpy2D((void*)temp, roiw*sizeof(float), (void*)topleft, width_px*sizeof(float), roiw*sizeof(float), roih, cudaMemcpyDeviceToDevice);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	//Calculate
	if (!Max_F32(temp, roiw, roih, result)) return FALSE;
	//Free
	cudaFree(temp);
	return TRUE;
}


AUEXTC __declspec(dllexport) int Histogram_F32C1(float* d_src, const int width_px, const int height_px, const float low_limit, const float high_limit, const int bin_count, accutl::HIST &result)
{
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	void* d_temp_store = NULL;
	size_t temp_byte = 0;
	int* d_histogram = NULL;
	CubDebugExit(cudaMalloc((void**)&d_histogram, bin_count*sizeof(int)));
	CubDebugExit(cudaMemset(d_histogram, 0, bin_count*sizeof(int)));
	cub::DeviceHistogram::HistogramEven(d_temp_store, temp_byte, d_src, d_histogram, bin_count + 1, low_limit, high_limit, width_px*height_px); //get temp store param
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	//Allocate temp
	CubDebugExit(cudaMalloc((void**)&d_temp_store, temp_byte));
	//run
	cub::DeviceHistogram::HistogramEven(d_temp_store, temp_byte, d_src, d_histogram, bin_count + 1, low_limit, high_limit, width_px*height_px);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	//free temp store
	cudaFree(d_temp_store);
	//Transfer histogram to host
	int* h_histogram = new int[bin_count];
	CubDebugExit(cudaMemcpy(h_histogram, d_histogram, bin_count*sizeof(int), cudaMemcpyDeviceToHost));
	//push result to output vector
	result.clear();
	for (int i = 0; i < bin_count; i++)
	{
		result.push_back(h_histogram[i]);
	}
	//Free host histogram
	delete[] h_histogram;
#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "Histogram_F32C1@ %d x %d: %.2f ms", width_px, height_px, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;

}

AUEXTC __declspec(dllexport) int Histogram_F32C3(float* d_src, const int width_px, const int height_px,
	float low_limit[3], float high_limit[3], int bin_count[3],
	accutl::HIST &channel0, accutl::HIST &channel1, accutl::HIST &channel2)
{
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	void* d_temp_store = NULL;
	size_t temp_byte = 0;

	int* d_histogram[3] = { 0 };
	CubDebugExit(cudaMalloc((void**)&d_histogram[0], bin_count[0]*sizeof(int)));
	CubDebugExit(cudaMalloc((void**)&d_histogram[1], bin_count[1]*sizeof(int)));
	CubDebugExit(cudaMalloc((void**)&d_histogram[2], bin_count[2]*sizeof(int)));

	int numlevel[3] = { bin_count[0] + 1, bin_count[1] + 1, bin_count[2] + 2 };
	cub::DeviceHistogram::MultiHistogramEven<3, 3>(d_temp_store, temp_byte, d_src, d_histogram, numlevel, low_limit, high_limit, width_px*height_px);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	//Alloca temp
	CubDebugExit(cudaMalloc((void**)&d_temp_store, temp_byte));
	//Actual run
	cub::DeviceHistogram::MultiHistogramEven<3, 3>(d_temp_store, temp_byte, d_src, d_histogram, numlevel , low_limit, high_limit, width_px*height_px);
	//Free temp
	cudaFree(d_temp_store);
	//Copy result to host
	int* c0 = new int[bin_count[0]];
	int* c1 = new int[bin_count[1]];
	int* c2 = new int[bin_count[2]];
	cudaMemcpy(c0, d_histogram[0], bin_count[0]*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(c1, d_histogram[1], bin_count[1]*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(c2, d_histogram[2], bin_count[2]*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	//push to output vectors
	channel0.clear();
	channel1.clear();
	channel2.clear();
	for (int i = 0; i < bin_count[0]; i++)
	{
		channel0.push_back(c0[i]);
		
	}
	for (int i = 0; i < bin_count[1]; i++)
	{
		channel1.push_back(c1[i]);

	}
	for (int i = 0; i < bin_count[2]; i++)
	{
		channel2.push_back(c2[i]);

	}
	//Free host mem
	delete[] c0;
	delete[] c1;
	delete[] c2;
#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "Histogram_F32C3@ %d x %d: %.2f ms", width_px, height_px, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;
	
}


//:::::::::::::::::::::: Transformation functions ::::::::::::::::::::::::::::://
AUEXTC __declspec(dllexport) int Transpose_F32(float* d_src, const int src_w, const int src_h, float* d_dst)
{
	if (!d_src || !d_dst) return FALSE;
	if (src_w*src_h <= 1) return FALSE;
#if _DEBUG
	cudaEvent_t start, stop;
	float elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif
	if ((src_w%TILE_DIM == 0) && (src_h%TILE_DIM == 0)) //if dimensions are multiples of 16, it will be fine
	{
		//dim3 GridSize(static_cast<int>(ceilf((float)src_w / (float)TILE_DIM)), static_cast<int>(ceilf((float)src_h / (float)TILE_DIM)), 1);
		dim3 GridSize(src_w / TILE_DIM, src_h / TILE_DIM);
		dim3 BlockSize(TILE_DIM, BLOCK_ROWS);
		transposeDiagonal <<<GridSize, BlockSize >>>(d_dst, d_src, src_w, src_h);
		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaSuccess) return FALSE;
	}
	else //Otherwise, we have to pad it
	{
		int padded_w = static_cast<int>(ceilf((float)src_w / (float)TILE_DIM)*(float)TILE_DIM);
		int padded_h = static_cast<int>(ceilf((float)src_h / (float)TILE_DIM)*(float)TILE_DIM);

		float* padded_src = NULL, *padded_dst=NULL;
		cudaMalloc((void**)&padded_src, padded_w*padded_h*sizeof(float));
		cudaMalloc((void**)&padded_dst, padded_w*padded_h*sizeof(float));

		cudaMemset(padded_src, 0, padded_w*padded_h*sizeof(float));
		cudaMemset(padded_dst, 0, padded_w*padded_h*sizeof(float));
		cudaMemcpy2D((void*)padded_src, padded_w*sizeof(float), (void*)d_src, src_w*sizeof(float), src_w*sizeof(float), src_h, cudaMemcpyDeviceToDevice);
		dim3 GridSize(padded_w / TILE_DIM, padded_h / TILE_DIM);
		dim3 BlockSize(TILE_DIM, BLOCK_ROWS);
		transposeDiagonal <<<GridSize, BlockSize >>>(padded_dst, padded_src, padded_w, padded_h);
		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaSuccess) return FALSE;
		cudaMemcpy2D(d_dst, src_h*sizeof(float), padded_dst, padded_h*sizeof(float), src_h*sizeof(float), src_w, cudaMemcpyDeviceToDevice);
		cudaFree(padded_dst);
		cudaFree(padded_src);
	}
#if _DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	char msg[128] = { 0 };
	sprintf(msg, "Transpose@ %d x %d: %.2f ms", src_w, src_h, elapsed);
	MessageBox(NULL, msg, "DEBUG Msg", MB_OK | MB_ICONINFORMATION);
#endif
	return TRUE;
}

AUEXTC __declspec(dllexport) int Saxpy_F32(float* d_src, const int pixel_count, const float offset, const float multiplier, float* d_dst)
{
	//__device__ float ofs = offset;
	//__device__ float mul = multiplier;
	//__device__ int px_cnt = pixel_count;
	//Get System data
	cudaDeviceProp devInfo = { 0 };
	int devID = 0;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&devInfo, devID);

	int BlockSize = devInfo.maxThreadsPerBlock;
	int GridSize = (BlockSize >= pixel_count) ? 1 : 32;
	size_t sMemSize = BlockSize * sizeof(float);
	kSaxpy <<<GridSize, BlockSize, sMemSize >>>(d_src, d_dst, pixel_count, offset, multiplier);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	return TRUE;
}


AUEXTC __declspec(dllexport) int Saxpy_F32i(float* d_src, const int pixel_count, const float offset, const float multiplier)
{
	if (!Saxpy_F32(d_src, pixel_count, offset, multiplier, d_src)) return FALSE;
	return TRUE;
}

AUEXTC __declspec(dllexport) int ColorTwist_F32C3(float* d_src, const int pixel_count, float matrix[9], float* d_dst)
{
	//Get System data
	cudaDeviceProp devInfo = { 0 };
	int devID = 0;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&devInfo, devID);
	//Kernel
	int BlockSize = devInfo.maxThreadsPerBlock;
	int GridSize = static_cast<int>(ceilf((float)pixel_count / (float)BlockSize));
	kColorTwist <<<GridSize, BlockSize >>>(d_src, d_dst, pixel_count, matrix);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) return FALSE;
	return TRUE;
}
AUEXTC __declspec(dllexport) int ColorTwist_F32C3i(float* d_src, const int pixel_count, float matrix[9], float* d_dst)
{
	if (!ColorTwist_F32C3(d_src, pixel_count, matrix, d_src)) return FALSE;
	return TRUE;
}



AUEXTC __declspec(dllexport) int Resize_F32(float *d_src, float *output, int width, int height, float scale, accutl::Mode filter_mode, float data_range)
{
	//Not implemented yet
	return FALSE;
}