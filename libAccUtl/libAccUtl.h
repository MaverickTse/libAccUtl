#pragma once
#include "aviutl_filter.h"
#include <vector>


extern "C"{
	namespace accutl{
		//:::::::::::::: Type defines ::::::::::::::::::::://

		///Region of interest
		using ROI = struct{
			int tlx; //Top-left x-coordinate
			int tly; //Top-left y-coordinate
			int w; //roi width
			int h; //roi height
		};

		using HIST = std::vector<int>; ///Histogram as a vector of int

		using Mode = enum { MODE_NEAREST, MODE_BILINEAR, MODE_BICUBIC, MODE_FAST_BICUBIC, MODE_CATROM }; ///Resize modes

		//:::::::::::::: Memory Functions ::::::::::::::::::://

		///Wrapper for cudaMalloc()
		int GPUMemAlloc(void** device_ptr, size_t size_byte);

		///Wrapper for cudaFree()
		int GPUMemFree(void* device_ptr);

		///Wrapper for cudaMemcpy(), device-to-device
		int GPUMemcpy_DD(void* d_dst, void* d_src, size_t size_byte);

		///Wrapper for cudaMemcpy(), device-to-host
		int GPUMemcpy_DH(void* h_dst, void* d_src, size_t size_byte);

		///Wrapper for cudaMemcpy(), host-to-device
		int GPUMemcpy_HD(void* d_dst, void* h_src, size_t size_byte);

		///Wrapper for cudaDeviceReset()
		int GPUEndSession();


		//:::::::::::::: Type Conversion ::::::::::::::::::://

		int F32Packed_Planar(float* d_packed, float* d_Y, float* d_U, float* d_V, const int width_px, const int height_px);


		int F32Planar_Packed(float* d_packed, float* d_Y, float* d_U, float* d_V, const int width_px, const int height_px);


		int YC48H_F32DPlanar(float* d_dstY, float* d_dstU, float* d_dstV, PIXEL_YC* h_src, const int width_px, const int height_px, const int src_stride_px);

		int YC48H_F32DPacked(float* d_dst, PIXEL_YC* h_src, const int width_px, const int height_px, const int src_stride_px);

		int F32D_YC48H(PIXEL_YC* h_dst, const int dst_stride_px, float* d_srcY, float* d_srcU, float* d_srcV, const int width_px, const int height_px);

		//:::::::::::::: Statistical Functions ::::::::::::::://
		int Sum_F32(float* d_src, const int width_px, const int height_px, float* result);
		int Sum_F32ROI(float* d_src, const int width_px, const int height_px, const accutl::ROI &roi, float* result);

		int Min_F32(float* d_src, const int width_px, const int height_px, float* result);
		int Min_F32ROI(float* d_src, const int width_px, const int height_px, const accutl::ROI &roi, float* result);

		int Max_F32(float* d_src, const int width_px, const int height_px, float* result);
		int Max_F32ROI(float* d_src, const int width_px, const int height_px, const accutl::ROI &roi, float* result);

		int Histogram_F32C1(float* d_src, const int width_px, const int height_px, const float low_limit, const float high_limit, const int bin_count, accutl::HIST &result);
		int Histogram_F32C3(float* d_src, const int width_px, const int height_px,
			float low_limit[3], float high_limit[3], int bin_count[3],
			accutl::HIST &channel0, accutl::HIST &channel1, accutl::HIST &channel2);

		//::::::::::::::: Transformation Functions :::::::::::::::://
		int Transpose_F32(float* d_src, const int src_w, const int src_h, float* d_dst);

		int Saxpy_F32(float* d_src, const int pixel_count, const float offset, const float multiplier, float* d_dst);
		int Saxpy_F32i(float* d_src, const int pixel_count, const float offset, const float multiplier);

		int ColorTwist_F32C3(float* d_src, const int pixel_count, float matrix[9], float* d_dst);
		int ColorTwist_F32C3i(float* d_src, const int pixel_count, float matrix[9], float* d_dst);

		int Resize_F32(float *d_src, float *output, int width, int height, float scale, accutl::Mode filter_mode, float data_range);

		//AUEXTC __declspec(dllexport) int Convolute_F32(float* d_src, const int src_w, const int src_h, std::vector<float> &matrix, float* d_dst);
	}
}