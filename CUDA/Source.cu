#include <iostream>
#include <math.h>
#include <functional>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 1024

__global__ void kernel_global(float* a, float* b, int n, float* c, int block_size)
{
	int bx = blockIdx.x; // ����� ����� �� x
	int by = blockIdx.y; // ����� ����� �� y
	int tx = threadIdx.x; // ����� ���� � ����� �� x
	int ty = threadIdx.y; // ����� ���� � ����� �� y
	float sum = 0.0f;
	int ia = n * (block_size * by + ty); // ����� ������ �� A�
	int ib = block_size * bx + tx; // ����� ������� �� B�
	int ic = ia + ib; // ����� �������� �� ђ
	// ���������� �������� ������� C
	for (int k = 0; k < n; k++) 
		sum += a[ia + k] * b[ib + k * n];
	c[ic] = sum;
}

template<typename T>
void fillMatrix(T* A, T* B)
{
	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i * N + j] = rand() / 10.0;
		}
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			B[i * N + j] = rand() / 10.0;
		}
	}
}


int main()
{
	int m, n, k;
	// �������� ����������-�������
	float timerValueGPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int numBytes = N * N * sizeof(float);
	float* adev, * bdev, * cdev, * a, * b, * c;
	// ��������� ������ �� host
	a = (float*)malloc(numBytes); //������� A
	b = (float*)malloc(numBytes); //������� B
	c = (float*)malloc(numBytes); //������� � ��� GPU-��������
	// ������� ������� A, B � ����������������� ������� B
	fillMatrix(a, b);
	// ������� ����� ����� � ������
	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 blocks(N / threads.x, N / threads.y);
	// ��������� ������ �� GPU
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);
	// ---------------- GPU-������� ------------------------
	// ����������� ������ A � B � host �� device
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);
	// ������ �������
	cudaEventRecord(start, 0);
	// ������ �������-����
	kernel_global << < blocks, threads >> > (adev, bdev, N, cdev, block_size);
	// ������ ������� ���������� GPU-��������
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("%f\n", timerValueGPU / 1000.0);
	// �����������, ����������� ������� C � device �� host
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

	// ������������ ������
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);
	free(a);
	free(b);
	free(c);
	// ����������� ����������-�������
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}
