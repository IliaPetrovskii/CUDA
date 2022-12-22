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
	int bx = blockIdx.x; // номер блока по x
	int by = blockIdx.y; // номер блока по y
	int tx = threadIdx.x; // номер нити в блоке по x
	int ty = threadIdx.y; // номер нити в блоке по y
	float sum = 0.0f;
	int ia = n * (block_size * by + ty); // номер строки из AТ
	int ib = block_size * bx + tx; // номер столбца из BТ
	int ic = ia + ib; // номер элемента из —Т
	// вычисление элемента матрицы C
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
	// создание переменных-событий
	float timerValueGPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int numBytes = N * N * sizeof(float);
	float* adev, * bdev, * cdev, * a, * b, * c;
	// выделение пам€ти на host
	a = (float*)malloc(numBytes); //матрица A
	b = (float*)malloc(numBytes); //матрица B
	c = (float*)malloc(numBytes); //матрица — дл€ GPU-варианта
	// задание матрицы A, B и транспонированной матрицы B
	fillMatrix(a, b);
	// задание сетки нитей и блоков
	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 blocks(N / threads.x, N / threads.y);
	// выделение пам€ти на GPU
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);
	// ---------------- GPU-вариант ------------------------
	// копирование матриц A и B с host на device
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);
	// запуск таймера
	cudaEventRecord(start, 0);
	// запуск функции-€дра
	kernel_global << < blocks, threads >> > (adev, bdev, N, cdev, block_size);
	// оценка времени вычислени€ GPU-варианта
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("%f\n", timerValueGPU / 1000.0);
	// копирование, вычисленной матрицы C с device на host
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

	// освобождение пам€ти
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);
	free(a);
	free(b);
	free(c);
	// уничтожение переменных-событий
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}
