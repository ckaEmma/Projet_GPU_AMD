
#include <iostream>
#include <cstdlib>
#include <hip/hip_runtime.h>

__global__ void vector_add_naive(float* A, float* B, float* C, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}



int main(int argc, char** argv)
{
    size_t N = 1 << 20;
    if (argc > 1)
    {
        N = atoi(argv[1]);
    }

    size_t taille = N * sizeof(float);

    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];

    for (size_t i = 0; i < N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }
    float *d_A, *d_B, *d_C;

    hipMalloc(&d_A, taille);
    hipMalloc(&d_B, taille);
    hipMalloc(&d_C, taille);

    //data CPU vers le GPU
    hipMemcpy(d_A, A, taille, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, taille, hipMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;


    hipLaunchKernelGGL(vector_add_naive, dim3(blocks), dim3(threads), 0, 0, d_A, d_B, d_C, N);

    hipError_t erreur = hipGetLastError();

    if (erreur != hipSuccess){
        std::cout << "Erreur lors du lancement du kernel" << std::endl;
        return 1;
    }

    erreur = hipDeviceSynchronize();

    if (erreur != hipSuccess){
        std::cout << "Erreur lors de l'execution du kernel" << std::endl;
        return 1;
    }

    hipMemcpy(C, d_C, taille, hipMemcpyDeviceToHost);

    //verifier résultat
    bool correct = true;

    for(size_t i = 0; i < N; i++){
        if (C[i] != 3.0){
            correct = false;
            break;
        }
    }

    if (correct){
        std::cout << "Addition naive : ok" << std::endl;
    }
    else
    {
        std::cout << "Addition naive : pas correct" << std::endl;
    }

    //pas oublier de free les pointeurs gpu
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    //libération de la mémoire CPU
    delete[] A;
    delete[] B;
    delete[] C;

    if (correct){
        return 0;
    }
    else
    {
        return 1;
    }
}
