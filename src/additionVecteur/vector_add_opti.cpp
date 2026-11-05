
#include <iostream>
#include <cstdlib>
#include <hip/hip_runtime.h>

__global__ void vector_add_optimized(float* A, float* B, float* C, size_t N)
{
    //thread traite 4 éléments et float4 -> vecteur de 4 float
    //range les 4 floats dans un registre 4x4=16 octets pour faire du SIMD
    size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if(i + 3 < N){
        float4* A4 = (float4*)A;
        float4* B4 = (float4*)B;
        float4* C4 = (float4*)C;

        float4 a = A4[i / 4];
        float4 b = B4[i / 4];
        float4 c;

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        C4[i / 4] = c;
    }
    else{
        //cas des éléments restants
        for(size_t j = i; j < N; j++){
            C[j] = A[j] + B[j];
        }
    }
}

int main(int argc, char** argv){
    size_t N = 1 << 20;
    if (argc > 1){
        N = atoi(argv[1]);
    }
    size_t taille = N * sizeof(float);

    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];

    for (size_t i = 0; i < N; i++){
        A[i] = 1.0;
        B[i] = 2.0;
    }
    float *d_A, *d_B, *d_C;

    hipMalloc(&d_A, taille);
    hipMalloc(&d_B, taille);
    hipMalloc(&d_C, taille);

    hipMemcpy(d_A, A, taille, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, taille, hipMemcpyHostToDevice);

    int threads = 256;
    //chaque thread traite 4 éléments
    int blocks = (N+(4*threads)-1)/(4*threads);

    hipLaunchKernelGGL(vector_add_optimized, dim3(blocks), dim3(threads), 0, 0, d_A, d_B, d_C, N);

    hipError_t erreur = hipGetLastError();

    if(erreur != hipSuccess){
        std::cout << "Erreur" << std::endl;
        return 1;
    }

    erreur = hipDeviceSynchronize();

    if(erreur != hipSuccess)
    {
        std::cout << "Erreur execution du kernel" << std::endl;
        return 1;
    }

    hipMemcpy(C, d_C, taille, hipMemcpyDeviceToHost);

    bool correct = true;
    for(size_t i = 0; i < N; i++){
        if (C[i] != 3.0){
            correct = false;
            break;
        }
    }
    if (correct){
        std::cout << "Addition optimisee: ok" << std::endl;
    }
    else
    {
        std::cout << "Addition optimisee: incorrect" << std::endl;
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

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
