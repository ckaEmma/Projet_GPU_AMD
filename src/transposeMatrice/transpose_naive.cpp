#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t errorCode = (call);                                      \
        if (errorCode != hipSuccess) {                                      \
            fprintf(stderr, "Erreur HIP : %s (fichier %s, ligne %d)\n",     \
                    hipGetErrorString(errorCode), __FILE__, __LINE__);      \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


__global__ void naiveTransposeKernel(const float* inputMat, float* outputMat, int numRows, int numCols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < numRows && col < numCols) {
        int inputIndex  = row * numCols + col; //parcours row major
        int outputIndex = col * numRows + row; //parcours column major
        outputMat[outputIndex] = inputMat[inputIndex];
    }
}


void randomMatrix(std::vector<float>& mat, int numRows, int numCols){
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            mat[row * numCols + col] = (float)(rand() % 10);
        }
    }
}

void transposeOnCpu(const std::vector<float>& inputMat, std::vector<float>& outputMat, int numRows, int numCols){
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            outputMat[col*numRows+row] = inputMat[row*numCols + col];
        }
    }
}

int countDifferences(const std::vector<float>& firstMat, const std::vector<float>& secondMat){
    int numDifferences = 0;
    for (size_t index = 0; index < firstMat.size(); index++) {
        if (firstMat[index] != secondMat[index]) {
            numDifferences++;
        }
    }
    return numDifferences;
}

void verifResult(const std::vector<float>& gpuMat,const std::vector<float>& cpuMat){
    int numDifferences = countDifferences(gpuMat, cpuMat);
    if (numDifferences==0) {
        printf("La transposee GPU est identique a celle du CPU\n");
    } else {
        printf("La transposee GPU differe de celle du CPU : %d elements differents\n", numDifferences);
    }
}

void printMatrix(const char* title, const std::vector<float>& mat, int numRows, int numCols){
    printf("%s (%d x %d)\n", title, numRows, numCols);
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            printf("%2d ", (int)mat[row*numCols + col]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(){
    int M = 18;   //changer comme vous voulez nb lignes (1000,3500 etc)
    int N = 3;   // idem ici colonnes
    size_t numElements = (size_t)M * (size_t)N;

    //faut mettre ça pour les hipMalloc
    const size_t sizeInBytes = numElements * sizeof(float);

    printf("Transposition d'une matrice %d x %d \n\n", M, N);

    //matrices côté CPU
    srand(42);
    std::vector<float> hostInputMat(numElements);
    std::vector<float> hostOutputMatCpu(numElements); //contient la matrice tranposée par le cpu elle-même
    std::vector<float> hostOutputMatGpu(numElements); //il va contenir la matrice de la transpose par le gpu

    randomMatrix(hostInputMat, M, N);

    //matrices cote GPU
    float* deviceInputMat=nullptr;
    float* deviceOutputMat=nullptr;

    HIP_CHECK(hipMalloc((void**)&deviceInputMat, sizeInBytes));
    HIP_CHECK(hipMalloc((void**)&deviceOutputMat, sizeInBytes));

    HIP_CHECK(hipMemcpy(deviceInputMat, hostInputMat.data(), sizeInBytes,
                        hipMemcpyHostToDevice));

    transposeOnCpu(hostInputMat, hostOutputMatCpu, M, N);
    dim3 threadsPerBlock(16, 16); //changer par 32 et 8, comme vous voulez
    dim3 numBlocks((N+threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naiveTransposeKernel<<<numBlocks, threadsPerBlock>>>(deviceInputMat, deviceOutputMat, M, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());


    //le résultat trouvé par gpu copié dans cpu
    HIP_CHECK(hipMemcpy(hostOutputMatGpu.data(), deviceOutputMat, sizeInBytes,
                        hipMemcpyDeviceToHost));

    verifResult(hostOutputMatGpu, hostOutputMatCpu);

    //affichage pour les matrices petites, donc changer M et N en haut pour voir
    if (M <= 20 && N <= 20) {
        printf("\n");
        printMatrix("Matrice de depart", hostInputMat, M, N);
        printMatrix("Transposee GPU", hostOutputMatGpu, N, M);
    }

    HIP_CHECK(hipFree(deviceInputMat));
    HIP_CHECK(hipFree(deviceOutputMat));

    return 0;
}
