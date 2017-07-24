#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

__global__ void array_set_kernel(int n, float *arr, float value) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    arr[idx] = value;
}

int DLGpuArraySet(DLArrayHandle arr, float value) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < arr->ndim; ++i)
        n *= arr->shape[i];
    int threads = 0;
    if (n <= 1024) threads = n;
    else threads = 1024;
    int blocks = (n + 1023) / 1024;
    
    float *arr_data = (float *)arr->data;
    
    array_set_kernel<<<blocks, threads>>>(n, arr_data, value);
    
    return 0;
}

__global__ void broadcast_to(int n, int copies, const float *input, float *output) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * copies) return;
    output[idx] = input[idx % n];
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    assert(output->ndim == input->ndim + 1);
    for (int i = 0; i < input->ndim; ++i)
        assert(input->shape[i] == output->shape[i + 1]);
    
    int n = 1;
    for (int i = 0; i < input->ndim; ++i) n *= input->shape[i];
    int copies = output->shape[0];
    int threads = min(1024, n * copies);
    int blocks = (n * copies + 1023) / 1024;
    
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    
    broadcast_to<<<blocks, threads>>>(n, copies, input_data, output_data);
    return 0;
}

__global__ void reduce_sum_axis_zero(int n, int m, const float *input, float *output) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = 0;
    for (int i = 0; i < m; ++i) output[idx] += input[idx + n * i];
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    assert(output->ndim + 1 == input->ndim);
    for (int i = 1; i < input->ndim; ++i)
        assert(output->shape[i - 1] == input->shape[i]);
    
    int n = 1;
    for (int i = 1; i < input->ndim; ++i) n *= input->shape[i];
    int m = input->shape[0];
    int threads = min(1024, n);
    int blocks = (n + 1023) / 1024;
    
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    
    reduce_sum_axis_zero<<<blocks, threads>>>(n, m, input_data, output_data);
    return 0;
}

__global__ void matrix_elementwise_add(int n, const float *matA,
                                              const float *matB, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = matA[idx] + matB[idx];
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
    /* TODO: Your code here */
    assert(matA->ndim == matB->ndim);
    for(int i = 0; i < matA->ndim; ++i)
        assert(matA->shape[i] == matB->shape[i]);
    assert(matA->ndim == output->ndim);
    for(int i = 0; i < matA->ndim; ++i)
        assert(matA->shape[i] == output->shape[i]);
    
    int n = 1;
    for(int i = 0; i < matA->ndim; ++i)
        n *= matA->shape[i];
    int threads = 0;
    if (n <= 1024) threads = n;
    else threads = 1024;
    int blocks = (n + 1023) / 1024;
    
    const float *matA_data = (const float *)matA->data;
    const float *matB_data = (const float *)matB->data;
    float *output_data = (float *)output->data;
    
    matrix_elementwise_add<<<blocks, threads>>>(n, matA_data, matB_data, output_data);
    return 0;
}

__global__ void matrix_elementwise_add_by_const(int n, const float *matA,
                                                float *output, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = matA[idx] + val;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle matA, float val,
                                     DLArrayHandle output) {
    /* TODO: Your code here */
    assert(matA->ndim == output->ndim);
    for(int i = 0; i < matA->ndim; ++i)
        assert(matA->shape[i] == output->shape[i]);
    
    int n = 1;
    for(int i = 0; i < matA->ndim; ++i)
        n *= matA->shape[i];
    int threads = 0;
    if (n <= 1024) threads = n;
    else threads = 1024;
    int blocks = (n + 1023) / 1024;
    
    const float *matA_data = (const float *)matA->data;
    float *output_data = (float *)output->data;
    
    matrix_elementwise_add_by_const<<<blocks, threads>>>(n, matA_data, 
                                                         output_data, val);
    return 0;
}

__global__ void matrix_elementwise_multiply(int n, const float *matA,
                                                   const float *matB, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = matA[idx] * matB[idx];
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
    /* TODO: Your code here */
    assert(matA->ndim == matB->ndim);
    for(int i = 0; i < matA->ndim; ++i)
        assert(matA->shape[i] == matB->shape[i]);
    assert(matA->ndim == output->ndim);
    for(int i = 0; i < matA->ndim; ++i)
        assert(matA->shape[i] == output->shape[i]);
    
    int n = 1;
    for(int i = 0; i < matA->ndim; ++i)
        n *= matA->shape[i];
    int threads = 0;
    if (n <= 1024) threads = n;
    else threads = 1024;
    int blocks = (n + 1023) / 1024;
    
    const float *matA_data = (const float *)matA->data;
    const float *matB_data = (const float *)matB->data;
    float *output_data = (float *)output->data;
    
    matrix_elementwise_multiply<<<blocks, threads>>>(n, matA_data, matB_data, output_data);
    return 0;
}

__global__ void matrix_elementwise_multiply_by_const(int n, const float *matA,
                                                     float *output, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = matA[idx] * val;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle matA, float val,
                               DLArrayHandle output) {
    /* TODO: Your code here */
    assert(matA->ndim == output->ndim);
    for(int i = 0; i < matA->ndim; ++i)
        assert(matA->shape[i] == output->shape[i]);
    
    int n = 1;
    for(int i = 0; i < matA->ndim; ++i)
        n *= matA->shape[i];
    int threads = 0;
    if (n <= 1024) threads = n;
    else threads = 1024;
    int blocks = (n + 1023) / 1024;
    
    const float *matA_data = (const float *)matA->data;
    float *output_data = (float *)output->data;
    
    matrix_elementwise_multiply_by_const<<<blocks, threads>>>(n, matA_data, 
                                                              output_data, val);
    return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
    /* TODO: Your code here */
    // Hint: use cublas
    // cublas assume matrix is column major
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS);
    cudaThreadSynchronize();
    
    assert(matA->ndim == 2);
    assert(matB->ndim == 2);
    assert(matC->ndim == 2);
    
    int m = matC->shape[0];
    int n = matC->shape[1];
    int k = matA->shape[transposeA ? 0 : 1];
    assert(matA->shape[transposeA ? 1 : 0] == m);
    assert(matB->shape[transposeB ? 0 : 1] == n);
    assert(matB->shape[transposeB ? 1 : 0] == k);
    //printf("\nm %d n %d k %d\n", m, n, k);
    // C = (A1)(B1) = (m * k) * (k * n) = m * n
    
    const float *matA_data = (const float *)matA->data;
    const float *matB_data = (const float *)matB->data;
    float *matC_data = (float *)matC->data;
    const float alpha = 1.0, beta = 0.0;
    
    cublasOperation_t transa = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSgemm(handle,
                transb, transa, n, m, k,
                &alpha,
                matB_data, transb == CUBLAS_OP_T ? k : n,
                matA_data, transa == CUBLAS_OP_T ? m : k,
                &beta,
                matC_data, n);
    cudaThreadSynchronize();
    return 0;
}

__global__ void relu(int n, const float *input, float *output) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = max(0.00, input[idx]);
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    assert(input->ndim == output->ndim);
    for (int i = 0; i < input->ndim; ++i)
        assert(input->shape[i] == output->shape[i]);
    
    int n = 1;
    for (int i = 0; i < input->ndim; ++i) n *= input->shape[i];
    int threads = min(1024, n);
    int blocks = (n + 1023) / 1024;
    
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    
    relu<<<blocks, threads>>>(n, input_data, output_data);
    return 0;
}

__global__ void relu_gradient(int n, const float *input, 
                                     const float *in_grad, float *output) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = input[idx] > 0 ? 1 : 0;
    output[idx] *= in_grad[idx];
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
    /* TODO: Your code here */
    assert(input->ndim == 2);
    assert(in_grad->ndim == 2);
    assert(output->ndim == 2);
    for(int i = 0; i < 2; ++i)
        assert(input->shape[i] == in_grad->shape[i]),
        assert(input->shape[i] == output->shape[i]);
    
    int nrow = input->shape[0];
    int ncol = input->shape[1];
    int n = nrow * ncol;
    int threads = min(1024, n);
    int blocks = (n + 1023) / 1024;
    
    const float *input_data = (const float *)input->data;
    const float *in_grad_data = (const float *)in_grad->data;
    float *output_data = (float *)output->data;
    
    relu_gradient<<<blocks, threads>>>(n, input_data, in_grad_data, output_data);
    return 0;
}

__global__ void softmax_kernel(int nrow, int ncol, 
                               const float *input, float *output) {
    
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= nrow) return;
    input += y * ncol;
    output += y * ncol;
    float maxval = *input;
    for (int x = 1; x < ncol; ++x) maxval = max(maxval, input[x]);
    float sum = 0;
    for (int x = 0; x < ncol; ++x) sum += exp(input[x] - maxval);
    for (int x = 0; x < ncol; ++x) output[x] = exp(input[x] - maxval) / sum;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    assert(input->shape[0] == output->shape[0] &&
           input->shape[1] == output->shape[1]);
    
    int nrow = input->shape[0];
    int ncol = input->shape[1];
    int threads = 0;
    if (nrow <= 1024) threads = nrow;
    else threads = 1024;
    int blocks = (nrow + 1023) / 1024;
    
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    
    softmax_kernel<<<blocks, threads>>>(nrow, ncol, input_data, output_data);
    return 0;
}

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
    // Dynamic shared memory, size provided at kernel launch.
    extern __shared__ float loss_per_row[];
    // Two dimensional thread blocks.
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= nrow) {
        return;
    }
    input_a += y * ncol;
    input_b += y * ncol;
    float maxval = *input_a;
    // Find max for a row.
    for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input_a[x]);
    }
    // Deduct by max for a row, and raise to exp.
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input_a[x] - maxval);
    }
    // Compute per-row loss.
    float loss = 0;
    for (int x = 0; x < ncol; ++x) {
        loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
    }
    loss_per_row[y] = loss;
    __syncthreads();
    // Compute reduce_mean across rows.
    float mean_loss = 0;
    // Use a single thread to reduce mean across rows.
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        for (int i = 0; i < nrow; ++i) {
            mean_loss += loss_per_row[i];
        }
        mean_loss /= nrow;
        output[0] = mean_loss;
    }
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(input_b->ndim == 2);
    assert(output->ndim == 1);
    assert(input_a->shape[0] == input_b->shape[0] &&
        input_a->shape[1] == input_b->shape[1]);
    int nrow = input_a->shape[0];
    // Maximum x- or y-dimension of a block = 1024
    // But we need 'nrow' shared memory, and max shared memory is 48KB.
    // Conservatively allow max 16KB shared memory.
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float *input_data_a = (const float *)input_a->data;
    const float *input_data_b = (const float *)input_b->data;
    float *output_data = (float *)output->data;
    int threads = 0;
    if (nrow <= 1024) threads = nrow;
    else threads = 1024;
    int blocks = (nrow + 1023) / 1024;
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_softmax_cross_entropy_kernel<<<blocks, threads, nrow * sizeof(float)>>>(
        nrow, ncol, input_data_a, input_data_b, output_data);
    return 0;
}
