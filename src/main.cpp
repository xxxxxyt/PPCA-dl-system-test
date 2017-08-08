#include <iostream>
#include <cstdio>
//#include <omp.h>
#include <cblas.h>
using namespace std;

extern "C" {
    
void matmul(const float *A, const float *B, float *C,
            bool TA, bool TB, int m, int k, int n) {
    //puts("in matmul");
    const CBLAS_ORDER order = CblasRowMajor;
    const CBLAS_TRANSPOSE trans_A = TA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE trans_B = TB ? CblasTrans : CblasNoTrans;
    const float alpha = 1.0;
    const float beta = 0.0;
    const int lda = TA ? m : k;
    const int ldb = TB ? k : n;
    const int ldc = n;
    cblas_sgemm(order, trans_A, trans_B, 
                m, n, k, alpha, 
                A, lda, 
                B, ldb, beta, 
                C, ldc);
    //puts("out matmul");
}

void conv2d(const float *X, const float *W, float *Y,
            int batch,
            int fil_h, int fil_w,
            int in_h, int in_w, int in_ch,
            int ou_h, int ou_w, int ou_ch,
            int strides1, int strides2) {
    //puts("in conv2d");
    float *X_col = new float[ou_h * ou_w * fil_h * fil_w * in_ch];
    //#pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        //printf("b %d\n", b);
        const float *X_batch = X + b * in_h * in_w * in_ch;
        float *Y_batch = Y + b * ou_h * ou_w * ou_ch;
        float *subX_col = X_col;
        int p = 0;
        for (int i = 0; i < ou_h; ++i) {
            int q = 0;
            for (int j = 0; j < ou_w; ++j) {
                int index = 0;
                for (int x = p; x < p + fil_h; ++x)
                    for (int y = q; y < q + fil_w; ++y)
                        for (int k = 0; k < in_ch; ++k)
                            subX_col[index++] = 
                                X_batch[x * in_w * in_ch + y * in_ch + k];
                q += strides2;
                subX_col += fil_h * fil_w * in_ch;
            }
            p += strides1;
        }
        
        matmul(X_col, W, Y_batch,
               false, false, 
               ou_h * ou_w,
               fil_h * fil_w * in_ch,
               ou_ch);
    }
    delete [] X_col;
    //puts("out conv2d");
}

void conv2d_gradient_x(const float *D, const float *W, float *DX,
                       int batch,
                       int fil_h, int fil_w,
                       int in_h, int in_w, int in_ch,
                       int ou_h, int ou_w, int ou_ch,
                       int strides1, int strides2) {
    //puts("in conv2d_gradient_x");
    float *DX_col = new float[ou_h * ou_w * fil_h * fil_w * in_ch];
    //#pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        //printf("b %d\n", b);
        const float *D_batch = D + b * ou_h * ou_w * ou_ch;
        float *DX_batch = DX + b * in_h * in_w * in_ch;
        float *subDX_col = DX_col;
        matmul(D_batch, W, DX_col,
               false, true,
               ou_h * ou_w,
               ou_ch,
               fil_h * fil_w * in_ch);
        int p = 0;
        for (int i = 0; i < ou_h; ++i) {
            int q = 0;
            for (int j = 0; j < ou_w; ++j) {
                int index = 0;
                for (int x = p; x < p + fil_h; ++x)
                    for (int y = q; y < q + fil_w; ++y)
                        for (int k = 0; k < in_ch; ++k)
                            DX_batch[x * fil_w * in_ch + y * in_ch + k] = 
                                subDX_col[index++];
                q += strides1;
                subDX_col += fil_h * fil_w * in_ch;
            }
            p += strides2;
        }
    }
    delete [] DX_col;
    //puts("out conv2d_gradient_x");
}

void conv2d_gradient_w(const float *D, const float *X, float *DW,
                       int batch,
                       int fil_h, int fil_w,
                       int in_h, int in_w, int in_ch,
                       int ou_h, int ou_w, int ou_ch,
                       int strides1, int strides2) {
    //puts("in conv2d_gradient_w");
    float *DW_batch = new float[fil_h * fil_w * in_ch * ou_ch];
    float *X_col = new float[ou_h * ou_w * fil_h * fil_w * in_ch];
    //#pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        const float *D_batch = D + b * ou_h * ou_w * ou_ch;
        const float *X_batch = X + b * in_h * in_w * in_ch;
        float *subX_col = X_col;
        int p = 0;
        for (int i = 0; i < ou_h; ++i) {
            int q = 0;
            for (int j = 0; j < ou_w; ++j) {
                int index = 0;
                for (int x = p; x < p + fil_h; ++x)
                    for (int y = q; y < q + fil_w; ++y)
                        for (int k = 0; k < in_ch; ++k)
                            subX_col[index++] = 
                                X_batch[x * in_w * in_ch + y * in_ch + k];
                q += strides2;
                subX_col += fil_h * fil_w * in_ch;
            }
            p += strides1;
        }
        matmul(X_col, D_batch, DW_batch,
               true, false,
               fil_h * fil_w * in_ch,
               ou_h * ou_w,
               ou_ch);
        for (int i = 0; i < fil_h * fil_w * in_ch * ou_ch; ++i) 
            DW[i] += DW_batch[i];
    }
    delete [] DW_batch;
    delete [] X_col;
    //puts("out conv2d_gradient_w");
}

}
