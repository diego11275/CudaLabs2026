#include <mpi.h>
#include <png.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::cerr << #x << ": " << cudaGetErrorString(e) << "\n"; \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

__host__ __device__ inline int ci(int v, int n) {
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}
__host__ __device__ inline unsigned char blurP(const unsigned char* in, int w, int h, int x, int y) {
    int s = 0;
    for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx)
        s += in[ci(y+dy,h)*w + ci(x+dx,w)];
    return (unsigned char)(s/9);
}
__host__ __device__ inline unsigned char sobelP(const unsigned char* in, int w, int h, int x, int y) {
    const int kx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    const int ky[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
    int gx=0, gy=0;
    for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
        int v = in[ci(y+dy,h)*w + ci(x+dx,w)];
        gx += v*kx[dy+1][dx+1]; gy += v*ky[dy+1][dx+1];
    }
    int m = (int)sqrtf((float)(gx*gx + gy*gy));
    return (unsigned char)(m > 255 ? 255 : m);
}
__host__ __device__ inline unsigned char medianP(const unsigned char* in, int w, int h, int x, int y) {
    unsigned char v[9]; int k = 0;
    for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx)
        v[k++] = in[ci(y+dy,h)*w + ci(x+dx,w)];
    for (int i=1; i<9; ++i) { unsigned char t=v[i]; int j=i-1;
        while (j>=0 && v[j]>t) { v[j+1]=v[j]; --j; } v[j+1]=t; }
    return v[4];
}

__global__ void blurK(const unsigned char* in, unsigned char* out, int w, int h, int y0, int y1) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y + y0;
    if (x < w && y < y1) out[y*w+x] = blurP(in, w, h, x, y);
}
__global__ void sobelK(const unsigned char* in, unsigned char* out, int w, int h, int y0, int y1) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y + y0;
    if (x < w && y < y1) out[y*w+x] = sobelP(in, w, h, x, y);
}
__global__ void medianK(const unsigned char* in, unsigned char* out, int w, int h, int y0, int y1) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y + y0;
    if (x < w && y < y1) out[y*w+x] = medianP(in, w, h, x, y);
}

void cpuRun(const std::string& f, const unsigned char* in, unsigned char* out, int w, int h) {
    for (int y=0; y<h; ++y) for (int x=0; x<w; ++x) {
        if      (f=="blur")   out[y*w+x] = blurP(in,w,h,x,y);
        else if (f=="sobel")  out[y*w+x] = sobelP(in,w,h,x,y);
        else                  out[y*w+x] = medianP(in,w,h,x,y);
    }
}
void gpuRun(const std::string& f, const unsigned char* d_in, unsigned char* d_out,
            int w, int h, int y0, int y1) {
    int rows = y1 - y0;
    dim3 bl(16,16), gr((w+15)/16, (rows+15)/16);
    if      (f=="blur")   blurK  <<<gr,bl>>>(d_in, d_out, w, h, y0, y1);
    else if (f=="sobel")  sobelK <<<gr,bl>>>(d_in, d_out, w, h, y0, y1);
    else                  medianK<<<gr,bl>>>(d_in, d_out, w, h, y0, y1);
    CK(cudaGetLastError());
}

unsigned char* readGray(const char* fn, int& w, int& h) {
    png_image img{}; img.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_file(&img, fn)) return nullptr;
    img.format = PNG_FORMAT_GRAY;
    unsigned char* buf = (unsigned char*)malloc(PNG_IMAGE_SIZE(img));
    if (!buf || !png_image_finish_read(&img, nullptr, buf, 0, nullptr)) {
        free(buf); png_image_free(&img); return nullptr;
    }
    w = (int)img.width; h = (int)img.height; png_image_free(&img);
    return buf;
}
bool writeGray(const char* fn, const unsigned char* data, int w, int h) {
    png_image img{}; img.version = PNG_IMAGE_VERSION;
    img.width = (png_uint_32)w; img.height = (png_uint_32)h;
    img.format = PNG_FORMAT_GRAY;
    return png_image_write_to_file(&img, fn, 0, data, 0, nullptr);
}
void genSynthetic(const char* fn, int w, int h) {
    std::vector<unsigned char> img(h*w, 128);
    const double sy = (double)h/512.0, sx = (double)w/512.0;
    auto R = [&](int y0,int y1,int x0,int x1,int v){
        for (int y=(int)(y0*sy); y<(int)(y1*sy); y++)
            for (int x=(int)(x0*sx); x<(int)(x1*sx); x++) img[y*w+x]=(unsigned char)v;
    };
    R( 50,180, 50,180,220); R(220,400,220,400,40); R(80,420,350,470,60);
    srand(42);
    for (int i=0;i<h*w;i++) { int r=rand()%100;
        if (r<5) img[i]=0; else if (r<10) img[i]=255; }
    writeGray(fn, img.data(), w, h);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int devCount = 0; CK(cudaGetDeviceCount(&devCount));
    if (devCount < 1) { std::cerr << "no CUDA device on rank " << rank << "\n";
                        MPI_Abort(MPI_COMM_WORLD, 1); }
    CK(cudaSetDevice(0));   // каждая нода видит свою локальную GPU

    const int GEN_W = 4096, GEN_H = 4096;
    const char* inFn = (argc > 1) ? argv[1] : "input.png";
    int w = 0, h = 0;
    unsigned char* in = nullptr;

    if (rank == 0) {
        in = readGray(inFn, w, h);
        if (!in) {
            std::cerr << "generating " << inFn << " (" << GEN_W << "x" << GEN_H << ")\n";
            genSynthetic(inFn, GEN_W, GEN_H);
            in = readGray(inFn, w, h);
        }
    }
    MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    size_t n = (size_t)w * h;
    if (rank != 0) in = (unsigned char*)malloc(n);
    MPI_Bcast(in, (int)n, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Срез строк этого ранга
    int y0 = (int)((long long)h * rank / nranks);
    int y1 = (int)((long long)h * (rank+1) / nranks);
    int rows = y1 - y0;
    size_t local_n = (size_t)rows * w;

    // Копируем на устройство срез + 1 строка гало с каждой стороны
    int cy0 = std::max(0, y0 - 1);
    int cy1 = std::min(h, y1 + 1);
    size_t copy_in_n = (size_t)(cy1 - cy0) * w;

    unsigned char *d_in, *d_out;
    CK(cudaMalloc(&d_in, n));
    CK(cudaMalloc(&d_out, n));

    unsigned char* outLocal   = (unsigned char*)malloc(local_n);
    unsigned char* outFull    = (rank == 0) ? (unsigned char*)malloc(n) : nullptr;
    unsigned char* outCpu     = (rank == 0) ? (unsigned char*)malloc(n) : nullptr;

    std::vector<int> rc(nranks), displ(nranks);
    for (int r=0; r<nranks; ++r) {
        int ry0 = (int)((long long)h * r / nranks);
        int ry1 = (int)((long long)h * (r+1) / nranks);
        rc[r]    = (ry1 - ry0) * w;
        displ[r] = ry0 * w;
    }

    const int REP = 10;
    auto now = [](){ return std::chrono::high_resolution_clock::now(); };
    auto ms  = [](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };

    if (rank == 0) {
        std::cout << "Image: " << w << "x" << h << " | GPUs (MPI ranks): " << nranks << "\n\n";
        std::cout << std::left << std::setw(8) << "Filter"
                  << std::right << std::setw(10) << "CPU,ms"
                  << std::setw(12) << "GPU_kern,ms"
                  << std::setw(12) << "GPU_tot,ms"
                  << std::setw(10) << "Spd_kern"
                  << std::setw(10) << "Spd_tot"
                  << std::setw(8)  << "Diff" << "\n";
        std::cout << std::string(74, '-') << "\n";
    }

    for (const std::string& f : {"blur", "sobel", "median"}) {
        // Разогрев
        CK(cudaMemcpy(d_in + (size_t)cy0*w, in + (size_t)cy0*w, copy_in_n, cudaMemcpyHostToDevice));
        gpuRun(f, d_in, d_out, w, h, y0, y1);
        CK(cudaDeviceSynchronize());

        // CPU-эталон (только на rank 0)
        double cpu_ms = 0;
        if (rank == 0) {
            cpuRun(f, in, outCpu, w, h);
            for (int r=0; r<REP; ++r) { auto t1=now(); cpuRun(f,in,outCpu,w,h); auto t2=now(); cpu_ms += ms(t1,t2); }
            cpu_ms /= REP;
        }

        // Чистое время ядра (параллельно на 2 GPU)
        cudaEvent_t e1, e2;
        CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
        CK(cudaEventRecord(e1));
        for (int r=0; r<REP; ++r) gpuRun(f, d_in, d_out, w, h, y0, y1);
        CK(cudaEventRecord(e2)); CK(cudaEventSynchronize(e2));
        float ktot = 0; CK(cudaEventElapsedTime(&ktot, e1, e2));
        double kernel_ms = ktot / REP;
        CK(cudaEventDestroy(e1)); CK(cudaEventDestroy(e2));

        // Полный конвейер (H2D -> kernel -> D2H) на своём срезе
        double gpu_ms = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        for (int r=0; r<REP; ++r) {
            auto t1 = now();
            CK(cudaMemcpy(d_in + (size_t)cy0*w, in + (size_t)cy0*w, copy_in_n, cudaMemcpyHostToDevice));
            gpuRun(f, d_in, d_out, w, h, y0, y1);
            CK(cudaMemcpy(outLocal, d_out + (size_t)y0*w, local_n, cudaMemcpyDeviceToHost));
            auto t2 = now();
            gpu_ms += ms(t1, t2);
        }
        gpu_ms /= REP;
        MPI_Barrier(MPI_COMM_WORLD);

        // Собираем результат на rank 0
        MPI_Gatherv(outLocal, (int)local_n, MPI_UNSIGNED_CHAR,
                    outFull, rc.data(), displ.data(), MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::string outFn = "out_" + f + ".png";
            writeGray(outFn.c_str(), outFull, w, h);
            int diffs = 0;
            for (size_t i=0; i<n; ++i)
                if (std::abs((int)outCpu[i] - (int)outFull[i]) > 1) ++diffs;
            std::cout << std::left << std::setw(8) << f
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(10) << cpu_ms
                      << std::setw(12) << kernel_ms
                      << std::setw(12) << gpu_ms
                      << std::setprecision(1)
                      << std::setw(10) << (cpu_ms / kernel_ms)
                      << std::setw(10) << (cpu_ms / gpu_ms)
                      << std::setprecision(0)
                      << std::setw(8)  << (double)diffs << "\n";
        }
    }

    CK(cudaFree(d_in)); CK(cudaFree(d_out));
    free(in); free(outLocal); free(outFull); free(outCpu);
    MPI_Finalize();
    return 0;
}