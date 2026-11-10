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

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::cerr << #x << ": " << cudaGetErrorString(e) << "\n"; std::exit(1); } } while (0)

__host__ __device__ inline int ci(int v, int n) {
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

__host__ __device__ inline unsigned char blurP(const unsigned char* in, int w, int h, int x, int y) {
    int s = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            s += in[ci(y + dy, h) * w + ci(x + dx, w)];
    return (unsigned char)(s / 9);
}

__host__ __device__ inline unsigned char sobelP(const unsigned char* in, int w, int h, int x, int y) {
    const int kx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int ky[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    int gx = 0, gy = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int v = in[ci(y + dy, h) * w + ci(x + dx, w)];
            gx += v * kx[dy+1][dx+1];
            gy += v * ky[dy+1][dx+1];
        }
    int m = (int)sqrtf((float)(gx*gx + gy*gy));
    return (unsigned char)(m > 255 ? 255 : m);
}

__host__ __device__ inline unsigned char medianP(const unsigned char* in, int w, int h, int x, int y) {
    unsigned char v[9];
    int k = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            v[k++] = in[ci(y+dy,h)*w + ci(x+dx,w)];
    for (int i = 1; i < 9; ++i) {
        unsigned char t = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > t) { v[j+1] = v[j]; --j; }
        v[j+1] = t;
    }
    return v[4];
}

__global__ void blurK(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < w && y < h) out[y*w+x] = blurP(in, w, h, x, y);
}
__global__ void sobelK(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < w && y < h) out[y*w+x] = sobelP(in, w, h, x, y);
}
__global__ void medianK(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < w && y < h) out[y*w+x] = medianP(in, w, h, x, y);
}

void cpuRun(const std::string& f, const unsigned char* in, unsigned char* out, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (f == "blur")    out[y*w+x] = blurP(in, w, h, x, y);
            else if (f == "sobel") out[y*w+x] = sobelP(in, w, h, x, y);
            else                out[y*w+x] = medianP(in, w, h, x, y);
        }
}

void gpuRun(const std::string& f, const unsigned char* d_in, unsigned char* d_out, int w, int h) {
    dim3 bl(16,16), gr((w+15)/16, (h+15)/16);
    if (f == "blur")       blurK<<<gr,bl>>>(d_in, d_out, w, h);
    else if (f == "sobel") sobelK<<<gr,bl>>>(d_in, d_out, w, h);
    else                   medianK<<<gr,bl>>>(d_in, d_out, w, h);
    CK(cudaGetLastError());
}

unsigned char* readGray(const char* fn, int& w, int& h) {
    png_image img{}; img.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_file(&img, fn)) { return nullptr; }
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
    img.width = (png_uint_32)w; img.height = (png_uint_32)h; img.format = PNG_FORMAT_GRAY;
    return png_image_write_to_file(&img, fn, 0, data, 0, nullptr);
}

void genSynthetic(const char* fn, int w, int h) {
    std::vector<unsigned char> img(h*w, 128);
    auto rect = [&](int y0,int y1,int x0,int x1,int v){
        for (int y=y0;y<y1;y++) for (int x=x0;x<x1;x++) img[y*w+x]=(unsigned char)v;
    };
    // координаты заданы на базовой сетке 512x512 и масштабируются под w x h
    const double sy = (double)h / 512.0;
    const double sx = (double)w / 512.0;
    auto R = [&](int y0,int y1,int x0,int x1,int v){
        rect((int)(y0*sy), (int)(y1*sy), (int)(x0*sx), (int)(x1*sx), v);
    };
    R( 50, 180,  50, 180, 220);
    R(220, 400, 220, 400,  40);
    R( 80, 420, 350, 470,  60);
    srand(42);
    for (int i=0;i<h*w;i++) {
        int r = rand()%100;
        if (r < 5) img[i]=0; else if (r < 10) img[i]=255;
    }
    writeGray(fn, img.data(), w, h);
}

int main(int argc, char** argv) {
    // размер синтетического изображения по умолчанию — 4096x4096
    const int GEN_W = 4096, GEN_H = 4096;

    const char* inFn = (argc > 1) ? argv[1] : "input.png";
    int w = 0, h = 0;
    unsigned char* in = readGray(inFn, w, h);
    if (!in) {
        std::cerr << "generating " << inFn << " (" << GEN_W << "x" << GEN_H << ")\n";
        genSynthetic(inFn, GEN_W, GEN_H);
        in = readGray(inFn, w, h);
        if (!in) { std::cerr << "cannot read " << inFn << "\n"; return 1; }
    }
    const size_t n = (size_t)w*h;
    unsigned char *outCpu = (unsigned char*)malloc(n);
    unsigned char *outGpu = (unsigned char*)malloc(n);
    unsigned char *d_in = nullptr, *d_out = nullptr;
    CK(cudaMalloc((void**)&d_in, n));
    CK(cudaMalloc((void**)&d_out, n));

    const int REP = 10;
    auto now = [](){ return std::chrono::high_resolution_clock::now(); };
    auto ms  = [](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };

    std::cout << "Image: " << w << "x" << h << "\n\n";
    std::cout << std::left << std::setw(8)  << "Filter"
              << std::right << std::setw(10) << "CPU,ms"
              << std::setw(12) << "GPU_kern,ms"
              << std::setw(12) << "GPU_tot,ms"
              << std::setw(14) << "Overhead,ms"
              << std::setw(10) << "Spd_kern"
              << std::setw(10) << "Spd_tot"
              << std::setw(8)  << "Diff" << "\n";
    std::cout << std::string(84, '-') << "\n";

    for (const std::string& f : {"blur","sobel","median"}) {
        // warmup
        CK(cudaMemcpy(d_in, in, n, cudaMemcpyHostToDevice));
        gpuRun(f, d_in, d_out, w, h);
        CK(cudaDeviceSynchronize());
        cpuRun(f, in, outCpu, w, h);

        double cpu_ms = 0;
        for (int r=0;r<REP;r++){ auto t1=now(); cpuRun(f,in,outCpu,w,h); auto t2=now(); cpu_ms+=ms(t1,t2); }
        cpu_ms /= REP;

        double gpu_ms = 0;
        for (int r=0;r<REP;r++){
            auto t1=now();
            CK(cudaMemcpy(d_in, in, n, cudaMemcpyHostToDevice));
            gpuRun(f, d_in, d_out, w, h);
            CK(cudaMemcpy(outGpu, d_out, n, cudaMemcpyDeviceToHost));
            auto t2=now();
            gpu_ms += ms(t1,t2);
        }
        gpu_ms /= REP;

        cudaEvent_t e1,e2; CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
        CK(cudaEventRecord(e1));
        for (int r=0;r<REP;r++) gpuRun(f, d_in, d_out, w, h);
        CK(cudaEventRecord(e2)); CK(cudaEventSynchronize(e2));
        float ktot=0; CK(cudaEventElapsedTime(&ktot,e1,e2));
        double kernel_ms = ktot / REP;
        CK(cudaEventDestroy(e1)); CK(cudaEventDestroy(e2));

        std::string outFn = "out_" + f + ".png";
        writeGray(outFn.c_str(), outGpu, w, h);

        int diffs=0;
        for (size_t i=0;i<n;i++) if (std::abs((int)outCpu[i]-(int)outGpu[i])>1) ++diffs;

        std::cout << std::left << std::setw(8) << f
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << cpu_ms
                  << std::setw(12) << kernel_ms
                  << std::setw(12) << gpu_ms
                  << std::setw(14) << (gpu_ms - kernel_ms)
                  << std::setprecision(1)
                  << std::setw(10) << (cpu_ms / kernel_ms)
                  << std::setw(10) << (cpu_ms / gpu_ms)
                  << std::setprecision(0)
                  << std::setw(8)  << (double)diffs << "\n";
    }

    CK(cudaFree(d_in)); CK(cudaFree(d_out));
    free(in); free(outCpu); free(outGpu);
    return 0;
}