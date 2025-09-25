#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CHECK(err)                                                      \
    {                                                                        \
        cudaError_t err_code = err;                                          \
        if (err_code != cudaSuccess)                                         \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err_code)      \
                      << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f
#define INV_TWO_PI 0.15915494309189533577f // 1/(2*pi)

__forceinline__ __device__ float
clampf(float x, float a, float b)
{
    return fminf(b, fmaxf(a, x));
}

__forceinline__ __device__ float mod2pi(float phi)
{
    // Fast normalization into [0, 2π)
    phi -= floorf(phi * INV_TWO_PI) * TWO_PI;
    return phi;
}

struct State
{
    float t, r, th, ph;
    float vt, vr, vth, vph;
};

__forceinline__ __device__ void christoffels(
    float r, float b, float theta,
    float &g112, float &g133, float &g212, float &g233, float &g323)
{
    float s, c;
    sincosf(theta, &s, &c);
    g112 = -r;                  // Γ^r_{θθ}
    g133 = -r * s * s;          // Γ^r_{φφ}
    g212 = r / (b * b + r * r); // Γ^θ_{rθ} = Γ^θ_{θr} = Γ^φ_{rφ} = Γ^φ_{φr}
    g233 = -s * c;              // Γ^θ_{φφ}
    g323 = c / (s + 1e-12f);    // Γ^φ_{θφ} = Γ^φ_{φθ}
}

__forceinline__ __device__ void rhs(const State &s, float b, State &out)
{
    float g112, g133, g212, g233, g323;
    christoffels(s.r, b, s.th, g112, g133, g212, g233, g323);

    const float vt = s.vt, vr = s.vr, vth = s.vth, vph = s.vph;

    float ar = -g112 * vth * vth - g133 * vph * vph;
    float ath = -g212 * (vr * vth + vth * vr) - g233 * vph * vph;
    float aph = -g212 * (vr * vph + vph * vr) - g323 * (vth * vph + vph * vth);

    out.t = vt;
    out.r = vr;
    out.th = vth;
    out.ph = vph;
    out.vt = 0.0f;
    out.vr = ar;
    out.vth = ath;
    out.vph = aph;
}

__forceinline__ __device__ void rk4_step(State &s, float dt, float b)
{
    State k1, k2, k3, k4;
    State tmp;

    rhs(s, b, k1);

    tmp.t = __fmaf_rn(0.5f * dt, k1.t, s.t);
    tmp.r = __fmaf_rn(0.5f * dt, k1.r, s.r);
    tmp.th = __fmaf_rn(0.5f * dt, k1.th, s.th);
    tmp.ph = __fmaf_rn(0.5f * dt, k1.ph, s.ph);
    tmp.vt = __fmaf_rn(0.5f * dt, k1.vt, s.vt);
    tmp.vr = __fmaf_rn(0.5f * dt, k1.vr, s.vr);
    tmp.vth = __fmaf_rn(0.5f * dt, k1.vth, s.vth);
    tmp.vph = __fmaf_rn(0.5f * dt, k1.vph, s.vph);
    rhs(tmp, b, k2);

    tmp.t = __fmaf_rn(0.5f * dt, k2.t, s.t);
    tmp.r = __fmaf_rn(0.5f * dt, k2.r, s.r);
    tmp.th = __fmaf_rn(0.5f * dt, k2.th, s.th);
    tmp.ph = __fmaf_rn(0.5f * dt, k2.ph, s.ph);
    tmp.vt = __fmaf_rn(0.5f * dt, k2.vt, s.vt);
    tmp.vr = __fmaf_rn(0.5f * dt, k2.vr, s.vr);
    tmp.vth = __fmaf_rn(0.5f * dt, k2.vth, s.vth);
    tmp.vph = __fmaf_rn(0.5f * dt, k2.vph, s.vph);
    rhs(tmp, b, k3);

    tmp.t = __fmaf_rn(dt, k3.t, s.t);
    tmp.r = __fmaf_rn(dt, k3.r, s.r);
    tmp.th = __fmaf_rn(dt, k3.th, s.th);
    tmp.ph = __fmaf_rn(dt, k3.ph, s.ph);
    tmp.vt = __fmaf_rn(dt, k3.vt, s.vt);
    tmp.vr = __fmaf_rn(dt, k3.vr, s.vr);
    tmp.vth = __fmaf_rn(dt, k3.vth, s.vth);
    tmp.vph = __fmaf_rn(dt, k3.vph, s.vph);
    rhs(tmp, b, k4);

    const float w = dt / 6.0f;
    s.t = __fmaf_rn(w, (k1.t + 2.f * k2.t + 2.f * k3.t + k4.t), s.t);
    s.r = __fmaf_rn(w, (k1.r + 2.f * k2.r + 2.f * k3.r + k4.r), s.r);
    s.th = __fmaf_rn(w, (k1.th + 2.f * k2.th + 2.f * k3.th + k4.th), s.th);
    s.ph = __fmaf_rn(w, (k1.ph + 2.f * k2.ph + 2.f * k3.ph + k4.ph), s.ph);
    s.vt = __fmaf_rn(w, (k1.vt + 2.f * k2.vt + 2.f * k3.vt + k4.vt), s.vt);
    s.vr = __fmaf_rn(w, (k1.vr + 2.f * k2.vr + 2.f * k3.vr + k4.vr), s.vr);
    s.vth = __fmaf_rn(w, (k1.vth + 2.f * k2.vth + 2.f * k3.vth + k4.vth), s.vth);
    s.vph = __fmaf_rn(w, (k1.vph + 2.f * k2.vph + 2.f * k3.vph + k4.vph), s.vph);
}

__forceinline__ __device__ void init_camera_basis(float th0, float ph0, float3 &e_r, float3 &e_th, float3 &e_ph)
{
    float st, ct;
    float sp, cp;
    sincosf(th0, &st, &ct);
    sincosf(ph0, &sp, &cp);
    e_r = make_float3(st * cp, st * sp, ct);
    e_th = make_float3(ct * cp, ct * sp, -st);
    e_ph = make_float3(-sp, cp, 0.0f);
}

__forceinline__ __device__ void pixel_to_direction(
    int i, int j, int W, int H, float tanHalfFov, float aspect,
    const float3 &e_r, const float3 &e_th, const float3 &e_ph,
    float &c_r, float &c_th, float &c_ph)
{
    // pixel to normalized screen offsets (u: right, v: up)
    float u = ((j + 0.5f) / (float)W - 0.5f) * 2.0f * tanHalfFov;
    float v = (0.5f - (i + 0.5f) / (float)H) * 2.0f * tanHalfFov * aspect;

    float3 d = make_float3(
        -e_r.x + u * e_ph.x + v * e_th.x,
        -e_r.y + u * e_ph.y + v * e_th.y,
        -e_r.z + u * e_ph.z + v * e_th.z);

    float invn = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    d.x *= invn;
    d.y *= invn;
    d.z *= invn;

    c_r = d.x * e_r.x + d.y * e_r.y + d.z * e_r.z;
    c_th = d.x * e_th.x + d.y * e_th.y + d.z * e_th.z;
    c_ph = d.x * e_ph.x + d.y * e_ph.y + d.z * e_ph.z;
}

__forceinline__ __device__
    uint32_t
    get_rgb(const unsigned char *__restrict__ img, int width, int x, int y)
{
    int idx = 3 * (y * width + x);
    return ((uint32_t)img[idx] << 16) |
           ((uint32_t)img[idx + 1] << 8) |
           (uint32_t)img[idx + 2];
}

__forceinline__ __device__
    uint32_t
    map_coordinates_to_pixel(
        float r, float theta, float phi,
        float th0, float ph0,
        const unsigned char *__restrict__ space1, int width1, int height1,
        const unsigned char *__restrict__ space2, int width2, int height2)
{
    const unsigned char *__restrict__ space = (r > 0.0f) ? space1 : space2;
    const int width = (r > 0.0f) ? width1 : width2;
    const int height = (r > 0.0f) ? height1 : height2;

    if (r < 0.0f)
        phi += ph0 + PI; // flip to avoid seam
    phi = mod2pi(phi);
    theta = clampf(theta, 0.0f, PI);

    float xf = (phi * (0.5f / PI)) * (float)(width - 1);
    float yf = (theta * (1.0f / PI)) * (float)(height - 1);

    int x = (int)clampf(xf, 0.0f, (float)(width - 1));
    int y = (int)clampf(yf, 0.0f, (float)(height - 1));

    return get_rgb(space, width, x, y);
}

__forceinline__ __device__ void init_state(
    float r0, float th0, float ph0, float b,
    float c_r, float c_th, float c_ph,
    State &s)
{
    float R = sqrtf(r0 * r0 + b * b);
    float S = fmaxf(sinf(th0), 1e-9f);

    float vr = c_r;
    float vth = c_th / R;
    float vph = c_ph / (R * S);
    float vt = sqrtf(vr * vr + R * R * (vth * vth + (S * S) * vph * vph));

    s = {0.0f, r0, th0, ph0, vt, vr, vth, vph};
}

__forceinline__ __device__
    uint32_t
    trace_geodesic(
        State &s, float dt, float tmax, float b,
        const unsigned char *__restrict__ space1, int w1, int h1,
        const unsigned char *__restrict__ space2, int w2, int h2)
{
    const float th0 = s.th;
    const float ph0 = s.ph;
    const int steps = (int)(tmax / dt);
    for (int i = 0; i < steps; ++i)
    {
        rk4_step(s, dt, b);
    }
    return map_coordinates_to_pixel(
        s.r, s.th, s.ph, th0, ph0,
        space1, w1, h1, space2, w2, h2);
}

// ----------------- kernel -----------------
#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

__launch_bounds__(BLOCK_DIM, 2)
    __global__ void wormhole_kernel(
        unsigned char *__restrict__ output, // 3 bytes per pixel
        int W, int H,
        const unsigned char *__restrict__ space1, int w1, int h1,
        const unsigned char *__restrict__ space2, int w2, int h2, float ph0)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float fov = 120.0f * (float)PI / 180.0f;
    const float tanHalf = tanf(0.5f * fov);
    const float aspect = (float)H / (float)W;

    const float b = 3.0f;
    const float dt = 1e-3f;
    const float tmax = 30.0f;

    const float r0 = 10.0f;
    const float th0 = (float)PI * 0.5f;

    float3 e_r, e_th, e_ph;
    init_camera_basis(th0, ph0, e_r, e_th, e_ph);

    for (int idx = tid; idx < W * H; idx += stride)
    {
        int i = idx / W;
        int j = idx % W;

        float c_r, c_th, c_ph;
        pixel_to_direction(i, j, W, H, tanHalf, aspect, e_r, e_th, e_ph, c_r, c_th, c_ph);

        State s;
        init_state(r0, th0, ph0, b, c_r, c_th, c_ph, s);

        uint32_t pix = trace_geodesic(s, dt, tmax, b, space1, w1, h1, space2, w2, h2);

        int p = 3 * idx;
        output[p + 0] = (pix >> 16) & 0xFF;
        output[p + 1] = (pix >> 8) & 0xFF;
        output[p + 2] = (pix) & 0xFF;
    }
}

void trace_wormhole(
    unsigned char *output, int W, int H,
    const unsigned char *space1, int w1, int h1,
    const unsigned char *space2, int w2, int h2)
{
    // device buffers
    unsigned char *d_out = nullptr;
    unsigned char *d_space1 = nullptr;
    unsigned char *d_space2 = nullptr;

    const size_t out_size = (size_t)W * H * 3;
    const size_t s1_size = (size_t)w1 * h1 * 3;
    const size_t s2_size = (size_t)w2 * h2 * 3;

    CUDA_CHECK(cudaMalloc(&d_out, out_size));
    CUDA_CHECK(cudaMalloc(&d_space1, s1_size));
    CUDA_CHECK(cudaMalloc(&d_space2, s2_size));

    CUDA_CHECK(cudaMemcpy(d_space1, space1, s1_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_space2, space2, s2_size, cudaMemcpyHostToDevice));

    // launch
    const int N = W * H;
    const int block = BLOCK_DIM;
    // Cap grid to avoid oversubscription: ~32 * SMs is usually good. Use a big default if unknown.
    int grid = (N + block - 1) / block;
    grid = min(grid, 65535); // safe cap for many GPUs

    const int duration = 30;
    const int fps = 30;
    const int frames = duration * fps;
    const float start_ph0 = -M_PI / 4;
    const float end_ph0 = M_PI / 4;
    const float dph = (end_ph0 - start_ph0) / frames;

    for (int i = 0; i < frames; i++)
    {
        float ph0 = start_ph0 + i * dph;
        wormhole_kernel<<<grid, block>>>(d_out, W, H, d_space1, w1, h1, d_space2, w2, h2, ph0);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(output, d_out, out_size, cudaMemcpyDeviceToHost));

        std::string filename = "output/wormhole_" + std::to_string(i) + ".png";
        int ok = stbi_write_png(filename.c_str(), W, H, 3, output, W * 3);
        if (ok)
            std::cout << i << "/" << frames << std::endl;
        else
            std::cerr << "Error: Failed to save image.\n";
    }

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_space1));
    CUDA_CHECK(cudaFree(d_space2));
}

int main()
{
    int w1, h1, w2, h2;
    int outW = 3840, outH = 2160;

    // Load as 3-channel RGB
    int comp1 = 0, comp2 = 0;
    unsigned char *space1 = stbi_load("space5.jpg", &w1, &h1, &comp1, 3);
    unsigned char *space2 = stbi_load("space1.jpg", &w2, &h2, &comp2, 3);

    if (!space1 || !space2)
    {
        std::cerr << "Failed to load images." << std::endl;
        return 1;
    }

    unsigned char *output = (unsigned char *)malloc((size_t)outW * outH * 3);

    std::printf("space1: %dx%d\n", w1, h1);
    std::printf("space2: %dx%d\n", w2, h2);
    std::printf("output: %dx%d\n", outW, outH);

    trace_wormhole(output, outW, outH, space1, w1, h1, space2, w2, h2);

    stbi_image_free(space1);
    stbi_image_free(space2);
    free(output);
    return 0;
}