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
    float t, l, th, ph;
    float vt, vl, vth, vph;
};

inline static float r_of_l(float l, float b, float L)
{
    return powf(pow(l, L) + pow(b, L), 1 / L);
}

inline static float r_prime_of_l(float l, float b, float L)
{
    return powf(pow(l, L) + pow(b, L), 1 / L - 1) * powf(l, L - 1);
}

__forceinline__ __device__ void christoffels(
    float l, float b, float L, float theta,
    float &g122, float &g133, float &g212, float &g233, float &g323)
{
    float st, ct;
    __sincosf(theta, &st, &ct);

    float r = r_of_l(l, b, L);
    float rp = r_prime_of_l(l, b, L);

    // Γ^l_{θθ}
    g122 = -r * rp;

    // Γ^l_{φφ}
    g133 = -r * rp * st * st;

    // Γ^θ_{lθ} = Γ^θ_{θl} = Γ^φ_{lφ} = Γ^φ_{φl}
    g212 = rp / (r + 1e-12);

    // Γ^θ_{φφ}
    g233 = -st * ct;

    // Γ^φ_{θφ} = Γ^φ_{φθ}
    g323 = ct / (st + 1e-12);
}

__forceinline__ __device__ void rhs(const State &s, float b, float L, State &out)
{
    float g122, g133, g212, g233, g323;
    christoffels(s.l, b, L, s.th, g122, g133, g212, g233, g323);

    float al = -g122 * s.vth * s.vth - g133 * s.vph * s.vph;
    float ath = -2 * g212 * s.vl * s.vth - g233 * s.vph * s.vph;
    float aph = -2 * g212 * s.vl * s.vph - 2 * g323 * s.vth * s.vph;

    out.t = s.vt;
    out.l = s.vl;
    out.th = s.vth;
    out.ph = s.vph;
    out.vt = 0;
    out.vl = al;
    out.vth = ath;
    out.vph = aph;
}

__forceinline__ __device__ void rk4_step(State &s, float dt, float b, float L)
{
    State k1, k2, k3, k4;
    State tmp;

    rhs(s, b, L, k1);

    tmp.t = __fmaf_rn(0.5f * dt, k1.t, s.t);
    tmp.l = __fmaf_rn(0.5f * dt, k1.l, s.l);
    tmp.th = __fmaf_rn(0.5f * dt, k1.th, s.th);
    tmp.ph = __fmaf_rn(0.5f * dt, k1.ph, s.ph);
    tmp.vt = __fmaf_rn(0.5f * dt, k1.vt, s.vt);
    tmp.vl = __fmaf_rn(0.5f * dt, k1.vl, s.vl);
    tmp.vth = __fmaf_rn(0.5f * dt, k1.vth, s.vth);
    tmp.vph = __fmaf_rn(0.5f * dt, k1.vph, s.vph);
    rhs(tmp, b, L, k2);

    tmp.t = __fmaf_rn(0.5f * dt, k2.t, s.t);
    tmp.l = __fmaf_rn(0.5f * dt, k2.l, s.l);
    tmp.th = __fmaf_rn(0.5f * dt, k2.th, s.th);
    tmp.ph = __fmaf_rn(0.5f * dt, k2.ph, s.ph);
    tmp.vt = __fmaf_rn(0.5f * dt, k2.vt, s.vt);
    tmp.vl = __fmaf_rn(0.5f * dt, k2.vl, s.vl);
    tmp.vth = __fmaf_rn(0.5f * dt, k2.vth, s.vth);
    tmp.vph = __fmaf_rn(0.5f * dt, k2.vph, s.vph);
    rhs(tmp, b, L, k3);

    tmp.t = __fmaf_rn(dt, k3.t, s.t);
    tmp.l = __fmaf_rn(dt, k3.l, s.l);
    tmp.th = __fmaf_rn(dt, k3.th, s.th);
    tmp.ph = __fmaf_rn(dt, k3.ph, s.ph);
    tmp.vt = __fmaf_rn(dt, k3.vt, s.vt);
    tmp.vl = __fmaf_rn(dt, k3.vl, s.vl);
    tmp.vth = __fmaf_rn(dt, k3.vth, s.vth);
    tmp.vph = __fmaf_rn(dt, k3.vph, s.vph);
    rhs(tmp, b, L, k4);

    const float w = dt / 6.0f;
    s.t = __fmaf_rn(w, (k1.t + 2.f * k2.t + 2.f * k3.t + k4.t), s.t);
    s.l = __fmaf_rn(w, (k1.l + 2.f * k2.l + 2.f * k3.l + k4.l), s.l);
    s.th = __fmaf_rn(w, (k1.th + 2.f * k2.th + 2.f * k3.th + k4.th), s.th);
    s.ph = __fmaf_rn(w, (k1.ph + 2.f * k2.ph + 2.f * k3.ph + k4.ph), s.ph);
    s.vt = __fmaf_rn(w, (k1.vt + 2.f * k2.vt + 2.f * k3.vt + k4.vt), s.vt);
    s.vl = __fmaf_rn(w, (k1.vl + 2.f * k2.vl + 2.f * k3.vl + k4.vl), s.vl);
    s.vth = __fmaf_rn(w, (k1.vth + 2.f * k2.vth + 2.f * k3.vth + k4.vth), s.vth);
    s.vph = __fmaf_rn(w, (k1.vph + 2.f * k2.vph + 2.f * k3.vph + k4.vph), s.vph);
}

__forceinline__ __device__ void init_world_basis(float th0, float ph0, float3 &e_r, float3 &e_th, float3 &e_ph)
{
    float st, ct;
    float sp, cp;
    __sincosf(th0, &st, &ct);
    __sincosf(ph0, &sp, &cp);
    e_r = make_float3(st * cp, st * sp, ct);
    e_th = make_float3(ct * cp, ct * sp, -st);
    e_ph = make_float3(-sp, cp, 0.0f);
}

__forceinline__ __device__ void init_camera_basis(
    float3 &e_r, float3 &e_th, float3 &e_ph,
    float3 &cam_r, float3 &cam_th, float3 &cam_ph,
    float r_angle, float th_angle, float ph_angle)
{
    float sr, cr;
    float sth, cth;
    float sph, cph;
    __sincosf(r_angle, &sr, &cr);
    __sincosf(th_angle, &sth, &cth);
    __sincosf(ph_angle, &sph, &cph);

    cam_r.x = -e_r.x;
    cam_r.y = -e_r.y;
    cam_r.z = -e_r.z;

    cam_th.x = -e_th.x;
    cam_th.y = -e_th.y;
    cam_th.z = -e_th.z;

    cam_ph.x = e_ph.x;
    cam_ph.y = e_ph.y;
    cam_ph.z = e_ph.z;

    // Rotation around cam_ph axis
    float3 temp_r, temp_th;
    temp_r.x = cph * cam_r.x - sph * cam_th.x;
    temp_r.y = cph * cam_r.y - sph * cam_th.y;
    temp_r.z = cph * cam_r.z - sph * cam_th.z;

    temp_th.x = sph * cam_r.x + cph * cam_th.x;
    temp_th.y = sph * cam_r.y + cph * cam_th.y;
    temp_th.z = sph * cam_r.z + cph * cam_th.z;

    // Rotation around cam_th axis
    cam_r.x = cth * temp_r.x + sth * cam_ph.x;
    cam_r.y = cth * temp_r.y + sth * cam_ph.y;
    cam_r.z = cth * temp_r.z + sth * cam_ph.z;

    cam_ph.x = -sth * temp_r.x + cth * cam_ph.x;
    cam_ph.y = -sth * temp_r.y + cth * cam_ph.y;
    cam_ph.z = -sth * temp_r.z + cth * cam_ph.z;

    // Rotation around cam_r axis
    cam_th.x = cr * temp_th.x - sr * cam_ph.x;
    cam_th.y = cr * temp_th.y - sr * cam_ph.y;
    cam_th.z = cr * temp_th.z - sr * cam_ph.z;

    cam_ph.x = sr * temp_th.x + cr * cam_ph.x;
    cam_ph.y = sr * temp_th.y + cr * cam_ph.y;
    cam_ph.z = sr * temp_th.z + cr * cam_ph.z;
}

__forceinline__ __device__ void pixel_to_direction(
    int i, int j, int W, int H, float tanHalfFov, float aspect,
    const float3 &e_r, const float3 &e_th, const float3 &e_ph,
    const float3 &cam_r, const float3 &cam_th, const float3 &cam_ph,
    float &c_r, float &c_th, float &c_ph)
{
    float height = 2 * tanHalfFov;
    float width = height * aspect;

    // pixel to normalized screen offsets (u: up, v: right)
    float u = (1 - 2 * (i + 0.5) / H) * height;
    float v = (2 * (j + 0.5) / W - 1) * width;

    float3 d = make_float3(
        cam_r.x + u * cam_th.x + v * cam_ph.x,
        cam_r.y + u * cam_th.y + v * cam_ph.y,
        cam_r.z + u * cam_th.z + v * cam_ph.z);

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
        float l, float theta, float phi,
        float th0, float ph0,
        const unsigned char *__restrict__ space1, int width1, int height1,
        const unsigned char *__restrict__ space2, int width2, int height2)
{
    const unsigned char *__restrict__ space = (l > 0.0f) ? space1 : space2;
    const int width = (l > 0.0f) ? width1 : width2;
    const int height = (l > 0.0f) ? height1 : height2;

    if (l < 0.0f)
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
    float l0, float th0, float ph0,
    float b, float L,
    float c_r, float c_th, float c_ph,
    State &s)
{

    float r = r_of_l(l0, b, L);
    float st = fmaxf(sinf(th0), 1e-9f);

    float vl = c_r;
    float vth = c_th / r;
    float vph = c_ph / (r * st);
    float vt = sqrtf(vl * vl + r * r * (vth * vth + (st * st) * vph * vph));

    s = {0.0f, l0, th0, ph0, vt, vl, vth, vph};
}

__forceinline__ __device__
    uint32_t
    trace_geodesic(
        State &s, float dt, float tmax, float b, float L,
        const unsigned char *__restrict__ space1, int w1, int h1,
        const unsigned char *__restrict__ space2, int w2, int h2)
{
    const float th0 = s.th;
    const float ph0 = s.ph;
    const int steps = (int)(tmax / dt);
    for (int i = 0; i < steps; i++)
    {
        rk4_step(s, dt, b, L);
    }
    return map_coordinates_to_pixel(
        s.l, s.th, s.ph, th0, ph0,
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
        const unsigned char *__restrict__ space2, int w2, int h2,
        const float fov,
        const float b,
        const float L,
        const float dt,
        const float tmax,
        const float l0,
        const float th0,
        const float ph0,
        const float r_ang,
        const float th_ang,
        const float ph_ang)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float tanHalf = __tanf(0.5f * fov);
    const float aspect = ((float)W / H);

    float c_r, c_th, c_ph;
    State s;

    float3 e_r, e_th, e_ph, cam_r, cam_th, cam_ph;
    init_world_basis(th0, ph0, e_r, e_th, e_ph);
    init_camera_basis(e_r, e_th, e_ph, cam_r, cam_th, cam_ph, r_ang, th_ang, ph_ang);

    for (int idx = tid; idx < W * H; idx += stride)
    {
        int i = idx / W;
        int j = idx % W;

        pixel_to_direction(i, j, W, H, tanHalf, aspect, e_r, e_th, e_ph, cam_r, cam_th, cam_ph, c_r, c_th, c_ph);
        init_state(l0, th0, ph0, b, L, c_r, c_th, c_ph, s);

        uint32_t pix = trace_geodesic(s, dt, tmax, b, L, space1, w1, h1, space2, w2, h2);

        int p = 3 * idx;
        output[p + 0] = (pix >> 16) & 0xFF;
        output[p + 1] = (pix >> 8) & 0xFF;
        output[p + 2] = (pix) & 0xFF;
    }
}

inline static float clamp_norm(float x, float xmin, float xmax)
{
    if (x < xmin)
    {
        return 0;
    }
    if (x > xmax)
    {
        return 1;
    }
    return (x - xmin) / (xmax - xmin);
}

inline static float lerp(float x, float xmin, float xmax, float ymin, float ymax)
{
    float n = clamp_norm(x, xmin, xmax);
    return ymin + n * (ymax - ymin);
}

inline static float aderp(float x, float xmin, float xmax, float ymin, float ymax)
{
    float n = clamp_norm(x, xmin, xmax);
    float r = (cos((n + 1) * PI) + 1) / 2;
    return ymin + r * (ymax - ymin);
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

    const float fov = 120 * PI / 180;
    const float b = 1;
    const float L = 4;
    const float dt = 1e-3;
    const float tmax = 20.0;

    // const float l0 = 3;
    const float th0 = PI / 2;
    // const float ph0 = 0;

    const float r_ang = 0;
    // const float th_ang = 0;
    const float ph_ang = 0;

    const int fps = 30;
    const int duration = 45;
    const int frames = fps * duration;
    const int hframes = frames / 2;

    for (int i = 0; i < frames; i++)
    {
        float l0 = aderp(i, 0, frames - 1, 3, -3);
        float ph0 = aderp(i, 0, frames - 1, 0, 4 * PI);
        float th_ang = aderp(i, hframes - 8 * fps, hframes + 8 * fps, 0, PI);

        wormhole_kernel<<<grid, block>>>(
            d_out, W, H, d_space1, w1, h1, d_space2, w2, h2,
            fov, b, L, dt, tmax,
            l0, th0, ph0,
            r_ang, th_ang, ph_ang);

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(output, d_out, out_size, cudaMemcpyDeviceToHost));

        std::string filename = "/content/drive/MyDrive/output/wormhole_" + std::to_string(i) + ".png";
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
    unsigned char *space1 = stbi_load("space8.jpg", &w1, &h1, &comp1, 3);
    unsigned char *space2 = stbi_load("space9.jpg", &w2, &h2, &comp2, 3);

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