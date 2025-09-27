#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cimg.h"
#include "pool.h"

using namespace cimg_library;
typedef CImg<unsigned char> Img;

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float INV_TWO_PI = 1.0f / TWO_PI;

typedef struct
{
    float t;
    float r;
    float th;
    float ph;
    float vt;
    float vr;
    float vth;
    float vph;
} State;

// Compute Christoffel symbols
// t=0,r=1,θ=2,φ=3
inline static void christoffels(
    float r,
    float b,
    float theta,
    float &g122,
    float &g133,
    float &g212,
    float &g233,
    float &g323)
{
    float st, ct;
    __sincosf(theta, &st, &ct);

    // Γ^r_{θθ}
    g122 = -r;

    // Γ^r_{φφ}
    g133 = -r * st * st;

    // Γ^θ_{rθ} = Γ^θ_{θr} = Γ^φ_{rφ} = Γ^φ_{φr}
    g212 = r / (b * b + r * r);

    // Γ^θ_{φφ}
    g233 = -st * ct;

    // Γ^φ_{θφ} = Γ^φ_{φθ}
    g323 = ct / (st + 1e-12);
}

// Right-hand side of differential equation
void rhs(const State &state, float b, State &result)
{
    float g122, g133, g212, g233, g323;
    christoffels(state.r, b, state.th, g122, g133, g212, g233, g323);

    float ar = -g122 * state.vth * state.vth - g133 * state.vph * state.vph;
    float ath = -2 * g212 * state.vr * state.vth - g233 * state.vph * state.vph;
    float aph = -2 * g212 * state.vr * state.vph - 2 * g323 * state.vth * state.vph;

    result.t = state.vt;
    result.r = state.vr;
    result.th = state.vth;
    result.ph = state.vph;
    result.vt = 0;
    result.vr = ar;
    result.vth = ath;
    result.vph = aph;
}

// Runge-Kutta 4th order step
void rk4_step(State &s, float dt, float b)
{
    State k1, k2, k3, k4, tmp;

    rhs(s, b, k1);

    tmp.t = s.t + 0.5f * dt * k1.t;
    tmp.r = s.r + 0.5f * dt * k1.r;
    tmp.th = s.th + 0.5f * dt * k1.th;
    tmp.ph = s.ph + 0.5f * dt * k1.ph;
    tmp.vt = s.vt + 0.5f * dt * k1.vt;
    tmp.vr = s.vr + 0.5f * dt * k1.vr;
    tmp.vth = s.vth + 0.5f * dt * k1.vth;
    tmp.vph = s.vph + 0.5f * dt * k1.vph;
    rhs(tmp, b, k2);

    tmp.t = s.t + 0.5f * dt * k2.t;
    tmp.r = s.r + 0.5f * dt * k2.r;
    tmp.th = s.th + 0.5f * dt * k2.th;
    tmp.ph = s.ph + 0.5f * dt * k2.ph;
    tmp.vt = s.vt + 0.5f * dt * k2.vt;
    tmp.vr = s.vr + 0.5f * dt * k2.vr;
    tmp.vth = s.vth + 0.5f * dt * k2.vth;
    tmp.vph = s.vph + 0.5f * dt * k2.vph;
    rhs(tmp, b, k3);

    tmp.t = s.t + dt * k3.t;
    tmp.r = s.r + dt * k3.r;
    tmp.th = s.th + dt * k3.th;
    tmp.ph = s.ph + dt * k3.ph;
    tmp.vt = s.vt + dt * k3.vt;
    tmp.vr = s.vr + dt * k3.vr;
    tmp.vth = s.vth + dt * k3.vth;
    tmp.vph = s.vph + dt * k3.vph;
    rhs(tmp, b, k4);

    const float w = dt / 6.0f;
    s.t = s.t + w * (k1.t + 2.f * k2.t + 2.f * k3.t + k4.t);
    s.r = s.r + w * (k1.r + 2.f * k2.r + 2.f * k3.r + k4.r);
    s.th = s.th + w * (k1.th + 2.f * k2.th + 2.f * k3.th + k4.th);
    s.ph = s.ph + w * (k1.ph + 2.f * k2.ph + 2.f * k3.ph + k4.ph);
    s.vt = s.vt + w * (k1.vt + 2.f * k2.vt + 2.f * k3.vt + k4.vt);
    s.vr = s.vr + w * (k1.vr + 2.f * k2.vr + 2.f * k3.vr + k4.vr);
    s.vth = s.vth + w * (k1.vth + 2.f * k2.vth + 2.f * k3.vth + k4.vth);
    s.vph = s.vph + w * (k1.vph + 2.f * k2.vph + 2.f * k3.vph + k4.vph);
}

inline static int rgb(unsigned char red, unsigned char green, unsigned char blue)
{
    return (red << 16) | (green << 8) | blue;
}

inline static int pixel(const Img &image, const int x, const int y)
{
    unsigned char red = image(x, y, 0, 0);
    unsigned char green = image(x, y, 0, 1);
    unsigned char blue = image(x, y, 0, 2);
    return rgb(red, green, blue);
}

inline static unsigned char red(int rgb)
{
    return (rgb >> 16) & 0xff;
}

inline static unsigned char green(int rgb)
{
    return (rgb >> 8) & 0xff;
}

inline static unsigned char blue(int rgb)
{
    return rgb & 0xff;
}

inline static float clampf(float value, float min, float max)
{
    return fmaxf(min, fminf(value, max));
}

inline float mod2pi(float x)
{
    x -= TWO_PI * floorf(x * INV_TWO_PI);
    return x;
}

int map_coordinates_to_pixel(
    float r, float theta, float phi,
    float th0, float ph0,
    const Img &space1, const Img &space2)
{
    const auto &space = r > 0 ? space1 : space2;
    const int width = space.width();
    const int height = space.height();

    // Flip phi on the other side so we don't face the seam
    if (r < 0)
    {
        phi += ph0 + PI;
    }

    phi = mod2pi(phi);
    theta = clampf(theta, 0.0, PI);

    float x = (phi / TWO_PI) * (width - 1.0);
    float y = (theta / PI) * (height - 1.0);

    int ix = (int)clampf(x, 0, width - 1);
    int iy = (int)clampf(y, 0, height - 1);
    return pixel(space, ix, iy);
}

// Trace geodesic and return a color value
int trace_geodesic(State &state, float dt, int tmax, float b, const Img &space1, const Img &space2)
{
    float th0 = state.th;
    float ph0 = state.ph;
    int steps = (int)(tmax / dt);
    for (int i = 0; i < steps; i++)
    {
        rk4_step(state, dt, b);
    }
    return map_coordinates_to_pixel(state.r, state.th, state.ph, th0, ph0, space1, space2);
}

// Makes the initial state for RK4 integration
void init_state(
    float r0, float th0, float ph0, float b,
    float c_r, float c_th, float c_ph,
    State &state)
{
    float R = sqrt(r0 * r0 + b * b);
    float S = fmaxf(sin(th0), 1e-9);

    // null normalization: choose t' so that k^μ k_μ = 0 → t'^2 = r'^2 + R^2(θ'^2 + sin^2θ φ'^2)
    float vr = c_r;
    float vth = c_th / R;
    float vph = c_ph / (R * S);
    float vt = sqrt(vr * vr + R * R * (vth * vth + (S * S) * vph * vph));

    state.t = 0.0;
    state.r = r0;
    state.th = th0;
    state.ph = ph0;
    state.vt = vt;
    state.vr = vr;
    state.vth = vth;
    state.vph = vph;
}

// Build a normalized view ray in local basis and return its components along e_r, e_th, e_ph
void pixel_to_direction(
    int i, int j, int W, int H, float fov,
    const float e_r[3], const float e_th[3], const float e_ph[3],
    float &c_r, float &c_th, float &c_ph)
{
    float height = 2.0 * tan(fov / 2.0);
    float width = height * ((float)W / H);

    // pixel to normalized screen offsets (u: up, v: right)
    float u = (1 - 2 * (i + 0.5) / H) * height;
    float v = (2 * (j + 0.5) / W - 1) * width;

    // ray in wormhole coordinates
    float d_r = -e_r[0] + u * e_th[0] + v * e_ph[0];
    float d_th = -e_r[1] + u * e_th[1] + v * e_ph[1];
    float d_ph = -e_r[2] + u * e_th[2] + v * e_ph[2];

    // normalize direction vector
    float norm = sqrt(d_r * d_r + d_th * d_th + d_ph * d_ph);
    d_r /= norm;
    d_th /= norm;
    d_ph /= norm;

    // components in the local orthonormal frame
    c_r = d_r * e_r[0] + d_th * e_r[1] + d_ph * e_r[2];
    c_th = d_r * e_th[0] + d_th * e_th[1] + d_ph * e_th[2];
    c_ph = d_r * e_ph[0] + d_th * e_ph[1] + d_ph * e_ph[2];
}

void init_camera_basis(float th0, float ph0, float *e_r, float *e_th, float *e_ph)
{
    float st, ct, sp, cp;
    __sincosf(th0, &st, &ct);
    __sincosf(ph0, &sp, &cp);
    e_r[0] = st * cp;
    e_r[1] = st * sp;
    e_r[2] = ct;
    e_th[0] = ct * cp;
    e_th[1] = ct * sp;
    e_th[2] = -st;
    e_ph[0] = -sp;
    e_ph[1] = cp;
    e_ph[2] = 0.0;
}

void render_row(const Img &space1, const Img &space2, Img &output, int W, int H, int row, int &progress)
{
    const float fov = 120 * PI / 180;
    const float b = 3.0;
    const float dt = 5e-3;
    const float tmax = 20.0;

    const float r0 = 3;
    const float th0 = PI / 2;
    const float ph0 = 0;

    float e_r[3];
    float e_th[3];
    float e_ph[3];

    State state;
    float c_r, c_th, c_ph;

    init_camera_basis(th0, ph0, e_r, e_th, e_ph);

    for (int j = 0; j < W; j++)
    {
        pixel_to_direction(row, j, W, H, fov, e_r, e_th, e_ph, c_r, c_th, c_ph);
        init_state(r0, th0, ph0, b, c_r, c_th, c_ph, state);
        int rgb = trace_geodesic(state, dt, tmax, b, space1, space2);
        output(j, row, 0, 0) = red(rgb);
        output(j, row, 0, 1) = green(rgb);
        output(j, row, 0, 2) = blue(rgb);
    }

    progress++;
    printf("%d/%d\n", progress, H);
}

int main()
{
    const int W = 160, H = 90;
    // const int W = 320, H = 180;
    // const int W = 640, H = 360;
    // const int W = 1280, H = 720;
    // const int W = 1920, H = 1080;
    // const int W = 3840, H = 2160;

    Img space1("images/space5.jpg");
    Img space2("images/space1.jpg");
    Img output(W, H, 1, 3, 0);

    ThreadPool pool(9);
    int progress = 0;
    for (int i = 0; i < H; i++)
    {
        pool.enqueue(&render_row, std::ref(space1), std::ref(space2), std::ref(output), W, H, i, std::ref(progress));
    }
    pool.wait_idle();

    output.save_png("wormhole.png");
}