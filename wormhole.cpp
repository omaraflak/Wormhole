#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cimg.h"

using namespace cimg_library;

// Structure to hold non-zero Christoffel symbols
typedef struct
{
    // t=0,r=1,θ=2,φ=3
    double gamma_1_2_2;
    double gamma_1_3_3;
    double gamma_2_1_2;
    double gamma_2_3_3;
    double gamma_3_2_3;
} Christoffels;

// Compute Christoffel symbols
void christoffels(double r, double b, double theta, Christoffels *gamma)
{
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    // Γ^r_{θθ} = -r
    gamma->gamma_1_2_2 = -r;

    // Γ^r_{φφ} = -r sin^2(θ)
    gamma->gamma_1_3_3 = -r * sin_theta * sin_theta;

    // Γ^θ_{rθ} = Γ^θ_{θr} = Γ^φ_{rφ} = Γ^φ_{φr} = r/(b^2+r^2)
    gamma->gamma_2_1_2 = r / (b * b + r * r);

    // Γ^θ_{φφ} = -sinθcosθ
    gamma->gamma_2_3_3 = -sin_theta * cos_theta;

    // Γ^φ_{θφ} = Γ^φ_{φθ} = cotθ
    gamma->gamma_3_2_3 = cos_theta / (sin_theta + 1e-12);
}

// Right-hand side of differential equation
void rhs(const double *state, double b, double *result)
{
    // State = [t, r, θ, φ, t', r', θ', φ']
    double t = state[0], r = state[1], th = state[2], ph = state[3];
    double vt = state[4], vr = state[5], vth = state[6], vph = state[7];
    double v[4] = {vt, vr, vth, vph};
    double a[4] = {0.0, 0.0, 0.0, 0.0};

    Christoffels gamma;
    christoffels(r, b, th, &gamma);

    // Γ^r_{θθ}
    a[1] -= gamma.gamma_1_2_2 * v[2] * v[2];

    // Γ^r_{φφ}
    a[1] -= gamma.gamma_1_3_3 * v[3] * v[3];

    // Γ^θ_{rθ} = Γ^θ_{θr} = Γ^φ_{rφ} = Γ^φ_{φr}
    a[2] -= gamma.gamma_2_1_2 * v[1] * v[2];
    a[2] -= gamma.gamma_2_1_2 * v[2] * v[1];
    a[3] -= gamma.gamma_2_1_2 * v[1] * v[3];
    a[3] -= gamma.gamma_2_1_2 * v[3] * v[1];

    // Γ^θ_{φφ}
    a[2] -= gamma.gamma_2_3_3 * v[3] * v[3];

    // Γ^φ_{θφ} = Γ^φ_{φθ}
    a[3] -= gamma.gamma_3_2_3 * v[2] * v[3];
    a[3] -= gamma.gamma_3_2_3 * v[3] * v[2];

    result[0] = vt;
    result[1] = vr;
    result[2] = vth;
    result[3] = vph;
    result[4] = a[0];
    result[5] = a[1];
    result[6] = a[2];
    result[7] = a[3];
}

// Runge-Kutta 4th order step
void rk4_step(double *state, double dt, double b)
{
    double k1[8], k2[8], k3[8], k4[8];
    double temp_state[8];

    // k1 = rhs(state, b)
    rhs(state, b, k1);

    // temp_state = state + 0.5 * dt * k1
    for (int i = 0; i < 8; i++)
    {
        temp_state[i] = state[i] + 0.5 * dt * k1[i];
    }
    rhs(temp_state, b, k2);

    // temp_state = state + 0.5 * dt * k2
    for (int i = 0; i < 8; i++)
    {
        temp_state[i] = state[i] + 0.5 * dt * k2[i];
    }
    rhs(temp_state, b, k3);

    // temp_state = state + dt * k3
    for (int i = 0; i < 8; i++)
    {
        temp_state[i] = state[i] + dt * k3[i];
    }
    rhs(temp_state, b, k4);

    // state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    for (int i = 0; i < 8; i++)
    {
        state[i] += dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0;
    }
}

int rgb(unsigned char red, unsigned char green, unsigned char blue)
{
    return (red << 16) | (green << 8) | blue;
}

int pixel(CImg<unsigned char> *image, const int x, const int y)
{
    unsigned char red = (*image)(x, y, 0, 0);
    unsigned char green = (*image)(x, y, 0, 1);
    unsigned char blue = (*image)(x, y, 0, 2);
    return rgb(red, green, blue);
}

unsigned char red(int rgb)
{
    return (rgb >> 16) & 0xff;
}

unsigned char green(int rgb)
{
    return (rgb >> 8) & 0xff;
}

unsigned char blue(int rgb)
{
    return rgb & 0xff;
}

double clamp(double value, double min, double max)
{
    return fmax(min, fmin(value, max));
}

int map_coordinates_to_pixel(double r, double theta, double phi, CImg<unsigned char> *space1, CImg<unsigned char> *space2)
{
    const auto &space = r > 0 ? space1 : space2;
    const int width = space->width();
    const int height = space->height();

    phi = fmod(phi, 2 * M_PI);
    if (phi < 0)
    {
        phi += 2.0 * M_PI;
    }

    theta = clamp(theta, 0.0, M_PI);

    double x = (phi / (2.0 * M_PI)) * (width - 1.0);
    double y = (theta / M_PI) * (height - 1.0);

    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return pixel(space, x, y);
}

// Trace geodesic and return a color value
int trace_geodesic(
    double *state, double dt, int tmax, double b,
    CImg<unsigned char> *space1, CImg<unsigned char> *space2)
{
    int steps = (int)(tmax / dt);
    for (int i = 0; i < steps; i++)
    {
        rk4_step(state, dt, b);
    }
    return map_coordinates_to_pixel(state[1], state[2], state[3], space1, space2);
}

// Makes the initial state for RK4 integration
void init_state(
    double r0, double th0, double ph0, double b,
    double c_r, double c_th, double c_ph,
    double *state)
{
    double R = sqrt(r0 * r0 + b * b);
    double S = sin(th0);
    if (S < 1e-9)
        S = 1e-9;

    // null normalization: choose t' so that k^μ k_μ = 0 → t'^2 = r'^2 + R^2(θ'^2 + sin^2θ φ'^2)
    double vr = c_r;
    double vth = c_th / R;
    double vph = c_ph / (R * S);
    double vt = sqrt(vr * vr + R * R * (vth * vth + (S * S) * vph * vph));

    state[0] = 0.0;
    state[1] = r0;
    state[2] = th0;
    state[3] = ph0;
    state[4] = vt;
    state[5] = vr;
    state[6] = vth;
    state[7] = vph;
}

// Build a normalized view ray in local basis and return its components along e_r, e_th, e_ph
void pixel_to_local(
    int i, int j, int W, int H, double fov,
    const double e_r[3], const double e_th[3], const double e_ph[3],
    double *c_r, double *c_th, double *c_ph)
{
    // pixel → normalized screen offsets (u: right, v: up)
    double u = ((j + 0.5) / W - 0.5) * 2.0 * tan(fov / 2.0);
    double v = (0.5 - (i + 0.5) / H) * 2.0 * tan(fov / 2.0) * ((double)H / W);

    // pinhole ray in world coords
    double d[3] = {-e_r[0] + u * e_ph[0] + v * e_th[0],
                   -e_r[1] + u * e_ph[1] + v * e_th[1],
                   -e_r[2] + u * e_ph[2] + v * e_th[2]};

    // normalize
    double L = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    d[0] /= L;
    d[1] /= L;
    d[2] /= L;

    // components in the local orthonormal frame
    *c_r = d[0] * e_r[0] + d[1] * e_r[1] + d[2] * e_r[2];
    *c_th = d[0] * e_th[0] + d[1] * e_th[1] + d[2] * e_th[2];
    *c_ph = d[0] * e_ph[0] + d[1] * e_ph[1] + d[2] * e_ph[2];
}

void init_camera_basis(double th0, double ph0, double *e_r, double *e_th, double *e_ph)
{
    double st = sin(th0), ct = cos(th0);
    double sp = sin(ph0), cp = cos(ph0);
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

int main()
{
    const int W = 600;
    const int H = 300;
    const double fov = 60 * M_PI / 180;
    const double b = 1.0;
    const double dt = 1e-2;
    const double tmax = 30.0;

    const double r0 = 10;
    const double th0 = M_PI / 2;
    const double ph0 = 0;

    double e_r[3];
    double e_th[3];
    double e_ph[3];

    double state[8];
    double c_r, c_th, c_ph;

    CImg<unsigned char> space1("space1.jpg");
    CImg<unsigned char> space2("space2.jpg");
    CImg<unsigned char> image(W, H, 1, 3, 0);

    init_camera_basis(th0, ph0, e_r, e_th, e_ph);

    for (int i = 0; i < H; i++)
    {
        printf("%d/%d\n", i + 1, H);
        for (int j = 0; j < W; j++)
        {
            pixel_to_local(i, j, W, H, fov, e_r, e_th, e_ph, &c_r, &c_th, &c_ph);
            init_state(r0, th0, ph0, b, c_r, c_th, c_ph, state);
            int rgb = trace_geodesic(state, dt, tmax, b, &space1, &space2);
            image(j, i, 0, 0) = red(rgb);
            image(j, i, 0, 1) = green(rgb);
            image(j, i, 0, 2) = blue(rgb);
        }
    }

    image.save_png("wormhole.png");
}