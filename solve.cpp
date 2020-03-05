/*
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 *
 * Modified and  restructured by Scott B. Baden, UCSD
 *
 */

#include <cmath>
#include <utility>
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;
double *alloc1D(int m, int n);

void repNorms(double l2norm, double mx, double dt, int m, int n, int niter, int stats_freq);

// Replace stats() function for sub-block
void stats_submatrix(const double *E, int m, int n, int stride, double *_mx, double *_sumSq) {
    double mx = -1;
    double sumSq = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double x = E[(i + 1) * stride + (1 + j)];
            sumSq += x * x;
            double fe = fabs(x);
            if (fe > mx) {
                mx = fe;
            }
        }
    }
    *_mx = mx;
    *_sumSq = sumSq;
}

extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq) {
    double l2norm = sumSq / (double) ((cb.m) * (cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

// Compute within sub-block
static inline void aliev_panfilov(double *__restrict__ E,
                                  double *__restrict__ E_prev,
                                  double *__restrict__ R,
                                  const double alpha,
                                  const double dt,
                                  const int stride,
                                  const int m,
                                  const int n) {
    const int innerBlockRowStartIndex = stride + 1;
    const int innerBlockRowEndIndex = m * stride + 1;

#ifdef FUSED
    for (int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += stride) {
        double *E_tmp = E + j;
        double *E_prev_tmp = E_prev + j;
        double *R_tmp = R + j;
        for (int i = 0; i < n; i++) {
            E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] +
                E_prev_tmp[i + stride] + E_prev_tmp[i - stride]);
            E_tmp[i] += -dt *
                (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
            R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) *
                (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
        }
    }
#else
    for (int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += stride) {
        double *E_tmp = E + j;
        double *E_prev_tmp = E_prev + j;
        for (int i = 0; i < n; i++) {
            E_tmp[i] = E_prev_tmp[i] + alpha
                * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + stride]
                    + E_prev_tmp[i - stride]);
        }
    }

    for (int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += stride) {
        double *E_tmp = E + j;
        double *R_tmp = R + j;
        double *E_prev_tmp = E_prev + j;
        for (int i = 0; i < n; i++) {
            E_tmp[i] +=
                -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
            R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2))
                * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
        }
    }
#endif
}

static inline void copy_arr(const double *__restrict__ from, double *__restrict__ to, const int stride, const int n) {
    for (int i = 0; i < n; i++) {
        to[i * stride] = from[i * stride];
    }
}

static inline void reorganize(const double *__restrict__ from,
                              double *__restrict__ to,
                              const int stride,
                              const int M,
                              const int N,
                              const int XL,
                              const int YL) {
    int x = 0;
    int y = 0;
    int to_pos = 0;
    while (x < cb.px) {
        while (y < cb.py) {

            int addx = (x - XL) < 0 ? 0 : (x - XL);
            int addy = (y - YL) < 0 ? 0 : (y - YL);
            int m = x < XL ? M : (M - 1);
            int n = y < YL ? N : (N - 1);
            int offset = (x * M + addx * (M - 1) + 1) * stride + (y * N + addy * (N - 1)) + 1;

            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    to[to_pos + i * N + j] = 0;

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    to[to_pos + i * N + j] = from[offset + i * stride + j];

            to_pos += M * N;
            y++;
        }
        x++;
        y = 0;
    }
}

static inline void place_sub_matrix(const double *__restrict__ from, double *__restrict__ to,
                                    const int M, const int N) {
    for (int i = 0; i < M + 2; i++)
        for (int j = 0; j < N + 2; j++)
            to[i * (N + 2) + j] = 0;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            to[(i + 1) * (N + 2) + (j + 1)] = from[i * N + j];
}

#define RANK(x, y) ((x)*cb.py+(y))

#define TAG_TOP 0
#define TAG_BOTTOM (TAG_TOP+1)
#define TAG_LEFT (TAG_TOP+2)
#define TAG_RIGHT (TAG_TOP+3)

void solve(double **_E, double **_E_prev, double *_R, double alpha, double dt, Plotter *plotter, double &L2,
           double &Linf) {
    double *E = *_E, *E_prev = *_E_prev, *R = _R;

    int stride = cb.n + 2;

    int myrank = 0;
#ifdef _MPI_
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif

    // 2D index of sub-block
    const int x = myrank / cb.py;
    const int y = myrank % cb.py;
    // dimension of sub-block generally
    const int M = (cb.m + cb.px - 1) / cb.px;
    const int N = (cb.n + cb.py - 1) / cb.py;
    //larger block boundary
    const int XL = cb.px - (cb.px * M - cb.m);
    const int YL = cb.py - (cb.py * N - cb.n);
    // dimension of this sub-block
    const int m = x < XL ? M : (M - 1);
    const int n = y < YL ? N : (N - 1);

    double *e = alloc1D(M + 2, N + 2);
    double *e_prev = alloc1D(M + 2, N + 2);
    double *r = alloc1D(M + 2, N + 2);
#ifdef _MPI_
    double *E_scatter = alloc1D(cb.px * M, cb.py * N);
    double *R_scatter = alloc1D(cb.px * M, cb.py * N);
    double *E_recv = alloc1D(M, N);
    double *R_recv = alloc1D(M, N);
    if (myrank == 0) {
        reorganize(E_prev, E_scatter, stride, M, N, XL, YL);
        reorganize(R, R_scatter, stride, M, N, XL, YL);
    }

    MPI_Scatter(E_scatter, M * N, MPI_DOUBLE, E_recv, M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(R_scatter, M * N, MPI_DOUBLE, R_recv, M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    place_sub_matrix(E_recv, e_prev, M, N);
    place_sub_matrix(R_recv, r, M, N);
#else
    copy_arr(E_prev, e_prev, 1, (M + 2) * (N + 2));
    copy_arr(R, r, 1, (M + 2) * (N + 2));
#endif

    stride = N + 2;

#ifdef _MPI_
    MPI_Datatype col_type;
    MPI_Type_vector(m, 1, stride, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);
#endif

    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations
    for (int niter = 0; niter < cb.niters; niter++) {
#ifdef _MPI_
        if (!cb.noComm) {
            MPI_Request requests[8] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                       MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

            // Top cells
            double *top_pad = e_prev + 1;
            double *top_row = top_pad + stride;
            if (x == 0) {
                copy_arr(top_row + stride, top_pad, 1, n);
            } else {
                MPI_Isend(top_row, n, MPI_DOUBLE, RANK(x - 1, y), TAG_TOP, MPI_COMM_WORLD, &requests[0]);
                MPI_Irecv(top_pad, n, MPI_DOUBLE, RANK(x - 1, y), TAG_BOTTOM, MPI_COMM_WORLD, &requests[1]);
            }

            // Bottom cells
            double *bottom_pad = e_prev + (m + 1) * stride + 1;
            double *bottom_row = bottom_pad - stride;
            if (x == cb.px - 1) {
                copy_arr(bottom_row - stride, bottom_pad, 1, n);
            } else {
                MPI_Isend(bottom_row, n, MPI_DOUBLE, RANK(x + 1, y), TAG_BOTTOM, MPI_COMM_WORLD, &requests[2]);
                MPI_Irecv(bottom_pad, n, MPI_DOUBLE, RANK(x + 1, y), TAG_TOP, MPI_COMM_WORLD, &requests[3]);
            }

            // Left cells
            double *left_pad = e_prev + stride;
            double *left_col = left_pad + 1;
            if (y == 0) {
                copy_arr(left_col + 1, left_pad, stride, m);
            } else {
                MPI_Isend(left_col, 1, col_type, RANK(x, y - 1), TAG_LEFT, MPI_COMM_WORLD, &requests[4]);
                MPI_Irecv(left_pad, 1, col_type, RANK(x, y - 1), TAG_RIGHT, MPI_COMM_WORLD, &requests[5]);
            }

            // Right cells
            double *right_pad = e_prev + stride + n + 1;
            double *right_col = right_pad - 1;
            if (y == cb.py - 1) {
                copy_arr(right_col - 1, right_pad, stride, m);
            } else {
                MPI_Isend(right_col, 1, col_type, RANK(x, y + 1), TAG_RIGHT, MPI_COMM_WORLD, &requests[6]);
                MPI_Irecv(right_pad, 1, col_type, RANK(x, y + 1), TAG_LEFT, MPI_COMM_WORLD, &requests[7]);
            }

            MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
        }
#else
        // Satisfy boundary conditions
        // Top cells
        double *top_pad = e_prev + 1;
        double *top_row = top_pad + stride;
        copy_arr(top_row + stride, top_pad, 1, n);

        // Bottom cells
        double *bottom_pad = e_prev + (m + 1) * stride + 1;
        double *bottom_row = bottom_pad - stride;
        copy_arr(bottom_row - stride, bottom_pad, 1, n);

        // Left cells
        double *left_pad = e_prev + stride;
        double *left_col = left_pad + 1;
        copy_arr(left_col + 1, left_pad, stride, m);

        // Right cells
        double *right_pad = e_prev + stride + n + 1;
        double *right_col = right_pad - 1;
        copy_arr(right_col - 1, right_pad, stride, m);
#endif

        aliev_panfilov(e, e_prev, r, alpha, dt, stride, m, n);

        if (cb.stats_freq && !(niter % cb.stats_freq)) {
            double mx, sumSq;
            stats_submatrix(e, m, n, stride, &mx, &sumSq);
#ifdef _MPI_
            MPI_Reduce(myrank == 0 ? MPI_IN_PLACE : &mx, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(myrank == 0 ? MPI_IN_PLACE : &sumSq, &sumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, cb.m, cb.n, niter, cb.stats_freq);
        }

        if (cb.plot_freq && !(niter % cb.plot_freq)) {
            plotter->updatePlot(E, niter, cb.m, cb.n);
        }

        // Swap current and previous meshes
        std::swap(e, e_prev);
    } //end of 'niter' loop at the beginning

#ifdef _MPI_
    MPI_Type_free(&col_type);
#endif

    double sumSq;
    stats_submatrix(e_prev, m, n, stride, &Linf, &sumSq);
#ifdef _MPI_
    MPI_Reduce(myrank == 0 ? MPI_IN_PLACE : &Linf, &Linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(myrank == 0 ? MPI_IN_PLACE : &sumSq, &sumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    L2 = L2Norm(sumSq);

    // Swap pointers so we can re-use the arrays
    if (cb.niters % 2) {
        std::swap(*_E, *_E_prev);
    }
}
