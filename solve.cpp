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

void repNorms(double l2norm, double mx, double dt, int m, int n, int niter, int stats_freq);

void stats(double *E, int m, int n, double *_mx, double *sumSq);

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

static inline void aliev_panfilov(double *E,
                                  double *E_prev,
                                  double *R,
                                  double alpha,
                                  double dt,
                                  int stride,
                                  int m,
                                  int n) {
    int innerBlockRowStartIndex = stride + 1;
    int innerBlockRowEndIndex = (((m + 2) * stride - 1) - (n)) - stride;
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
}

static inline void copy_arr(const double *from, double *to, int stride, int n) {
    for (int i = 0; i < n; i++) {
        to[i * stride] = from[i * stride];
    }
}

#define RANK(x, y) ((x)*cb.py+(y))

#define TAG_TOP 0
#define TAG_BOTTOM (TAG_TOP+1)
#define TAG_LEFT (TAG_TOP+2)
#define TAG_RIGHT (TAG_TOP+3)

void
solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {
    double *E = *_E, *E_prev = *_E_prev;
    const int stride = cb.n + 2;

    int myrank = 0;
#ifdef _MPI_
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif
    const int x = myrank / cb.py;
    const int y = myrank % cb.py;
    const int M = (cb.m + cb.px - 1) / cb.px;
    const int N = (cb.n + cb.py - 1) / cb.py;
    const int m = x == cb.px - 1 ? cb.m - (cb.px - 1) * M : M;
    const int n = y == cb.py - 1 ? cb.n - (cb.py - 1) * N : N;

    const int offset = x * M * stride + y * N;
    double *e = E + offset;
    double *e_prev = E_prev + offset;
    double *r = R + offset;

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
//            stats(E, cb.m, cb.n, &mx, &sumSq);
            stats_submatrix(e, m, n, stride, &mx, &sumSq);
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
//    stats(E_prev, cb.m, cb.n, &Linf, &sumSq);
    stats_submatrix(e_prev, m, n, stride, &Linf, &sumSq);
    L2 = L2Norm(sumSq);

    // Swap pointers so we can re-use the arrays
    if (cb.niters % 2) {
        std::swap(*_E, *_E_prev);
    }
}
