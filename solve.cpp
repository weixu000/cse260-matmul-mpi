/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <math.h>
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"

void repNorms(double l2norm, double mx, double dt, int m, int n, int niter, int stats_freq);

void stats(double *E, int m, int n, double *_mx, double *sumSq);

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

void
solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {
    double *E = *_E, *E_prev = *_E_prev;
    int m = cb.m, n = cb.n;

    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations
    for (int niter = 0; niter < cb.niters; niter++) {
        /*
         * Copy data from boundary of the computational box to the
         * padding region, set up for differencing computational box's boundary
         *
         * These are physical boundary conditions, and are not to be confused
         * with ghost cells that we would use in an MPI implementation
         *
         * The reason why we copy boundary conditions is to avoid
         * computing single sided differences at the boundaries
         * which increase the running time of solve()
         *
         */
        // Fills in the TOP Ghost Cells
        for (int i = 0; i < (n + 2); i++) {
            E_prev[i] = E_prev[i + (n + 2) * 2];
        }

        // Fills in the RIGHT Ghost Cells
        for (int i = (n + 1); i < (m + 2) * (n + 2); i += (n + 2)) {
            E_prev[i] = E_prev[i - 2];
        }

        // Fills in the LEFT Ghost Cells
        for (int i = 0; i < (m + 2) * (n + 2); i += (n + 2)) {
            E_prev[i] = E_prev[i + 2];
        }

        // Fills in the BOTTOM Ghost Cells
        for (int i = ((m + 2) * (n + 2) - (n + 2)); i < (m + 2) * (n + 2); i++) {
            E_prev[i] = E_prev[i - (n + 2) * 2];
        }

        // Solve for the excitation
        int innerBlockRowStartIndex = (n + 2) + 1;
        int innerBlockRowEndIndex = (((m + 2) * (n + 2) - 1) - (n)) - (n + 2);
        for (int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2)) {
            double *E_tmp = E + j;
            double *E_prev_tmp = E_prev + j;
            double *R_tmp = R + j;
            for (int i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] +
                                                    E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
                E_tmp[i] += -dt *
                            (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) *
                            (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
            }
        }

        if (cb.stats_freq && !(niter % cb.stats_freq)) {
            double mx, sumSq;
            stats(E, m, n, &mx, &sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, m, n, niter, cb.stats_freq);
        }

        if (cb.plot_freq && !(niter % cb.plot_freq)) {
            plotter->updatePlot(E, niter, m, n);
        }

        // Swap current and previous meshes
        double *tmp = E;
        E = E_prev;
        E_prev = tmp;
    } //end of 'niter' loop at the beginning

    double sumSq;
    stats(E_prev, m, n, &Linf, &sumSq);
    L2 = L2Norm(sumSq);

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}
