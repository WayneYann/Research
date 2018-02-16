/** Main function file for CSP model problem project.
 * \file main.c
 *
 * \author Kyle E. Niemeyer
 * \date 08/02/2011
 *
 * Contains main and integration driver functions.
 */

/** Include common code. */
#include "head.h"

/** Stiffness factor eps. */
const Real eps = 1.0e-2;

/** Time step factor mu.
 * Time step divided by controlling time scale.
 */
const Real mu = 0.005;

/** Pointer to output file */
FILE *file;

// Function prototypes
uint get_slow_projector ( Real tim, Real * y, Real * Qs, Real * taum1, Real * Rc );
void radical_correction ( Real tim, Real * y, Real * Rc, Real * g );
void RK4 ( Real t, Real h, Real * y0, Real * Q, Real * y );
void RK2 ( Real t, Real h, Real * y0, Real * Q, Real * y );
void euler ( Real t, Real h, Real * y0, Real * Q, Real * y );
void RKB6 ( Real t, Real h, Real * y0, Real * Q, Real * y );

////////////////////////////////////////////////////////////////////////

/** Integration driver function.
 *
 * \param[in,out] tim       in: starting time (s), out: updated time (s)
 * \param[in]     dt        desired time step (s)
 * \param[in,out] y_global  array of values at current time, size NUM*NN
 */
void intDriver ( Real * tim, Real dt, Real * y_global ) {

  // Loop over all "threads"
  // Emulates GPU kernel, but using a loop over each "thread" (cell)
  for ( uint tid = 0; tid < NUM; ++tid ) {

    // get value from time pointer
    Real t0 = *tim;

    // flag for printing
    int pflag = 0;

    // local array holding initial values
    Real y0_local[NN];
    // local array to hold integrated values
    Real yn_local[NN];

    // load local array y0_local with initial values from global array y_global
    for ( uint i = 0; i < NN; ++i ) {
      y0_local[i] = y_global[tid + NUM*i];
    }

    // time substepping
    while ( *tim < ( t0 + dt ) ) {

      // if using CSP projection method
      #if !defined(EXPLICIT)
      // CSP slow-manifold projector matrix
      Real Qs[NN*NN];

      // CSP radical correction tensor matrix
      Real Rc[NN*NN];

      // time scale of fastest slow mode (controlling time scale)
      Real tau;
      Real * ptau = &tau; // pointer to tau

      // get slow-manifold projector, driving time scale, and radical correction
      uint M = get_slow_projector ( *tim, y0_local, Qs, ptau, Rc );

      // time step size (controlled by fastest slow mode)
      Real h = mu * tau;

      fprintf ( file, "%17.10le %d %17.10le %17.10le %17.10le %17.10le\n", *tim, M, y0_local[0], y0_local[1], y0_local[2], y0_local[3]);

      // if using explicit integration only
      #else

      // pointer to empty projection matrix
      Real * Qs;

      // constant time step
      Real h = 1.0e-8;

      /*if ( pflag == 0 ) {
        fprintf ( file, "%17.10le %17.10le %17.10le %17.10le %17.10le\n", *tim, y0_local[0], y0_local[1], y0_local[2], y0_local[3]);
        pflag = 1;
      }*/

      #endif

      //
      // integrate one time step
      //

      // 4th-order Runge-Kutta
      RK4 ( *tim, h, y0_local, Qs, yn_local );

      // 2nd-order Runge-Kutta (midpoint)
      //RK2 ( *tim, h, y0_local, Qs, yn_local );

      // 1st-order Euler
      //euler ( *tim, h, y0_local, Qs, yn_local );

      // 6th-order Runge-Kutta-Butcher
      //RKB6 ( *tim, h, y0_local, Qs, yn_local );

      // update time
      *tim += h;

      #if !defined(EXPLICIT)
      // local array holding radical corrections
      Real rc_array[NN];

      // apply radical correction
      radical_correction ( *tim, yn_local, Rc, rc_array );

      for ( uint i = 0; i < NN; ++i ) {
        y0_local[i] = yn_local[i] - rc_array[i];

        // check if any > 1 or negative
        if ( y0_local[i] > ONE || y0_local[i] < ZERO ) {
          fprintf ( file, "%17.10le %d %17.10le %17.10le %17.10le %17.10le\n", *tim, M, y0_local[0], y0_local[1], y0_local[2], y0_local[3]);
          //exit(1);
        }
      }
      #else
      // no radical correction
      for ( uint i = 0; i < NN; ++i ) {
        y0_local[i] = yn_local[i];
      }
      #endif

    }

    // update global array y_global with integrated values
    for ( uint i = 0; i < NN; ++i ) {
      y_global[tid + NUM*i] = y0_local[i];
    }

  } // end tid loop

} // end intDriver

////////////////////////////////////////////////////////////////////////

/** Main function, launches integration driver.
 *
 * Loads initial conditions, then starts time integration loop,
 * calling integration driver for each time step. Increases time using time step
 * returned from CSP in integration function.
 *
 */
int main ( void ) {

  // t0 is the starting time (sec)
  Real t0 = 0.0;

  // tend is the ending integration time (sec)
  Real tend = 5.0;

  // Time variable
  Real tim;
  // Pointer for time
  Real * ptim = &tim;


  // size stores size of data array in bytes
  uint size = NUM * sizeof(Real) * NN;

  // Array holding data
  Real *Y = (Real *) malloc (size);

  // set initial conditions
  tim = t0;
  for ( int i = 0; i < NN*NUM; ++i ) {
    Y[i] = 1.0;
  }

  // printing time step
  Real dt = 1.0e-7;

  // open file for writing
  file = fopen("output.txt","w");


  // Start timer
  clock_t t_start = clock();

  // start time integration loop
  while ( tim < tend ) {

    if ( (dt < 1.0e-6) && (tim >= 1.0e-6) ) {
      dt = 1.0e-6;
    } else if ( (dt < 1.0e-5) && (tim >= 1.0e-5) ) {
      dt = 1.0e-5;
    } else if ( (dt < 1.0e-4) && (tim >= 1.0e-4) ) {
      dt = 1.0e-4;
    } else if ( (dt < 1.0e-3) && (tim >= 1.0e-3) ) {
      dt = 1.0e-3;
    } else if ( (dt < 1.0e-2) && (tim >= 1.0e-2) ) {
      dt = 1.0e-2;
    }

    // call integration driver, which updates Y and tim
    intDriver ( ptim, dt, Y );

    #ifdef EXPLICIT
    fprintf ( file, "%17.10le %17.10le %17.10le %17.10le %17.10le\n", tim, Y[0], Y[1], Y[2], Y[3] );
    #endif
  } // end while loop

  // stop the timer
  clock_t t_end = clock();

  // close output file
  fclose ( file );

  // Get the clock time in seconds
  Real cpu_tim = ( t_end - t_start ) / ( (Real)(CLOCKS_PER_SEC) );
  // Print time per step, and time per step per thread.
  printf("CPU time: %e (s)\n", cpu_tim);

  return 0;
}
