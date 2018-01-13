/** File holding CPU Euler (first order) explicit integrator.
 * \file euler.c
 *
 * \author Kyle E. Niemeyer
 * \date 08/16/2011
 *
 * This file holds the host (CPU) Euler (first order) code.
 * 
 */
 
#include "head.h"

void dydt ( Real t, Real * y, Real * Q, int qflag, Real * dy );

/** Function performing one step of explicit 1st order Euler integrator.
 * Performs explicit 1st order Euler integration on CPU. Manually
 * unrolled loops variables. Calls dydt() for derivative.
 * 
 * \param[in]   t    the current time, units [s].
 * \param[in]   h    the time step size, units [s]. 
 * \param[in]   y0   the array of data (temperature and species mass fractions) to be integrated, size NN.
 * \param[in]   Q    slow-manifold projector matrix, size NN*NN.
 * \param[out]  y    the array of integrated data, size NN.
 */
void euler ( Real t, Real h, Real * y0, Real * Q, Real * y ) {
  
  // Local array holding derivatives
  Real k[NN];
  
  // Calculate the derivative at original point
  dydt ( t, y0, Q, 1, k );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Add the first contribution to the integrated data.
    y[i] = y0[i] + h * k[i];
  } // end i loop
      
} // end euler