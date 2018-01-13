/** File holding CPU 2th-order Runge-Kutta explicit integrator.
 * \file RK2.c
 *
 * \author Kyle E. Niemeyer
 * \date 08/02/2011
 *
 * This file holds the host (CPU) 2nd order Runge-Kutta code.
 * 
 */
 
/** Include common libraries and variables. */
#include "head.h"

// derivative prototype
void dydt ( Real t, Real * y, Real * Q, int qflag, Real * dy );

/** Function performing one step of explicit 2nd-order Runge-Kutta integrator.
 * Performs explicit 2nd-order Runge-Kutta integration on CPU. Manually
 * unrolled loops variables. Calls dydt() for derivative.
 * 
 * \param[in]   t    the current time, units [s].
 * \param[in]   h    the time step size, units [s]. 
 * \param[in]   y0   the array of data (temperature and species mass fractions) to be integrated, size NN.
 * \param[in]   Q    slow-manifold projector matrix, size NN*NN.
 * \param[out]  y    the array of integrated data, size NN.
 */
void RK2 ( Real t, Real h, Real * y0, Real * Q, Real * y ) {
  
  // Variable holding time step divided by 2
  Real h2 = h / TWO;
  
  // Local array holding derivatives
  Real k[NN];
  
  // Local array holding intermediate y values
  Real ym[NN];
  
  // Calculate k1, the first derivative used.
  dydt ( t, y0, Q, 1, k );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the midpoint value used for each variable.
    ym[i] = y0[i] + h2 * k[i];
  }
  
  // Calculate k2, the second derivative used.
  dydt ( t + h2, ym, Q, 1, k );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the integrated data.
    y[i] = y0[i] + h * k[i];
  }
      
} // end RK2