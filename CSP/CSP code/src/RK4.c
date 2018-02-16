/** File holding CPU 4th-order Runge-Kutta explicit integrator.
 * \file RK4.c
 *
 * \author Kyle E. Niemeyer
 * \date 06/24/2011
 *
 * This file holds the host (CPU) Runge-Kutta code.
 * 
 */

/** Include common libraries and variables */
#include "head.h"

void dydt ( Real t, Real * y, Real * Q, int qflag, Real * dy );

/** Function performing one step of explicit 4th-order Runge-Kutta integrator.
 * Performs explicit 4th-order Runge-Kutta integration on CPU. Manually
 * unrolled loops variables. Calls dydt() for derivative.
 * 
 * \param[in]   t    the current time, units [s].
 * \param[in]   h    the time step size, units [s]. 
 * \param[in]   y0   the array of data (temperature and species mass fractions) to be integrated, size NN.
 * \param[in]   Q    slow-manifold projector matrix, size NN*NN.
 * \param[out]  y    the array of integrated data, size NN.
 */
void RK4 ( Real t, Real h, Real * y0, Real * Q, Real * y ) {
  
  // Variable holding time step divided by 2
  Real h2 = h / 2.0;
  // Variable holding time step divided by 6
  Real onesixh = h / 6.0;
  // Variable holding time step divided by 3
  Real onethirdh = h / 3.0;
  
  // Local array holding derivatives
  Real k[NN];
  // Local array holding intermediate y values
  Real ym[NN];
  
  // Calculate k1, the first derivative used.
  dydt ( t, y0, Q, 1, k );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the midpoint value used for each variable.
    ym[i] = y0[i] + h2 * k[i];
    
    // Add the first contribution to the integrated data.
    y[i] = y0[i] + onesixh * k[i];
  }
  
  // Calculate k2, the second derivative used.
  dydt ( t + h2, ym, Q, 1, k );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the next midpoint value used for each variable.
    ym[i] = y0[i] + h2 * k[i];
    
    // Add the second contribution to the integrated data.
    y[i] += onethirdh * k[i];
  }

  // Calculate k3, the third derivative used.
  dydt ( t + h2, ym, Q, 1, k );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the next midpoint value used for each variable.
    ym[i] = y0[i] + h * k[i];
    
    // Add the third contribution to the integrated data.
    y[i] += onethirdh * k[i];
  }
  
  // Calculate k4, the fourth derivative used.
  dydt ( t + h, ym, Q, 1, k );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Add the final contribution to the integrated data.
    y[i] += onesixh * k[i];
  }
      
} // end RK4