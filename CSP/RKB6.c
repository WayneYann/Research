/** File holding CPU 6th-order Runge-Kutta-Butcher explicit integrator.
 * \file RKB6.c
 *
 * \author Kyle E. Niemeyer
 * \date 08/15/2011
 *
 * This file holds the host (CPU) 6th-order Runge-Kutta-Butcher code.
 * 
 */
 
#include "head.h"

void dydt ( Real t, Real * y, Real * Q, int qflag, Real * dy );

static const Real one_third = 1.0 / 3.0;
static const Real two_third = 2.0 / 3.0;
static const Real one_half = 1.0 / 2.0;
static const Real one_twelfth = 1.0 / 12.0;
static const Real one_sixteenth = 1.0 / 16.0;
static const Real one_eighth = 1.0 / 8.0;
static const Real one_fortyfourth = 1.0 / 44.0;
static const Real one_onetwentieth = 1.0 / 120.0;

/** Function performing one step of explicit 6th-order Runge-Kutta-Butcher integrator.
 * Performs explicit 6th-order Runge-Kutta-Butcher integration on CPU. Manually
 * unrolled loops variables. Calls dydt() for derivative.
 *
 * taken from http://mymathlib.webtrellis.net/c_source/diffeq/runge_kutta/runge_kutta_butcher.c
 * 
 * \param[in]   t    the current time, units [s].
 * \param[in]   h    the time step size, units [s]. 
 * \param[in]   y0   the array of data (temperature and species mass fractions) to be integrated, size NN.
 * \param[in]   Q    slow-manifold projector matrix, size NN*NN.
 * \param[out]  y    the array of integrated data, size NN.
 */
void RKB6 ( Real t, Real h, Real * y0, Real * Q, Real * y ) {
  
  Real h3 = one_third * h;
  Real h2_3 = two_third * h;
  Real h2 = one_half * h;
  Real h12 = one_twelfth * h;
  Real h16 = one_sixteenth * h;
  Real h8 = one_eighth * h;
  Real h44 = one_fortyfourth * h;
  Real h120 = one_onetwentieth * h;
  
  // Local arrays holding derivatives
  Real k1[NN], k2[NN], k3[NN], k4[NN], k5[NN], k6[NN], k7[NN];
  
  // Local array holding intermediate y values
  Real ym[NN];
  
  // Calculate k1, the first derivative used.
  dydt ( t, y0, Q, 1, k1 );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the midpoint value used for each variable.
    ym[i] = y0[i] + h3 * k1[i];
  }
  
  // Calculate k2, the second derivative used.
  dydt ( t + h3, ym, Q, 1, k2 );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the next midpoint value used for each variable.
    ym[i] = y0[i] + h2_3 * k2[i];
  }

  // Calculate k3, the third derivative used.
  dydt ( t + h2_3, ym, Q, 1, k3 );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the next midpoint value used for each variable.
    ym[i] = y0[i] + h12 * ( k1[i] + 4.0 * k2[i] - k3[i] );
  }
  
  // Calculate k4, the fourth derivative used.
  dydt ( t + h3, ym, Q, 1, k4 );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the next midpoint value used for each variable.
    ym[i] = y0[i] + h16 * ( -k1[i] + 18.0 * k2[i] - 3.0 * k3[i] - 6.0 * k4[i] );
  }
  
  // calculate k5, fifth derivative used
  dydt ( t + h2, ym, Q, 1, k5 );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the next midpoint value used for each variable.
    ym[i] = y0[i] + h8 * ( 9.0 * k2[i] - 3.0 * k3[i] - 6.0 * k4[i] + 4.0 * k5[i] );
  }
  
  // calculate k6, sixth derivative used
  dydt ( t + h2, ym, Q, 1, k6 );
  
  for ( usint i = 0; i < NN; ++i ) {
    // Calculate the next midpoint value used for each variable.
    ym[i] = y0[i] + h44 * ( 9.0 * k1[i] - 36.0 * k2[i] + 63.0 * k3[i] + 72.0 * k4[i] - 64.0 * k5[i] );
  }
  
  // calculate k7, seventh derivative used
  dydt ( t + h, ym, Q, 1, k7 );
  
  // calculate integrated values
  for ( usint i = 0; i < NN; ++i ) {
    y[i] = y0[i] + h120 * ( 11.0 * ( k1[i] + k7[i] ) + 81.0 * ( k3[i] + k4[i] ) - 32.0 * ( k5[i] + k6[i] ) );
  }
  
} // end RKB6