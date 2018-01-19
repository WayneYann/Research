// Runge-Kutta-Chebyshev integrator
// Kyle Niemeyer
// 6/24/11
//
// based on Sommeijer, Shampine, Verwer RKC.f (http://www.netlib.org/ode/rkc.f)

#include "head.h"


void RKC (Real t, Real* Y0, Real h, Real* Yj) {
  // performs one step using RKC

  const usint s = 5;

  const Real w0 = 1.0 + (2.0 / (13.0 * s * s));
  const Real temp1 = (w0 * w0) - 1.0;

  const Real temp2 = sqrt(temp1);
  const Real arg = s * log(w0 + temp2);

  const Real sinharg = sinh(arg);
  const Real w1 = sinharg * temp1 / ( cosh(arg) * s * temp2 - w0 * sinharg );


  Real bj = 1.0 / (4.0 * w0 * w0);
  Real bjm1 = bj, bjm2 = bj;
  Real mu1t = bjm1 * w1;
  Real muj = 2.0 * w0;
  Real nuj = -1.0;
  Real mujt = 2.0 * w1;
  Real ajm1;

  Real F0[NN], Y1[NN], Yjm2[NN], Yjm1[NN];

  dydt (t, Y0, F0 );

  for (usint i = 0; i < NN; ++i) {
    Y1[i] = Y0[i] + ( mu1t * h * F0[i] );

    // set Yjm2 and Yjm1
    Yjm2[i] = Y0[i];
    Yjm1[i] = Y1[i];
  }

  // use Yj to store derivative array
  dydt ( t + mujt * h, Y1, Yj );

  bjm2 = bjm1;
  bjm1 = bj;

  // recurrence relations for Chebyshev polynomials
  Real Tj, dTj, d2Tj;
  Real Tjm1 = w0;
  Real Tjm2 = 1.0;
  Real dTjm1 = 1.0;
  Real dTjm2 = 0.0;
  Real d2Tjm1 = 0.0;
  Real d2Tjm2 = 0.0;
  Real cj;
  Real cjm2 = 0.0;
  Real cjm1 = mujt;

  for (usint j = 2; j <= s; ++j)
  {
    Tj = 2.0 * w0 * Tjm1 - Tjm2;
    dTj = 2.0 * w0 * dTjm1 - dTjm2 + 2.0 * Tjm1;
    d2Tj = 2.0 * w0 * d2Tjm1 - d2Tjm2 + 4.0 * dTjm1;
    bj = d2Tj / (dTj * dTj);
    ajm1 = 1.0 - bjm1 * Tjm1;
    muj = 2.0 * w0 * bj / bjm1;
    nuj = - bj / bjm2;
    mujt = muj * w1 / w0;

    // use Yj to store derivative array
    dydt ( t + cjm1 * h, Yjm1, Yj );

    for (usint i = 0; i < NN; ++i) {
      Yj[i] = (1.0 - muj - nuj) * Y0[i] + (muj * Yjm1[i]) + (nuj * Yjm2[i]) + h * mujt * (Yj[i] - ajm1 * F0[i]);

      Yjm2[i] = Yjm1[i];
      Yjm1[i] = Yj[i];
    }
    cj = muj * cjm1 + nuj * cjm2 + mujt * (1.0 - ajm1);

    Tjm2 = Tjm1;
    Tjm1 = Tj;
    dTjm2 = dTjm1;
    dTjm1 = dTj;
    d2Tjm2 = d2Tjm1;
    d2Tjm1 = d2Tj;
    bjm2 = bjm1;
    bjm1 = bj;
    cjm2 = cjm1;
    cjm1 = cj;

  } // end loop over j

  // the final Yj is returned

}
