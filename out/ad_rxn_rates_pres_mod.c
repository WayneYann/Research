#include <math.h>
#include "header.h"
#include "ad_rates.h"
#include "adept.h"
using adept::adouble;
#define fmax(a, b) (a.value() > b ? a : adouble(b))

void get_rxn_pres_mod (const adouble T, const adouble& pres, const adouble * __restrict__ C, adouble * __restrict__ pres_mod) {
  // third body variable declaration
  adouble thd;

  // pressure dependence variable declarations
  adouble k0;
  adouble kinf;
  adouble Pr;

  // troe variable declarations
  adouble logFcent;
  adouble A;
  adouble B;

  adouble logT = log(T);
  adouble m = pres / (8.31446210e+03 * T);

  // reaction 5;
  pres_mod[0] = m + 1.5 * C[1] + 2.8 * C[11] - 1.0 * C[8] + 11.0 * C[4] - 1.0 * C[9] + 0.8999999999999999 * C[10];

  // reaction 8;
  pres_mod[1] = m + 1.5 * C[1] + 2.8 * C[11] - 1.0 * C[8] + 11.0 * C[4] - 1.0 * C[9] + 0.8999999999999999 * C[10];

  // reaction 11;
  pres_mod[2] = m + 1.5 * C[1] + 2.8 * C[11] - 0.25 * C[8] + 11.0 * C[4] - 0.25 * C[9] + 0.8999999999999999 * C[10];

  // reaction 12;
  pres_mod[3] = m + 1.0 * C[12] + 2.0 * C[1] + 2.8 * C[11] - 1.0 * C[4] + 0.10000000000000009 * C[9] + 0.8999999999999999 * C[10] + 0.5 * C[5];

  // reaction 14;
  thd = m + 1.0 * C[1] + 2.8 * C[11] - 0.32999999999999996 * C[8] + 13.0 * C[4] - 0.19999999999999996 * C[9] + 0.8999999999999999 * C[10] - 0.21999999999999997 * C[5];
  k0 = exp(3.4087162630776540e+01 - 1.72 * logT - (2.6408962763808853e+02 / T));
  kinf = exp(2.2260313685392592e+01 + 0.44 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.00000000e-01 * exp(-T / 1.00000000e-30) + 5.00000000e-01 * exp(-T / 1.00000000e+30), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[4] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 21;
  thd = m + 0.5 * C[12] + 2.7 * C[1] + 0.6000000000000001 * C[11] + 6.5 * C[4] - 0.35 * C[9] + 6.7 * C[7] + 1.7999999999999998 * C[10] + 0.19999999999999996 * C[5];
  k0 = exp(4.9266569663351575e+01 - 2.3 * logT - (2.4531450567319320e+04 / T));
  kinf = exp(2.8324168296488494e+01 + 0.9 * logT - (2.4531450567319320e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.70000000e-01 * exp(-T / 1.00000000e-30) + 4.30000000e-01 * exp(-T / 1.00000000e+30), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[5] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

} // end get_rxn_pres_mod

