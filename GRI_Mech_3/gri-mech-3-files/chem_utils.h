#ifndef CHEM_UTILS_HEAD
#define CHEM_UTILS_HEAD

#include "header.h"

void eval_conc (const double, const double, const double*, double*, double*, double*, double*);
void eval_conc_rho (const double, const double, const double*, double*, double*, double*, double*);
void eval_h (const double, double*);
void eval_u (const double, double*);
void eval_cv (const double, double*);
void eval_cp (const double, double*);

#endif
