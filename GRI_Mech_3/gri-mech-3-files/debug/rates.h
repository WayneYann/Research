#ifndef RATES_HEAD
#define RATES_HEAD

#include "header.h"

void eval_rxn_rates (const double, const double, const double*, double*, double*);
void eval_spec_rates (const double*, const double*, const double*, double*, double*);
void get_rxn_pres_mod (const double, const double, const double*, double*);

#endif
