/*
 * ECOS - Embedded Conic Solver.
 * Copyright (C) 2012-14 Alexander Domahidi [domahidi@control.ee.ethz.ch],
 * Automatic Control Laboratory, ETH Zurich.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * aidxint with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* ECOS TESTER MODULE */
/* THE CODE FOR MINIMAL UNIT TESTING HAS BEEN TAKEN FROM http://www.jera.com/techinfo/jtns/jtn002.html */
#include <stdio.h>

#include "minunit.h"
#include "ecos.h"

/* Include Tests */
#include "MPC/MPC01.h"
#include "MPC/MPC02.h"
#include "updateData/update_data.h"
#include "cvxpyProblems/githubIssue98.h"
#include "feasibilityProblems/feas.h"
#include "unboundedProblems/unboundedLP1.h"
#include "infeasibleProblems/infeasible1.h"
#include "unboundedProblems/unboundedMaxSqrt.h"
#include "emptyProblem/emptyProblem.h"
#include "LPnetlib/lp_25fv47.h"
#include "LPnetlib/lp_adlittle.h"
#include "LPnetlib/lp_afiro.h"
#include "LPnetlib/lp_agg.h"
#include "LPnetlib/lp_agg2.h"
#include "LPnetlib/lp_agg3.h"
#include "LPnetlib/lp_bandm.h"
#include "LPnetlib/lp_beaconfd.h"
#include "LPnetlib/lp_blend.h"
#include "LPnetlib/lp_bnl1.h"

int tests_run = 0;

#include "timing.hpp"

static char *all_tests()
{
    mu_run_test(test_MPC01);
    mu_run_test(test_MPC02);
    mu_run_test(test_update_data);
    mu_run_test(test_unboundedLP1);
    mu_run_test(test_unboundedMaxSqrt);
    mu_run_test(test_feas);
    mu_run_test(test_infeasible1);
    mu_run_test(test_lp_25fv47);
    mu_run_test(test_lp_adlittle);
    mu_run_test(test_lp_afiro);
    mu_run_test(test_lp_agg);
    mu_run_test(test_lp_agg2);
    mu_run_test(test_lp_agg3);
    mu_run_test(test_lp_bandm);
    mu_run_test(test_lp_beaconfd);
    mu_run_test(test_lp_blend);
    mu_run_test(test_lp_bnl1);
    mu_run_test(test_emptyProblem);
    mu_run_test(test_issue98);

    return 0;
}

int main(void)
{
    double t0 = tic();
    char *result = all_tests();
    if (result != 0)
    {
        printf("%s\n", result);
    }
    else
    {
        printf("\nALL TESTS PASSED\n");
    }
    printf("Tests run: %d\n", tests_run);
    printf("Test time: %f\n", toc(t0));

    return result != 0;
}
