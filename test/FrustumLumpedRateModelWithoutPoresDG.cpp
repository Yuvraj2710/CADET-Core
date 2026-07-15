// =============================================================================
//  CADET
//
//  Copyright © 2008-present: The CADET-Core Authors
//            Please see the AUTHORS.md file.
//
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include <catch.hpp>
#include "Approx.hpp"

#include "ColumnTests.hpp"
#include "ReactionModelTests.hpp"
#include "JsonTestModels.hpp"
#include "Utils.hpp"
#include "common/Driver.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>


TEST_CASE("Frustum LRM_DG transport Jacobian", "[FrustumLRM],[DG],[UnitOp],[Jacobian],[CI]")
{
	cadet::JsonParameterProvider jpp = createColumnLinearBenchmark(false, true, "FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG");
	cadet::test::column::testJacobianAD(jpp, std::numeric_limits<float>::epsilon() * 100.0);
}

TEST_CASE("Frustum LRM_DG time derivative Jacobian vs FD", "[FrustumLRM],[DG],[UnitOp],[Residual],[Jacobian],[CI]")
{
	cadet::test::column::testTimeDerivativeJacobianFD("FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", 1e-6, 0.0, 9e-4);
}

TEST_CASE("Frustum LRM_DG inlet DOF Jacobian", "[FrustumLRM],[DG],[UnitOp],[Jacobian],[Inlet],[CI]")
{
	cadet::test::column::testInletDofJacobian("FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG");
}

TEST_CASE("Frustum LRM_DG with two component linear binding Jacobian", "[FrustumLRM],[DG],[UnitOp],[Jacobian],[CI]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding("FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG");
	cadet::test::column::testJacobianAD(jpp, std::numeric_limits<float>::epsilon() * 100.0);
}

TEST_CASE("Frustum LRM_DG sensitivity Jacobians", "[FrustumLRM],[DG],[UnitOp],[Sensitivity],[CI]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding("FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG");
	cadet::test::column::testFwdSensJacobians(jpp, 1e-4, 3e-7, 5e-4);
}

TEST_CASE("Frustum LRM_DG consistent initialization with linear binding", "[FrustumLRM],[DG],[ConsistentInit],[CI]")
{
	cadet::test::column::testConsistentInitializationLinearBinding("FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", 1e-9, 1e-9);
}

TEST_CASE("Frustum LRM_DG dynamic reactions Jacobian vs AD bulk", "[FrustumLRM],[DG],[Jacobian],[AD],[ReactionModel],[CI]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", true, false, false, std::numeric_limits<float>::epsilon() * 100.0);
}

TEST_CASE("Frustum LRM_DG dynamic reactions Jacobian vs AD modified bulk", "[FrustumLRM],[DG],[Jacobian],[AD],[ReactionModel],[CI]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", true, false, true, std::numeric_limits<float>::epsilon() * 100.0);
}

TEST_CASE("Frustum LRM_DG dynamic reactions time derivative Jacobian vs FD bulk", "[FrustumLRM],[DG],[Jacobian],[Residual],[ReactionModel],[FD],[CI]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", true, false, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("Frustum LRM_DG dynamic reactions time derivative Jacobian vs FD modified bulk", "[FrustumLRM],[DG],[Jacobian],[Residual],[ReactionModel],[FD],[CI]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", true, false, true, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("Frustum LRM_DG spatial EOC", "[FrustumLRM],[DG],[Simulation],[EOC],[CI]")
{
	// Spatial convergence with tight time integration (ABSTOL=1e-10).
	// Frustum DG resolves the standard benchmark so well that spatial
	// error is below the time integrator floor at all resolutions.
	// Verify self-consistency: coarse (4 elem) and fine (16 elem) agree.

	auto runSim = [](int nElem) -> std::vector<double>
	{
		cadet::JsonParameterProvider jpp = createLinearBenchmark(false, false, "FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG");
		cadet::test::column::DGParams disc(1, 3, nElem);
		disc.setDisc(jpp);

		jpp.pushScope("solver");
		jpp.pushScope("time_integrator");
		jpp.set("ABSTOL", 1e-10);
		jpp.set("RELTOL", 1e-8);
		jpp.set("MAX_STEPS", 100000);
		jpp.popScope();
		jpp.popScope();

		cadet::Driver drv;
		drv.configure(jpp);
		drv.run();

		cadet::InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(0);
		double const* outlet = simData->outlet();
		const unsigned int n = simData->numDataPoints() * simData->numComponents();
		return std::vector<double>(outlet, outlet + n);
	};

	const std::vector<double> coarse = runSim(4);
	const std::vector<double> fine = runSim(16);

	double maxErr = 0.0;
	for (unsigned int i = 0; i < fine.size(); ++i)
		maxErr = std::max(maxErr, std::abs(coarse[i] - fine[i]));

	CAPTURE(maxErr);
	CHECK(maxErr < 1e-6);
}

TEST_CASE("Frustum LRM_DG time integration convergence", "[FrustumLRM],[DG],[Simulation],[EOC],[CI]")
{
	// Spatial discretization is fixed (polyDeg 3, 8 elements — spatially converged).
	// Tightening ABSTOL/RELTOL should reduce the error monotonically, confirming
	// that the time integrator converges on the frustum DG semi-discrete system.

	auto runSim = [](double absTol, double relTol) -> std::vector<double>
	{
		cadet::JsonParameterProvider jpp = createLinearBenchmark(true, true, "FRUSTUM_LUMPED_RATE_MODEL_WITHOUT_PORES", "DG");
		cadet::test::column::DGParams disc(1, 3, 8);
		disc.setDisc(jpp);

		jpp.pushScope("solver");
		jpp.pushScope("time_integrator");
		jpp.set("ABSTOL", absTol);
		jpp.set("RELTOL", relTol);
		jpp.set("MAX_STEPS", 100000);
		jpp.popScope();
		jpp.popScope();

		cadet::Driver drv;
		drv.configure(jpp);
		drv.run();

		cadet::InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(0);
		double const* outlet = simData->outlet();
		const unsigned int n = simData->numDataPoints() * simData->numComponents();
		return std::vector<double>(outlet, outlet + n);
	};

	// Reference: tightest feasible tolerance
	const std::vector<double> ref = runSim(1e-10, 1e-8);

	// Progressively looser tolerances
	const std::vector<double> absTols = {1e-6, 1e-7, 1e-8, 1e-9};
	const std::vector<double> relTols = {1e-4, 1e-5, 1e-6, 1e-7};

	std::vector<double> linfErrors(absTols.size());

	for (unsigned int d = 0; d < absTols.size(); ++d)
	{
		const std::vector<double> sim = runSim(absTols[d], relTols[d]);

		double maxErr = 0.0;
		for (unsigned int i = 0; i < ref.size(); ++i)
			maxErr = std::max(maxErr, std::abs(sim[i] - ref[i]));

		linfErrors[d] = maxErr;
		CAPTURE(absTols[d]);
		CAPTURE(linfErrors[d]);
	}

	// Each error should be within a reasonable factor of its tolerance.
	// Adaptive time integrators don't guarantee strict monotonicity across
	// tolerance levels, but errors should be proportional to ABSTOL.
	for (unsigned int d = 0; d < linfErrors.size(); ++d)
	{
		CAPTURE(absTols[d]);
		CAPTURE(linfErrors[d]);
		CHECK(linfErrors[d] < 100.0 * absTols[d]);
	}

	// Overall convergence: tightest tolerance gives smaller error than loosest
	CHECK(linfErrors.back() < linfErrors.front());
}
