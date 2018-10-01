@testset "PODNonlinearModel loading tests" begin
    # Random Model 1
    test_solver = PODSolver(nlp_solver=IpoptSolver(),mip_solver=CbcSolver(loglevel=0),log_level=0)
    m = operator_c(solver=test_solver)

    status = JuMP.build(m)
    @test isa(m.internalModel, POD.PODNonlinearModel)

    # Expression Model 1
    test_solver = PODSolver(nlp_solver=IpoptSolver(),mip_solver=CbcSolver(loglevel=0),log_level=0)
    m = exprstest(solver=test_solver)
    status = JuMP.build(m)
    @test isa(m.internalModel, POD.PODNonlinearModel)
end

@testset "Partitioning variable selection tests :: nlp3" begin

    # all variables selection rule
    test_solver = PODSolver(nlp_solver = IpoptSolver(print_level=0),
                            mip_solver = CbcSolver(loglevel=0),
                            disc_var_pick = 0,
                            disc_uniform_rate = 10,
                            presolve_bp = false,
                            presolve_bt = false,
                            max_iter = 1,
                            log_level = 1)
    m = nlp3(solver=test_solver)
    status = solve(m)

    @test status == :UserLimits
    @test isapprox(m.objVal, 7049.2478976; atol=1e-3)
    @test length(m.internalModel.candidate_disc_vars) == 8
    @test length(m.internalModel.disc_vars) == 8
    @test m.internalModel.disc_var_pick == 0
    @test getsolvetime(m) > 0.

    # 15 variable selection rule 
    test_solver = PODSolver(nlp_solver = IpoptSolver(print_level=0),
                            mip_solver = CbcSolver(loglevel=0),
                            disc_var_pick = 2,
                            disc_uniform_rate = 10,
                            presolve_bp = false,
                            presolve_bt = true,
                            max_iter = 1,
                            log_level = 0)
    m = nlp3(solver=test_solver)
    status = solve(m)

    @test status == :UserLimits
    @test isapprox(m.objVal, 7049.2478976; atol=1e-3)
    @test length(m.internalModel.candidate_disc_vars) == 8
    @test length(m.internalModel.disc_vars) == 8
    @test m.internalModel.disc_var_pick == 2

    # min. vertex cover variable selection rule
    test_solver = PODSolver(nlp_solver = IpoptSolver(print_level=0),
                            mip_solver = CbcSolver(loglevel=0),
                            disc_var_pick = 1,
                            disc_uniform_rate=10,
                            presolve_bp = true,
                            presolve_bt = false,
                            max_iter = 1,
                            log_level = 2)
    m = nlp3(solver=test_solver)
    status = solve(m)

    @test status == :UserLimits
    @test isapprox(m.objVal, 7049.2478976; atol=1e-3)
    @test length(m.internalModel.candidate_disc_vars) == 8
    @test length(m.internalModel.disc_vars) == 3
    @test m.internalModel.disc_var_pick == 1

    # adaptive variable selection scheme :: disc_var_pick = 3
    test_solver = PODSolver(nlp_solver = IpoptSolver(print_level=0),
                            mip_solver = CbcSolver(loglevel=0),
                            disc_var_pick = 3,
                            presolve_bp = false,
                            presolve_bt = false,
                            max_iter = 2,
                            log_level = 100)
    m = nlp3(solver=test_solver)
    status = solve(m)

    @test status == :UserLimits
    @test isapprox(m.objVal, 7049.2478976; atol=1e-3)
    @test length(m.internalModel.candidate_disc_vars) == 8
    @test length(m.internalModel.disc_vars) == 8
    @test m.internalModel.disc_var_pick == 3
end

@testset "Partitioning variable (default) selection test :: blend029 (maximization problem)" begin

    # 15 variable selection rule (default disc_var_pick)
    test_solver = PODSolver(minlp_solver = pavito_solver,
                            nlp_solver = IpoptSolver(print_level=0),
                            mip_solver = CbcSolver(loglevel=0),
                            disc_uniform_rate = 10,
                            presolve_bp = false,
                            presolve_bt = false,
                            max_iter = 1,
                            log_level = 0)
    m = blend029_gl(solver=test_solver)
    JuMP.build(m)

    @test length(m.internalModel.candidate_disc_vars) == 26
    @test length(Set(m.internalModel.candidate_disc_vars)) == 26
    @test length(m.internalModel.disc_vars) == 10
    @test m.internalModel.disc_var_pick == 2
end

