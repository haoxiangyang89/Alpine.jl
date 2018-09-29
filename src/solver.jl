export PODSolver

type PODNonlinearModel <: MathProgBase.AbstractNonlinearModel

    # external developer parameters for testing and debugging
    colorful_pod::Any                                           # Turn on for a colorful logs

    # basic solver parameters
    log_level::Int                                              # Verbosity flag: 0 for quiet, 1 for basic solve info, 2 for iteration info (default = 1)
    timeout::Float64                                            # Time limit for algorithm (in seconds, default = Inf) 
    max_iter::Int                                               # Target Maximum Iterations (default = 99)
    rel_gap::Float64                                            # Relative optimality gap termination condition (default = 1e-4)
    rel_gap_ref::AbstractString                                 # Relative gap reference point (default = "ub"; other option = "lb")
    abs_gap::Float64                                            # Absolute optimality gap termination condition (default = 1e-6)
    tol::Float64                                                # Numerical tolerance value used in the algorithmic process (default = 1e-6)
    big_m::Float64                                              # Big M value for unbounded variables (default = 1e4)

    # piecewise convexification method
    convexity_recognition::Bool                                 # Convexity recognition (default value = true)
    bilinear_mccormick::Bool                                    # Piecewise Bilinear Relaxation using McCormick relaxations (default = false)
    bilinear_convexhull::Bool                                   # Piecewise Bilinear Relaxation using Lambda formulation (default = true)
    monomial_convexhull::Bool                                   # Piecewise Monomial Relaxation using quadratic constraints (default = true)

    # user functions for convexification, partitioning, and nonlinear pattern recognition functions
    piecewise_convex_relaxation_methods::Array{Function}        # Array of functions to construct piecewise relaxations for nonlinear terms :: no over-ride privilege
    partition_injection_methods::Array{Function}                # Array of functions for special methods to add partitions to variable using complex conditions
    term_pattern_methods::Array{Function}                       # Array of functions that user wish to use to parse/recognize nonlinear terms in constraint expression
    constr_pattern_methods::Array{Function}                     # Array of functions that user wish to use to parse/recognize structural constraint from expression

    # parameters used in partitioning algorithm (todo: need to implement disc_var_pick option 3)
    disc_ratio::Any                                             # Discretization ratio parameter; uses a fixed value (default = 4)
    disc_uniform_rate::Int                                      # Discretization rate when using uniform partitions (default = 2)
    disc_var_pick::Any                                          # Algorithm for choosing the variables to discretize: 0/1/2 (default = 2) (max. cover/min. vertex cover/automatic based on var count of 15)
    disc_divert_chunks::Int                                     # Initial number of uniform partitions to construct (default = 5)
    disc_add_partition_method::Any                              # Additional methods to add discretization (default = "adaptive")
    disc_abs_width_tol::Float64                                 # Absolute width tolerance used when setting up partition/discretizations (default = 1e-4)
    disc_rel_width_tol::Float64                                 # Relative width tolerance used when setting up partition/discretizations (default = 1e-6)
    disc_consecutive_forbid_iter::Int                           # Number of consecutive iter. for which bounding solution is in same region, after which no partitions will be added to that region; done per variable (default = 0) 
    disc_ratio_branch::Bool                                     # Branching tests for automated picking of discretization ratios (default = false)

    # piecewise relaxation parameters 
    convhull_formulation::String                                # Type of formulation to be used for formulating the Lambda-based piecewise multilinear relaxation: "sos2"/"facet" (default = "sos2")
    convhull_warmstart::Bool                                    # Warm start the bounding solves (default = true)
    convhull_no_good_cuts::Bool                                 # Add no good cuts to bounding solves using pooled solutions (default = true)
    convhull_ebd::Bool                                          # Enable embedding in Lambda formulation; if true it will default to using log. number of partition binary variables (default = false) 
    convhull_ebd_encode::Any                                    # Encoding method used for log. number of binary variables (default = "default")
    convhull_ebd_ibs::Bool                                      # Enable independent branching scheme
    convhull_ebd_link::Bool                                     # Linking constraints between x and α, if no encoding uses a traditional bounding constraint and with encoding uses a big-M type constraint

    # presolve parameters
    presolve_track_time::Bool                                   # Accounting for presolve time for total time usage (default = true)
    presolve_bt::Any                                            # Perform bound tightening procedure before main algorithm (default = nothing)
    presolve_time_limit::Float64                                # Presolve time limit in seconds (default = 900 seconds)
    presolve_max_iter::Int                                      # Maximum number of bound-tightening iterations allowed (default = 10)
    presolve_bt_min_bound_width::Float64                        # Minimum variable-bound width (default = 1e-3)
    presolve_bt_precision::Float64                              # Variable bounds truncation precistion (default = 1e-5)
    presolve_bt_algo::Any                                       # Method used for bound tightening procedures, can either be index of default methods or functional inputs (default = 1)
    presolve_bt_relax::Bool                                     # During bound-tightening relax the binary variables in the original problem for performance (default = false)
    presolve_bt_mip_time_limit::Float64                         # Time limit for a single MIP solved in built-in bound tighening algorithm (default = Inf)
    presolve_bp::Bool                                           # Enable basic bound propagation (default = true)
    presolve_infeasible::Bool                                   # Presolve problem feasibility flag
    user_parameters::Dict                                       # Additional parameters used for user-defined function inputs

    # integer problem parameters (NOTE: no support for integer problems, making these parameters obsolete)
    int_enable::Bool                                            # Convert integer problem into binary problem by flatten the choice of variable domain (default = false)
    int_cumulative_disc::Bool                                   # [INACTIVE] Cummulatively involve integer variables for discretization (default = true)
    int_fully_disc::Bool                                        # [INACTIVE] Construct equalvaient formulation for integer variables (default = false)

    # solver inputs 
    local_solver::MathProgBase.AbstractMathProgSolver           # Local solver for solving NLPs or MINLPs (defaults to Ipopt for NLPs and Juniper for MINLPs)
    nlp_solver::MathProgBase.AbstractMathProgSolver             # Local continuous NLP solver for solving NLPs at each iteration : deprecate
    minlp_solver::MathProgBase.AbstractMathProgSolver           # Local MINLP solver for solving MINLPs at each iteration : deprecate 
    mip_solver::MathProgBase.AbstractMathProgSolver             # MILP solver for successive lower bound solves (defaults to Pavito())

    # solver ids for customized solver options
    local_solver_id::AbstractString                             # Local solver identifier string
    nlp_solver_id::AbstractString                               # NLP Solver identifier string : deprecate
    minlp_solver_id::AbstractString                             # MINLP local solver identifier string : deprecate
    mip_solver_id::AbstractString                               # MIP solver identifier string

    # initial data provided by user
    num_var_orig::Int                                           # Number of variables in the NLP/MINLP 
    num_cont_var_orig::Int                                      # Number of continuous variables in the NLP/MINLP 
    num_int_var_orig::Int                                       # Number of binary/integer variables in the NLP/MINLP 
    num_constr_orig::Int                                        # Number of constraints in the NLP/MINLP 
    num_lin_constr_orig::Int                                    # Number of linear constraints in the NLP/MINLP 
    num_nl_constr_orig::Int                                     # Number of nonlinear constraints in the NLP/MINLP 
    var_type_orig::Vector{Symbol}                               # Variable type vector on original variables (only :Bin, :Cont, :Int)
    var_start_orig::Vector{Float64}                             # Variable warm start vector on original variables
    constr_type_orig::Vector{Symbol}                            # Constraint type vector on original variables (only :(==), :(>=), :(<=))
    constr_expr_orig::Vector{Expr}                              # Constraint expressions
    obj_expr_orig::Expr                                         # Objective expression

    # extra initial data that is useful for local solves 
    l_var_orig::Vector{Float64}                                 # Variable lower bounds
    u_var_orig::Vector{Float64}                                 # Variable upper bounds
    l_constr_orig::Vector{Float64}                              # Constraint lower bounds
    u_constr_orig::Vector{Float64}                              # Constraint upper bounds
    sense_orig::Symbol                                          # Problem type (:Min, :Max)
    d_orig::JuMP.NLPEvaluator                                   # Instance of AbstractNLPEvaluator for evaluating gradient, Hessian-vector products, and Hessians of the Lagrangian

    # additional initial data that may be useful later on - non populated for now
    A_orig::Any                                                 # Linear constraint matrix
    A_l_orig::Vector{Float64}                                   # Linear constraint matrix LHS
    A_u_orig::Vector{Float64}                                   # Linear constraint matrix RHS
    is_obj_linear_orig::Bool                                    # Bool variable for type of objective
    c_orig::Vector{Float64}                                     # Coefficient vector for linear objective
    num_lconstr_updated::Int                                    # Updated number of linear constraints - includes linear constraints added via @NLconstraint macro
    num_nlconstr_updated::Int                                   # Updated number of non-linear constraints
    indexes_lconstr_updated::Vector{Int}                        # Indexes of updated linear constraints

    # variable bound vectors for local solves at each iteration 
    l_var::Vector{Float64}                                      # Updated variable lower bounds for local solve
    u_var::Vector{Float64}                                      # Updated variable upper bounds for local solve

    # mixed-integer convex program bounding model
    model_mip::JuMP.Model                                       # JuMP convex MIP model for obtaining bounds at each iteration 

    num_var_linear_mip::Int                                     # Number of linear lifting variables required
    num_var_nonlinear_mip::Int                                  # Number of lifted variables
    num_var_disc_mip::Int                                       # Number of variables on which discretization is performed
    num_constr_convex::Int                                      # Number of structural constraints

    # Expression related and Structural Property Placeholder
    linear_terms::Dict{Any, Any}                                # Dictionary containing details of lifted linear terms
    nonconvex_terms::Dict{Any,Any}                              # Dictionary containing details of lifted non-linear terms
    term_seq::Dict{Int, Any}                                    # Vector-Dictionary for nl terms detection
    nonlinear_constrs::Dict{Any,Any}                            # Dictionary containing details of special constraints
    obj_structure::Symbol                                       # A symbolic indicator of the expression type of objective function
    constr_structure::Vector{Symbol}                            # A vector indicate whether a constraint is with sepcial structure
    bounding_obj_expr_mip::Expr                                 # Lifted objective expression; if linear, same as obj_expr_orig
    bounding_constr_expr_mip::Vector{Expr}                      # Lifted constraints; if linear, same as corresponding constr_expr_orig
    bounding_obj_mip::Dict{Any, Any}                            # Lifted objective expression in affine form
    bounding_constr_mip::Vector{Dict{Any, Any}}                 # Lifted constraint expressions in affine form

    # Discretization Related
    candidate_disc_vars::Vector{Int}                            # A vector of all original variable indices that is involved in the nonlinear terms
    discretization::Dict{Any,Any}                               # Discretization points keyed by the variables
    disc_vars::Vector{Int}                                      # Variables on which discretization is performed
    int_vars::Vector{Int}                                       # Index vector of integer variables
    bin_vars::Vector{Int}                                       # Index vector of binary variable

    # Reformulated problem
    l_var_tight::Vector{Float64}                                # Tightened variable upper bounds
    u_var_tight::Vector{Float64}                                # Tightened variable Lower Bounds
    var_type::Vector{Symbol}                                    # Updated variable type for local solve

    # Solution information
    best_bound::Float64                                         # Best bound to the original NLP/MINLP 
    best_obj::Float64                                           # Best incumbent objective for the original NLP/MINLP  
    best_sol::Vector{Float64}                                   # Best incumbent solution values for the original NLP/MINLP  
    best_bound_sol::Vector{Float64}                             # Best bound variable values for the original NLP/MINLP 
    best_rel_gap::Float64                                       # Relative optimality gap = |best_bound - best_obj|/|best_obj|
    best_abs_gap::Float64                                       # Absolute gap = |best_bound - best_obj|
    bound_sol_history::Vector{Vector{Float64}}                  # History of bounding solutions limited by parameter disc_consecutive_forbid_iter
    bound_sol_pool::Dict{Any, Any}                              # A pool of solutions from solving model_mip

    # Logging information and status
    logs::Dict{Symbol,Any}                                      # Logging information
    status::Dict{Symbol,Symbol}                                 # Detailed status of each different phases in algorithm
    pod_status::Symbol                                          # Current POD solver status 

    # constructor
    function PODNonlinearModel(colorful_pod,
                                log_level, timeout, max_iter, rel_gap, rel_gap_ref, abs_gap, tol, big_m,
                                nlp_solver,
                                minlp_solver,
                                mip_solver,
                                convexity_recognition,
                                bilinear_mccormick,
                                bilinear_convexhull,
                                monomial_convexhull,
                                piecewise_convex_relaxation_methods,
                                partition_injection_methods,
                                term_pattern_methods,
                                constr_pattern_methods,
                                disc_var_pick,
                                disc_ratio,
                                disc_uniform_rate,
                                disc_add_partition_method,
                                disc_divert_chunks,
                                disc_abs_width_tol,
                                disc_rel_width_tol,
                                disc_consecutive_forbid_iter,
                                disc_ratio_branch,
                                convhull_formulation,
                                convhull_ebd,
                                convhull_ebd_encode,
                                convhull_ebd_ibs,
                                convhull_ebd_link,
                                convhull_warmstart,
                                convhull_no_good_cuts,
                                presolve_track_time,
                                presolve_bt,
                                presolve_time_limit,
                                presolve_max_iter,
                                presolve_bt_min_bound_width,
                                presolve_bt_precision,
                                presolve_bt_algo,
                                presolve_bt_relax,
                                presolve_bt_mip_time_limit,
                                presolve_bp,
                                user_parameters,
                                int_enable,
                                int_cumulative_disc,
                                int_fully_disc)

        m = new()

        m.colorful_pod = colorful_pod

        m.log_level = log_level
        m.timeout = timeout
        m.max_iter = max_iter
        m.rel_gap = rel_gap
        m.rel_gap_ref = rel_gap_ref
        m.abs_gap = abs_gap
        m.tol = tol
        m.big_m = big_m

        m.convexity_recognition = convexity_recognition
        m.bilinear_mccormick = bilinear_mccormick
        m.bilinear_convexhull = bilinear_convexhull
        m.monomial_convexhull = monomial_convexhull

        m.piecewise_convex_relaxation_methods = piecewise_convex_relaxation_methods
        m.partition_injection_methods = partition_injection_methods
        m.term_pattern_methods = term_pattern_methods
        m.constr_pattern_methods = constr_pattern_methods

        m.disc_var_pick = disc_var_pick
        m.disc_ratio = disc_ratio
        m.disc_uniform_rate = disc_uniform_rate
        m.disc_add_partition_method = disc_add_partition_method
        m.disc_divert_chunks = disc_divert_chunks
        m.disc_abs_width_tol = disc_abs_width_tol
        m.disc_rel_width_tol = disc_rel_width_tol
        m.disc_consecutive_forbid_iter = disc_consecutive_forbid_iter
        m.disc_ratio_branch = disc_ratio_branch

        m.convhull_formulation = convhull_formulation
        m.convhull_ebd = convhull_ebd
        m.convhull_ebd_encode = convhull_ebd_encode
        m.convhull_ebd_ibs = convhull_ebd_ibs
        m.convhull_ebd_link = convhull_ebd_link
        m.convhull_warmstart = convhull_warmstart
        m.convhull_no_good_cuts = convhull_no_good_cuts

        m.presolve_track_time = presolve_track_time
        m.presolve_bt = presolve_bt
        m.presolve_time_limit = presolve_time_limit
        m.presolve_max_iter = presolve_max_iter
        m.presolve_bt_min_bound_width = presolve_bt_min_bound_width
        m.presolve_bt_precision = presolve_bt_precision
        m.presolve_bt_algo = presolve_bt_algo
        m.presolve_bt_relax = presolve_bt_relax
        m.presolve_bt_mip_time_limit = presolve_bt_mip_time_limit

        m.presolve_bp = presolve_bp

        m.nlp_solver = nlp_solver
        m.minlp_solver = minlp_solver
        m.mip_solver = mip_solver

        m.user_parameters = user_parameters
        m.int_enable = int_enable
        m.int_cumulative_disc = int_cumulative_disc
        m.int_fully_disc = int_fully_disc

        m.num_var_orig = 0
        m.num_cont_var_orig = 0
        m.num_int_var_orig = 0
        m.num_constr_orig = 0
        m.num_lin_constr_orig = 0
        m.num_nl_constr_orig = 0
        m.var_type_orig = Symbol[]
        m.var_start_orig = Float64[]
        m.constr_type_orig = Symbol[]
        m.constr_expr_orig = Expr[]
        m.num_lconstr_updated = 0
        m.num_nlconstr_updated = 0
        m.indexes_lconstr_updated = Int[]

        m.linear_terms = Dict()
        m.nonconvex_terms = Dict()
        m.term_seq = Dict()
        m.nonlinear_constrs = Dict()
        m.candidate_disc_vars = Int[]
        m.bounding_constr_expr_mip = []
        m.bounding_constr_mip = []
        m.disc_vars = []
        m.int_vars = []
        m.bin_vars = []
        m.discretization = Dict()
        m.num_var_linear_mip = 0
        m.num_var_nonlinear_mip = 0
        m.num_var_disc_mip = 0
        m.num_constr_convex = 0
        m.constr_structure = []
        m.best_bound_sol = []
        m.bound_sol_history = []
        m.presolve_infeasible = false
        m.bound_sol_history = Vector{Vector{Float64}}(m.disc_consecutive_forbid_iter)

        m.best_obj = Inf
        m.best_bound = -Inf
        m.best_rel_gap = Inf
        m.best_abs_gap = Inf
        m.pod_status = :NotLoaded

        create_status!(m)
        create_logs!(m)

        return m
    end
end

type UnsetSolver <: MathProgBase.AbstractMathProgSolver
end

type PODSolver <: MathProgBase.AbstractMathProgSolver

    colorful_pod::Any

    log_level::Int
    timeout::Float64
    max_iter::Int
    rel_gap::Float64
    rel_gap_ref::AbstractString
    abs_gap::Float64
    tol::Float64
    big_m::Float64

    nlp_solver::MathProgBase.AbstractMathProgSolver
    minlp_solver::MathProgBase.AbstractMathProgSolver
    mip_solver::MathProgBase.AbstractMathProgSolver

    convexity_recognition::Bool
    bilinear_mccormick::Bool
    bilinear_convexhull::Bool
    monomial_convexhull::Bool

    piecewise_convex_relaxation_methods::Array{Function}
    partition_injection_methods::Array{Function}
    term_pattern_methods::Array{Function}
    constr_pattern_methods::Array{Function}

    disc_var_pick::Any
    disc_ratio::Any
    disc_uniform_rate::Int
    disc_add_partition_method::Any
    disc_divert_chunks::Int
    disc_abs_width_tol::Float64
    disc_rel_width_tol::Float64
    disc_consecutive_forbid_iter::Int
    disc_ratio_branch::Bool

    convhull_formulation::String
    convhull_ebd::Bool
    convhull_ebd_encode::Any
    convhull_ebd_ibs::Bool
    convhull_ebd_link::Bool
    convhull_warmstart::Bool
    convhull_no_good_cuts::Bool

    presolve_track_time::Bool
    presolve_bt::Any
    presolve_time_limit::Float64
    presolve_max_iter::Int
    presolve_bt_min_bound_width::Float64
    presolve_bt_precision::Float64
    presolve_bt_algo::Any
    presolve_bt_relax::Bool
    presolve_bt_mip_time_limit::Float64

    presolve_bp::Bool

    user_parameters::Dict
    int_enable::Bool
    int_cumulative_disc::Bool
    int_fully_disc::Bool

    # other options to be added later on
end

function PODSolver(;
    colorful_pod = false,

    log_level = 1,
    timeout = Inf,
    max_iter = 99,
    rel_gap = 1e-4,
    rel_gap_ref = "ub",
    abs_gap = 1e-6,
    tol = 1e-6,
    big_m = 1e4,

    nlp_solver = UnsetSolver(),
    minlp_solver = UnsetSolver(),
    mip_solver = UnsetSolver(),

    convexity_recognition = true,
    bilinear_mccormick = false,
    bilinear_convexhull = true,
    monomial_convexhull = true,

    piecewise_convex_relaxation_methods = Array{Function}(0),
    partition_injection_methods = Array{Function}(0),
    term_pattern_methods = Array{Function}(0),
    constr_pattern_methods = Array{Function}(0),

    disc_var_pick = 2,                      # By default use the 15-variable selective rule
    disc_ratio = 4,
    disc_uniform_rate = 2,
    disc_add_partition_method = "adaptive",
    disc_divert_chunks = 5,
    disc_abs_width_tol = 1e-4,
    disc_rel_width_tol = 1e-6,
    disc_consecutive_forbid_iter = 0,
    disc_ratio_branch=false,

    convhull_formulation = "sos2",
    convhull_ebd = false,
    convhull_ebd_encode = "default",
    convhull_ebd_ibs = false,
    convhull_ebd_link = false,
    convhull_warmstart = true,
    convhull_no_good_cuts = true,

    presolve_track_time = true,
    presolve_max_iter = 10,
    presolve_bt = nothing,
    presolve_time_limit = 900,
    presolve_bt_min_bound_width = 1e-3,
    presolve_bt_precision = 1e-5,
    presolve_bt_algo = 1,
    presolve_bt_relax = false,
    presolve_bt_mip_time_limit = Inf,

    presolve_bp = true,

    user_parameters = Dict(),
    int_enable = false,
    int_cumulative_disc = true,
    int_fully_disc = false,

    kwargs...
    )

    # Option Screening
    unsupport_opts = Dict(kwargs)
    !isempty(keys(unsupport_opts)) && error("Detected unsupported/experimental arguments = $(keys(unsupport_opts))")

    nlp_solver == UnsetSolver() && error("No NLP local solver specified (set nlp_solver)\n")
    mip_solver == UnsetSolver() && error("NO MIP solver specififed (set mip_solver)\n")

    rel_gap_ref in ["ub", "lb"] || error("Gap calculation only takes 'ub' pr 'lb'")

    # String Code Conversion
    if disc_var_pick in ["ncvar_collect_nodes", "all", "max"]
        disc_var_pick = 0
    elseif disc_var_pick in ["min_vertex_cover","min"]
        disc_var_pick = 1
    elseif disc_var_pick == "selective"
        disc_var_pick = 2
    elseif disc_var_pick == "dynamic"
        disc_var_pick = 3
    end

    # Deepcopy the solvers because we may change option values inside POD
    PODSolver(colorful_pod,
        log_level, timeout, max_iter, rel_gap, rel_gap_ref, abs_gap, tol, big_m,
        deepcopy(nlp_solver),
        deepcopy(minlp_solver),
        deepcopy(mip_solver),
        convexity_recognition,
        bilinear_mccormick,
        bilinear_convexhull,
        monomial_convexhull,
        piecewise_convex_relaxation_methods,
        partition_injection_methods,
        term_pattern_methods,
        constr_pattern_methods,
        disc_var_pick,
        disc_ratio,
        disc_uniform_rate,
        disc_add_partition_method,
        disc_divert_chunks,
        disc_abs_width_tol,
        disc_rel_width_tol,
        disc_consecutive_forbid_iter,
        disc_ratio_branch,
        convhull_formulation,
        convhull_ebd,
        convhull_ebd_encode,
        convhull_ebd_ibs,
        convhull_ebd_link,
        convhull_warmstart,
        convhull_no_good_cuts,
        presolve_track_time,
        presolve_bt,
        presolve_time_limit,
        presolve_max_iter,
        presolve_bt_min_bound_width,
        presolve_bt_precision,
        presolve_bt_algo,
        presolve_bt_relax,
        presolve_bt_mip_time_limit,
        presolve_bp,
        user_parameters,
        int_enable,
        int_cumulative_disc,
        int_fully_disc)
    end

# Create POD nonlinear model: can solve with nonlinear algorithm only
function MathProgBase.NonlinearModel(s::PODSolver)

    if !applicable(MathProgBase.NonlinearModel, s.nlp_solver)
        error("NLP local solver $(s.nlp_solver) specified is not a NLP solver recognized by POD\n")
    end

    # Translate options into old nonlinearmodel.jl fields
    colorful_pod = s.colorful_pod

    log_level = s.log_level
    timeout = s.timeout
    max_iter = s.max_iter
    rel_gap = s.rel_gap
    rel_gap_ref = s.rel_gap_ref
    abs_gap = s.abs_gap
    tol = s.tol
    big_m = s.big_m

    convexity_recognition = s.convexity_recognition
    bilinear_mccormick = s.bilinear_mccormick
    bilinear_convexhull = s.bilinear_convexhull
    monomial_convexhull = s.monomial_convexhull

    piecewise_convex_relaxation_methods = s.piecewise_convex_relaxation_methods
    partition_injection_methods = s.partition_injection_methods
    term_pattern_methods = s.term_pattern_methods
    constr_pattern_methods = s.constr_pattern_methods

    nlp_solver = s.nlp_solver
    minlp_solver = s.minlp_solver
    mip_solver = s.mip_solver

    disc_var_pick = s.disc_var_pick
    disc_ratio = s.disc_ratio
    disc_uniform_rate = s.disc_uniform_rate
    disc_add_partition_method = s.disc_add_partition_method
    disc_divert_chunks = s.disc_divert_chunks
    disc_abs_width_tol = s.disc_abs_width_tol
    disc_rel_width_tol = s.disc_rel_width_tol
    disc_consecutive_forbid_iter = s.disc_consecutive_forbid_iter
    disc_ratio_branch = s.disc_ratio_branch

    convhull_formulation = s.convhull_formulation
    convhull_ebd = s.convhull_ebd
    convhull_ebd_encode = s.convhull_ebd_encode
    convhull_ebd_ibs = s.convhull_ebd_ibs
    convhull_ebd_link = s.convhull_ebd_link
    convhull_warmstart = s.convhull_warmstart
    convhull_no_good_cuts = s.convhull_no_good_cuts

    presolve_track_time = s.presolve_track_time
    presolve_bt = s.presolve_bt
    presolve_time_limit = s.presolve_time_limit
    presolve_max_iter = s.presolve_max_iter
    presolve_bt_min_bound_width = s.presolve_bt_min_bound_width
    presolve_bt_precision = s.presolve_bt_precision
    presolve_bt_algo = s.presolve_bt_algo
    presolve_bt_relax = s.presolve_bt_relax
    presolve_bt_mip_time_limit = s.presolve_bt_mip_time_limit

    presolve_bp = s.presolve_bp

    user_parameters = s.user_parameters
    int_enable = s.int_enable
    int_cumulative_disc = s.int_cumulative_disc
    int_fully_disc = s.int_fully_disc

    return PODNonlinearModel(colorful_pod,
                            log_level, timeout, max_iter, rel_gap, rel_gap_ref, abs_gap, tol, big_m,
                            nlp_solver,
                            minlp_solver,
                            mip_solver,
                            convexity_recognition,
                            bilinear_mccormick,
                            bilinear_convexhull,
                            monomial_convexhull,
                            piecewise_convex_relaxation_methods,
                            partition_injection_methods,
                            term_pattern_methods,
                            constr_pattern_methods,
                            disc_var_pick,
                            disc_ratio,
                            disc_uniform_rate,
                            disc_add_partition_method,
                            disc_divert_chunks,
                            disc_abs_width_tol,
                            disc_rel_width_tol,
                            disc_consecutive_forbid_iter,
                            disc_ratio_branch,
                            convhull_formulation,
                            convhull_ebd,
                            convhull_ebd_encode,
                            convhull_ebd_ibs,
                            convhull_ebd_link,
                            convhull_warmstart,
                            convhull_no_good_cuts,
                            presolve_track_time,
                            presolve_bt,
                            presolve_time_limit,
                            presolve_max_iter,
                            presolve_bt_min_bound_width,
                            presolve_bt_precision,
                            presolve_bt_algo,
                            presolve_bt_relax,
                            presolve_bt_mip_time_limit,
                            presolve_bp,
                            user_parameters,
                            int_enable,
                            int_cumulative_disc,
                            int_fully_disc)
end

function MathProgBase.loadproblem!(m::PODNonlinearModel,
                                   num_var::Int,
                                   num_constr::Int,
                                   l_var::Vector{Float64},
                                   u_var::Vector{Float64},
                                   l_constr::Vector{Float64},
                                   u_constr::Vector{Float64},
                                   sense::Symbol,
                                   d::MathProgBase.AbstractNLPEvaluator)

    # Basic Problem Dimensions
    m.num_var_orig = num_var
    m.num_constr_orig = num_constr
    m.l_var_orig = l_var
    m.u_var_orig = u_var
    m.l_constr_orig = l_constr
    m.u_constr_orig = u_constr
    m.sense_orig = sense
    if m.sense_orig == :Max
        m.best_obj = -Inf
        m.best_bound = Inf
    else
        m.best_obj = Inf
        m.best_bound = -Inf
    end
    m.d_orig = d

    # Initialize NLP interface
    interface_init_nonlinear_data(m.d_orig)

    # Collect objective & constraints expressions
    m.obj_expr_orig = interface_get_obj_expr(m.d_orig)
    for i in 1:m.num_constr_orig
        push!(m.constr_expr_orig, interface_get_constr_expr(m.d_orig, i))
    end

    # Collect original variable type and build dynamic variable type space
    m.var_type_orig = [getcategory(Variable(d.m, i)) for i in 1:m.num_var_orig]
    m.var_type = copy(m.var_type_orig)
    m.int_vars = [i for i in 1:m.num_var_orig if m.var_type[i] == :Int]
    m.bin_vars = [i for i in 1:m.num_var_orig if m.var_type[i] == :Bin]

    # Summarize constraints information in original model
    @compat m.constr_type_orig = Array{Symbol}(m.num_constr_orig)

    for i in 1:m.num_constr_orig
        if l_constr[i] > -Inf && u_constr[i] < Inf
            m.constr_type_orig[i] = :(==)
        elseif l_constr[i] > -Inf
            m.constr_type_orig[i] = :(>=)
        else
            m.constr_type_orig[i] = :(<=)
        end
    end

    # Initialize recognizable structure properties with :none
    m.obj_structure = :none
    m.constr_structure = [:none for i in 1:m.num_constr_orig]
    for i = 1:m.num_constr_orig
        if interface_is_constr_linear(m.d_orig, i)
            m.num_lin_constr_orig += 1
            m.constr_structure[i] = :generic_linear
        else
            m.num_nl_constr_orig += 1
            m.constr_structure[i] = :generic_nonlinear
        end
    end

    @assert m.num_constr_orig == m.num_nl_constr_orig + m.num_lin_constr_orig
    m.is_obj_linear_orig = interface_is_obj_linear(m.d_orig)
    m.is_obj_linear_orig ? (m.obj_structure = :generic_linear) : (m.obj_structure = :generic_nonlinear)

    # Other preload Built-in Special Functions (append special functions to user-functions)

    # populate data to create the bounding model
    recategorize_var(m)             # Initial round of variable recategorization

    :Int in m.var_type_orig && warn("POD's support for integer variables is highly experimental.")
    :Int in m.var_type_orig ? m.int_enable = true : m.int_enable = false # Separator for safer runs

    # Conduct solver-dependent detection
    fetch_mip_solver_identifier(m)
    fetch_nlp_solver_identifier(m)
    fetch_minlp_solver_identifier(m)

    # Solver Dependent Options
    if m.mip_solver_id != :Gurobi
        m.convhull_warmstart == false
        m.convhull_no_good_cuts == false
    end

    # Main Algorithmic Initialization
    process_expr(m)                         # Compact process of every expression
    init_tight_bound(m)                     # Initialize bounds for algorithmic processes
    resolve_var_bounds(m)                   # resolve lifted var bounds
    pick_disc_vars(m)                       # Picking variables to be discretized
    init_disc(m)                            # Initialize discretization dictionarys

    # Turn-on bt presolver if not discrete variables
    if isempty(m.int_vars) && length(m.bin_vars) <= 50 && m.num_var_orig <= 10000 && length(m.candidate_disc_vars)<=300 && m.presolve_bt == nothing
        m.presolve_bt = true
        println("Automatically turning on bound-tightening presolver...")
    elseif m.presolve_bt == nothing  # If no use indication
        m.presolve_bt = false
    end

    if length(m.bin_vars) > 200 || m.num_var_orig > 2000
        println("Automatically turning OFF ratio branching due to the size of the problem")
        m.disc_ratio_branch=false
    end

    # Initialize the solution pool
    m.bound_sol_pool = initialize_solution_pool(m, 0)  # Initialize the solution pool

    # Record the initial solution from the warmstarting value, if any
    m.best_sol = m.d_orig.m.colVal

    # Check if any illegal term exist in the warm-solution
    any(isnan, m.best_sol) && (m.best_sol = zeros(length(m.best_sol)))

    # Initialize log
    logging_summary(m)

    ## Citation
    println("----------------------------------------------------------------------")
    println("If you find POD useful, please cite the following work. Thanks!!!")
    println("Nagarajan, H., Lu, M., Wang, S., Bent, R. and Sundar, K., 2017. ")
    println("An Adaptive, Multivariate Partitioning Algorithm for Global ")
    println("Optimization of Nonconvex Programs. arXiv preprint arXiv:1707.02514.")
    println("----------------------------------------------------------------------")

    return
end
