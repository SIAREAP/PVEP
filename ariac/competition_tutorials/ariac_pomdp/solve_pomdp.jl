#!/usr/bin/env julia

#############################
# 1. 你的 SymbolicMDPs 模块 #
#############################

module SymbolicMDPs

export SymbolicMDP, SymbolicRLEnv

using Random

import PDDL: PDDL, Domain, Problem, State, Term, Compound, Const, @pddl_str
import POMDPs: POMDPs, MDP
import POMDPTools: Deterministic, Uniform, SparseCat, MDPCommonRLEnv
import CommonRLInterface

## Utility functions ##

typerange(T) = nothing
typerange(T::Type{<:Integer}) = typemin(T):typemax(T)
typerange(T::Type{<:Enum}) = instances(T)

compare_terms(x::Term, y::Term) = x.name < y.name
compare_terms(x::Compound, y::Compound) = string(x) < string(y)

## SymbolicStateSpace ##

"Product representation of a symbolic state space for ground PDDL domains."
struct SymbolicStateSpace{S <: State, I, FT, FR}
    state::S # Reference state with static values of fluents
    ftypes::FT # Map from (non-static) fluent names to their Julia types
    franges::FR # Map from (non-static) fluent names to their value ranges
    fluents::Vector{Term} # List of all (non-static) ground fluents
    iter::I # Iterator over states
end

function SymbolicStateSpace(domain::Domain, state::State)
    # Make copy of state for reference
    state = deepcopy(state)
    # Infer static fluents so that we can avoid enumerating over their values
    statics = PDDL.infer_static_fluents(domain)
    # Extract and order fluent types and value ranges
    sigs = sort(collect(pairs(PDDL.get_fluents(domain))), by=first)
    ftypes = (; (f => PDDL.global_datatypes()[sig.type]
                 for (f, sig) in sigs if !(f in statics))...)
    franges = (; (f => typerange(ty) for (f, ty) in pairs(ftypes))...)
    # Construct list of all (non-static) ground fluents
    fluents = Term[]
    for (fname, sig) in sigs
        if fname in statics continue end
        if PDDL.arity(sig) == 0
            push!(fluents, Const(fname))
        else
            fs = [Compound(fname, collect(Term, args))
                  for args in PDDL.groundargs(domain, state, fname)] |> vec
            append!(fluents, sort!(fs, lt=compare_terms))
        end
    end
    # Construct iterator over states
    if (any(!(ftypes[f.name] <: Union{Integer,Enum}) for f in fluents) ||
        length(fluents) > Sys.WORD_SIZE)
        # Not iterable if some fluents are continuous, or there are too many
        state_iter = nothing
    else
        # Construct product iterator over ground fluent values
        val_iters = (franges[f.name] for f in fluents)
        state_iter = Base.Generator(Iterators.product(val_iters...)) do vals
            s = copy(state)
            for (f, val) in zip(fluents, vals)
                s[f] = val
            end
            return s
        end
    end
    return SymbolicStateSpace(state, ftypes, franges, fluents, state_iter)
end

Base.eltype(space::SymbolicStateSpace{S}) where {S} = S

Base.iterate(space::SymbolicStateSpace{S,Nothing}) where {S} =
    error("Non-iterable space.")
Base.iterate(space::SymbolicStateSpace{S,Nothing}, i) where {S} =
    error("Non-iterable space.")
Base.length(space::SymbolicStateSpace{S,Nothing}) where {S} =
    error("Non-iterable space.")

Base.iterate(space::SymbolicStateSpace{S}) where {S} =
    iterate(space.iter)
Base.iterate(space::SymbolicStateSpace{S}, i) where {S} =
    iterate(space.iter, i)
Base.length(space::SymbolicStateSpace{S}) where {S} =
    prod(length(space.franges[f.name]) for f in space.fluents)

function Base.rand(rng::AbstractRNG, space::SymbolicStateSpace)
    state = copy(space.state)
    val_iters = for f in space.fluents
        ty = space.ftypes[f.name]
        state[f] = Base.rand(rng, ty)
    end
    return state
end

function POMDPs.stateindex(space::SymbolicStateSpace{S}, state::S) where {S}
    if space.iter === nothing
        error("State space is not discrete.")
    end
    idx = 0
    for f in Iterators.reverse(space.fluents)
        ty = space.ftypes[f.name]
        vrng = space.franges[f.name]
        val = state[f]
        offset = (ty <: Enum ? Int(val) : findfirst(==(val), vrng)) - 1
        idx = idx * length(vrng) + offset
    end
    return idx + 1
end

function vectorize(space::SymbolicStateSpace, state::State)
    T = typejoin(space.ftypes...)
    return broadcast(f -> state[f]::T, space.fluents)
end

## SymbolicMDP ##

"""
    SymbolicMDP

MDP wrapper for a ground PDDL domain.
"""
struct SymbolicMDP{D<:Domain,S<:State,SS<:SymbolicStateSpace,C} <: MDP{S,Term}
    domain::D
    init::S
    goal::Term
    metric::Union{Term,Nothing}
    discount::Float64
    states::SS
    actions::Vector{Compound}
    a_cache::C
end

"""
    SymbolicMDP(domain, state, [goal, metric, discount]; cache_actions=false)

Construct a symbolic MDP from a PDDL `domain` and initial `state`.
"""
function SymbolicMDP(domain::Domain, state::State,
                     goal=pddl"(true)", metric=nothing, discount=1.0;
                     cache_actions::Bool=true)
    s_space = SymbolicStateSpace(domain, state)
    actions = Compound[act.term for act in PDDL.groundactions(domain, state)]
    actions = sort!(actions, lt=compare_terms)
    a_cache = cache_actions ? Dict{UInt64,Vector{Compound}}() : nothing
    return SymbolicMDP(domain, state, goal, metric, discount,
                       s_space, actions, a_cache)
end

"""
    SymbolicMDP(domain, problem; cache_actions=false)

Construct a symbolic MDP from a PDDL `domain` and `problem`.
"""
function SymbolicMDP(domain::Domain, problem::Problem;
    discount::Float64 = 1.0,
    cache_actions::Bool = true)
state = PDDL.initstate(domain, problem)
goal = PDDL.get_goal(problem)
metric = PDDL.get_metric(problem)
if metric !== nothing # Extract metric formula to minimize
metric = metric.name == :minimize ?
metric.args[1] : Compound(:-, metric.args)
end
# 这里用「位置参数」把 discount 传进去
return SymbolicMDP(domain, state, goal, metric, discount;
      cache_actions=cache_actions)
end


POMDPs.states(m::SymbolicMDP) = m.states

POMDPs.actions(m::SymbolicMDP) = m.actions

"只在 s 真的是一个 PDDL State 时，才用 available 过滤可用动作"
POMDPs.actions(m::SymbolicMDP, s::State) = m.a_cache isa Nothing ?
    collect(PDDL.available(m.domain, s)) :
    get!(() -> collect(PDDL.available(m.domain, s)), m.a_cache, hash(s))


POMDPs.initialstate(m::SymbolicMDP) =
    Deterministic(deepcopy(m.init))
POMDPs.transition(m::SymbolicMDP, s, a) =
    Deterministic(PDDL.transition(m.domain, deepcopy(s), a))
POMDPs.reward(m::SymbolicMDP, s, a, sp) = m.metric === nothing ?
    -1 : m.domain[s => m.metric] - m.domain[sp => m.metric]
POMDPs.discount(m::SymbolicMDP) =
    m.discount
POMDPs.isterminal(m::SymbolicMDP, s) =
    PDDL.satisfy(m.domain, s, m.goal)

# 使用 deepcopy 确保 PDDL.transition 不会在原地修改传入的状态
POMDPs.stateindex(m::SymbolicMDP, s) =
    POMDPs.stateindex(m.states, s)
POMDPs.actionindex(m::SymbolicMDP, a) =
    searchsortedfirst(m.actions, a; lt=compare_terms)

POMDPs.convert_s(::Type{Any}, s::State, m::SymbolicMDP) =
    s
POMDPs.convert_s(::Type{<:AbstractArray}, s::State, m::SymbolicMDP) =
    vectorize(m.states, s)
POMDPs.convert_s(::Type{Int}, s::State, m::SymbolicMDP) =
    POMDPs.stateindex(m, s)

POMDPs.convert_a(::Type{Any}, a::Term, m::SymbolicMDP) =
    a
POMDPs.convert_a(::Type{<:AbstractArray}, a::Term, m::SymbolicMDP) =
    [POMDPs.actionindex(m, a)]
POMDPs.convert_a(::Type{Int}, a::Term, m::SymbolicMDP) =
    POMDPs.actionindex(m, a)

## Implement additional common RL environment interface functions ##

const CRL = CommonRLInterface
const SymbolicMDPCommonRLEnv =
    MDPCommonRLEnv{RLO, M} where {RLO, M <: SymbolicMDP}


SymbolicRLEnv(S::Type, mdp::SymbolicMDP) =
    MDPCommonRLEnv{S}(mdp)
SymbolicRLEnv(S::Type, domain::Domain, args...; kwargs...) =
    SymbolicRLEnv(S, SymbolicMDP(domain, args...; kwargs...))
SymbolicRLEnv(domain::Domain, args...; kwargs...) =
    SymbolicRLEnv(Any, domain, args...; kwargs...)

CRL.@provide CRL.valid_action_mask(env::SymbolicMDPCommonRLEnv) =
    broadcast(in(CRL.valid_actions(env)), env.m.actions)

end # module SymbolicMDPs

#########################################################
# 2. RobotAssembly “单文件分层”重构                     #
#    模拟 domain / models / agent / env / problem 结构  #
#########################################################

module AriacKitting

using PDDL
using POMDPs
using BasicPOMCP
using ParticleFilters
using StatsBase: countmap, Weights, sample
using Random
using ..SymbolicMDPs
using PDDL: State, Term, Compound, Const, Domain
using POMDPTools: Deterministic, SparseCat

export init_prior, update_belief_loc_target, update_belief_quality, solve_trigger

########################
# 设计要点（kitting 版本）
#
# - 真实环境（transition）: 直接用 p_real 构成的 SymbolicMDP（确定性 PDDL.transition）
# - 不确定性（belief）: 只在「part_on(零件位置)」和「good/bad(质量)」上采样
# - 信息获取（observation）:
#   - inspect(robot, location) -> 观测该 location 上“看到的零件”（带少量漏检噪声）
#   - perform_quality_check(order, part, agv_slot, agv) -> 观测该 part 的 good/bad（带噪声）
########################

########################
# Domain helpers / Obs #
########################

const PARTS = Ref(Vector{Symbol}())
const BIN_SLOTS = Ref(Vector{Symbol}())
const PRIORITY_LOCS = Ref(Set{Symbol}())
const GRIPPER_STATIONS = Ref(Set{Symbol}())
const TRAY_SLOTS = Ref(Set{Symbol}())
const AGV_SLOTS = Ref(Set{Symbol}())
const AGV_DESTS = Ref(Set{Symbol}())

# 从 p_real 的任务描述抽取“目标订单”的关键信息（用于策略/动作剪枝）
const ORDER_SYM = Ref{Union{Symbol,Nothing}}(nothing)
const AGV_SYM = Ref{Union{Symbol,Nothing}}(nothing)
const TRAY_SYM = Ref{Union{Symbol,Nothing}}(nothing)
const REQUIRED_PARTS = Ref(Set{Symbol}())
const TARGET_SLOT_FOR_PART = Ref(Dict{Symbol,Symbol}())  # part => agv_slot

# 轻量 progressive widening（在 POMDP 层实现）
# 说明：BasicPOMCP.POMCPSolver 本身没有 action progressive widening 参数，
# 所以这里用“按状态访问次数逐步放开 move_floor 候选目的地”的近似来控分支。
const PW_STATE_VISITS = Dict{UInt64, Int}()
const PW_MOVE_MIN = 2
const PW_K = 2.0
const PW_ALPHA = 0.5

part_on_term(p::Symbol, b::Symbol) = Compound(:part_on, Term[Const(p), Const(b)])
good_term(p::Symbol) = Compound(:good, Term[Const(p)])
bad_term(p::Symbol)  = Compound(:bad,  Term[Const(p)])

holding_part_term(p::Symbol) = Compound(:holding_part, Term[Const(:floor), Const(p)])
holding_tray_term(t::Symbol) = Compound(:holding_tray, Term[Const(:floor), Const(t)])
on_agv_term(t::Symbol, a::Symbol) = Compound(:on_agv, Term[Const(t), Const(a)])
in_slot_term(p::Symbol, s::Symbol) = Compound(:in_slot, Term[Const(p), Const(s)])
tray_on_slot_term(t::Symbol, s::Symbol) = Compound(:tray_on_slot, Term[Const(t), Const(s)])
agv_at_term(a::Symbol, d::Symbol) = Compound(:agv_at, Term[Const(a), Const(d)])
home_term(a::Symbol, d::Symbol) = Compound(:home, Term[Const(a), Const(d)])

at_robot_term(loc::Symbol) = Compound(:at_robot, Term[Const(:floor), Const(loc)])

# Unary robot fluents (domain defines these with a robot parameter)
has_part_gripper_term(r::Symbol=:floor) = Compound(:has_part_gripper, Term[Const(r)])
has_tray_gripper_term(r::Symbol=:floor) = Compound(:has_tray_gripper, Term[Const(r)])
gripper_empty_term(r::Symbol=:floor)    = Compound(:gripper_empty, Term[Const(r)])

function _robot_loc(s::State)::Symbol
    # PRIORITY_LOCS 不包含 :floor_init，因此单独兜底。
    for loc in PRIORITY_LOCS[]
        if s[at_robot_term(loc)]
            return loc
        end
    end
    if s[at_robot_term(:floor_init)]
        return :floor_init
    end
    return :unknown
end


is_inspect_action(a::Term) = a isa Compound && (a::Compound).name == :inspect
is_qc_action(a::Term)      = a isa Compound && (a::Compound).name == :perform_quality_check
is_repair_action(a::Term)  = false  # repair action removed from domain

inspected_location(a::Term)::Symbol = (a::Compound).args[2].name
checked_part(a::Term)::Symbol       = (a::Compound).args[2].name
repaired_part(a::Term)::Symbol      = :none

@enum QualityLabel begin
    QNone
    QGood
    QBad
end

abstract type AriacObs end

struct QualityObs <: AriacObs
    part::Symbol
    label::QualityLabel
end

struct InspectObs <: AriacObs
    location::Symbol
    part::Symbol  # :none 表示空
end

import Base: ==, isequal, hash
==(a::QualityObs, b::QualityObs) = (a.part == b.part) && (a.label == b.label)
isequal(a::QualityObs, b::QualityObs) = (a.part === b.part) && (a.label == b.label)
hash(o::QualityObs, h::UInt) = hash(o.part, hash(o.label, h))

==(a::InspectObs, b::InspectObs) = (a.location == b.location) && (a.part == b.part)
isequal(a::InspectObs, b::InspectObs) = (a.location === b.location) && (a.part == b.part)
hash(o::InspectObs, h::UInt) = hash(o.location, hash(o.part, h))

function set_quality!(s::State, p::Symbol, is_good::Bool)
    s[good_term(p)] = is_good
    s[bad_term(p)]  = !is_good
    return s
end

function set_part_loc!(s::State, p::Symbol, b::Symbol)
    for b2 in BIN_SLOTS[]
        s[part_on_term(p, b2)] = (b2 == b)
    end
    return s
end

function part_at_bin(s::State, b::Symbol)::Symbol
    for p in PARTS[]
        if s[part_on_term(p, b)]
            return p
        end
    end
    return :none
end

########################
# POMDP wrapper         #
########################

_act_term(act) = hasproperty(act, :term) ? getproperty(act, :term) : act

struct KittingPOMDP{M,B} <: POMDP{State,Term,AriacObs}
    env_mdp::M
    init_belief::B
end

POMDPs.statetype(p::KittingPOMDP)  = POMDPs.statetype(p.env_mdp)
POMDPs.actiontype(p::KittingPOMDP) = POMDPs.actiontype(p.env_mdp)
POMDPs.obstype(::KittingPOMDP)     = AriacObs
const GLOBAL_ACTIONS = Ref(Vector{Term}())

function _tray_slot_in_init(mdp_env)::Union{Symbol,Nothing}
    if TRAY_SYM[] === nothing
        return nothing
    end
    s0 = deepcopy(mdp_env.init)
    t = TRAY_SYM[]::Symbol
    for ts in TRAY_SLOTS[]
        if s0[tray_on_slot_term(t, ts)]
            return ts
        end
    end
    return nothing
end

"为 BasicPOMCP/POMCP 构建一个小而覆盖关键序列的全局动作子集（解决 solver 可能忽略 actions(p,s) 的问题）。"
function build_global_actions(mdp_env)::Vector{Term}
    # 若任务抽取失败，则回退到原始全局 ground actions（可能很大）
    if !_task_ready()
        return Vector{Term}(collect(POMDPs.actions(mdp_env)))
    end

    o = ORDER_SYM[]::Symbol
    a = AGV_SYM[]::Symbol
    t = TRAY_SYM[]::Symbol

    # 关键地点集合（动作空间裁剪的核心）
    locs = Set{Symbol}([:floor_init])
    union!(locs, GRIPPER_STATIONS[])

    tray_slot = _tray_slot_in_init(mdp_env)
    if tray_slot !== nothing
        push!(locs, tray_slot)
    else
        union!(locs, TRAY_SLOTS[])
    end

    hd = _agv_home_dest(deepcopy(mdp_env.init))
    if hd !== nothing
        push!(locs, hd)
    else
        union!(locs, AGV_DESTS[])
    end

    # 为了应对 VLM/先验的不确定性：允许去任意 bin；但不会允许去无关 init_agvX（除 home）
    union!(locs, BIN_SLOTS[])

    # 目标槽位（放置用）
    for (_, sl) in TARGET_SLOT_FOR_PART[]
        push!(locs, sl)
    end

    acts = Term[]

    # move_floor：只在 locs 之间移动，显著减少 ground move 数量
    for l1 in locs, l2 in locs
        l1 == l2 && continue
        push!(acts, Compound(:move_floor, Term[Const(:floor), Const(l1), Const(l2)]))
    end

    # change_gripper：只在夹爪站
    for gs in GRIPPER_STATIONS[]
        push!(acts, Compound(:change_gripper, Term[Const(:floor), Const(gs)]))
    end

    # tray：只考虑目标托盘
    if tray_slot !== nothing
        push!(acts, Compound(:floor_pick_tray, Term[Const(:floor), Const(t), Const(tray_slot)]))
    else
        for ts in TRAY_SLOTS[]
            push!(acts, Compound(:floor_pick_tray, Term[Const(:floor), Const(t), Const(ts)]))
        end
    end
    push!(acts, Compound(:floor_place_tray, Term[Const(:floor), Const(t), Const(a), Const(o)]))

    # inspect：只在 bin 上（无意义的 inspect(location) 不加入动作空间）
    for b in BIN_SLOTS[]
        push!(acts, Compound(:inspect, Term[Const(:floor), Const(b)]))
    end

    # part pick：只考虑订单需要的零件，但允许在任意 bin 取（对抗先验错位）
    for p in REQUIRED_PARTS[]
        for b in BIN_SLOTS[]
            push!(acts, Compound(:floor_pick_part, Term[Const(:floor), Const(p), Const(b)]))
        end
    end

    # part place / QC：只考虑订单需要的零件与其目标槽位
    for (p, sl) in TARGET_SLOT_FOR_PART[]
        push!(acts, Compound(:floor_place_part, Term[Const(:floor), Const(p), Const(sl), Const(a), Const(o)]))
        push!(acts, Compound(:perform_quality_check, Term[Const(o), Const(p), Const(sl), Const(a)]))
    end

    # move_agv：只关心回到仓库提交（以及必要时回 home）
    for d1 in AGV_DESTS[]
        d1 == :warehouse && continue
        push!(acts, Compound(:move_agv, Term[Const(a), Const(d1), Const(:warehouse)]))
        push!(acts, Compound(:move_agv, Term[Const(a), Const(:warehouse), Const(d1)]))
    end

    # submit
    push!(acts, Compound(:submit_order, Term[Const(o), Const(a)]))

    return acts
end

function POMDPs.actions(p::KittingPOMDP)
    return isempty(GLOBAL_ACTIONS[]) ? POMDPs.actions(p.env_mdp) : GLOBAL_ACTIONS[]
end

function _priority_locs_from_problem(prob_env)::Set{Symbol}
    locs = Set{Symbol}()
    for obj in PDDL.get_objects(prob_env)
        name_sym = obj isa Symbol ? obj :
                   obj isa Const  ? obj.name :
                   Symbol(obj)
        name_str = String(name_sym)
        if startswith(name_str, "bin") ||
           startswith(name_str, "slot") ||
           startswith(name_str, "gripper_station") ||
           startswith(name_str, "init_agv") ||
           name_str == "warehouse" ||
           occursin(r"^agv\\d+_\\d+$", name_str)
            push!(locs, name_sym)
        end
    end
    return locs
end

function _extract_location_sets_from_problem(prob_env)
    grippers = Set{Symbol}()
    trayslots = Set{Symbol}()
    agvslots = Set{Symbol}()
    agvdests = Set{Symbol}()
    for obj in PDDL.get_objects(prob_env)
        name_sym = obj isa Symbol ? obj :
                   obj isa Const  ? obj.name :
                   Symbol(obj)
        name_str = String(name_sym)
        if startswith(name_str, "gripper_station")
            push!(grippers, name_sym)
        elseif startswith(name_str, "slot")
            push!(trayslots, name_sym)
        elseif occursin(r"^agv\\d+_\\d+$", name_str)
            push!(agvslots, name_sym)
        elseif startswith(name_str, "init_agv") || name_str == "warehouse"
            push!(agvdests, name_sym)
        end
    end
    return grippers, trayslots, agvslots, agvdests
end

function _extract_task_from_init(mdp_env)
    # 注意：order_needs_part / order_uses_tray 往往会被 PDDL.infer_static_fluents 判为 static，
    # 因而不会出现在 mdp_env.states.fluents 里。这里直接对这两个谓词做 grounding，
    # 再从 init state 读取为 true 的实例，避免任务抽取失败。
    s0 = deepcopy(mdp_env.init)

    # order_needs_part(o, p, slot)
    req = Dict{Symbol,Symbol}()
    o_sym = nothing
    for args in PDDL.groundargs(mdp_env.domain, s0, :order_needs_part)
        f = Compound(:order_needs_part, collect(Term, args))
        if s0[f]
            o  = args[1].name
            p  = args[2].name
            sl = args[3].name
            req[p] = sl
            o_sym = o_sym === nothing ? o : o_sym
        end
    end

    agv_sym = nothing
    tray_sym = nothing
    if o_sym !== nothing
        for args in PDDL.groundargs(mdp_env.domain, s0, :order_uses_tray)
            f = Compound(:order_uses_tray, collect(Term, args))
            if s0[f] && args[1].name == o_sym
                tray_sym = args[2].name
                agv_sym  = args[3].name
                break
            end
        end
    end

    return o_sym, agv_sym, tray_sym, req
end

_move_floor_dest(a::Term)::Union{Symbol,Nothing} =
    (a isa Compound && (a::Compound).name == :move_floor) ? (a::Compound).args[3].name : nothing

function _move_rank(dest::Symbol)::Int
    if dest in PRIORITY_LOCS[]
        return 0
    elseif dest in BIN_SLOTS[]
        return 1
    else
        return 2
    end
end

function _task_ready()::Bool
    return ORDER_SYM[] !== nothing && AGV_SYM[] !== nothing && TRAY_SYM[] !== nothing &&
           !isempty(REQUIRED_PARTS[]) && !isempty(TARGET_SLOT_FOR_PART[])
end

function _held_part(s::State)::Union{Symbol,Nothing}
    for p in REQUIRED_PARTS[]
        if s[holding_part_term(p)]
            return p
        end
    end
    return nothing
end

function _tray_on_agv(s::State)::Bool
    if AGV_SYM[] === nothing || TRAY_SYM[] === nothing
        return false
    end
    return s[on_agv_term(TRAY_SYM[]::Symbol, AGV_SYM[]::Symbol)]
end

function _part_placed(s::State, p::Symbol)::Bool
    slot = get(TARGET_SLOT_FOR_PART[], p, nothing)
    slot === nothing && return false
    return s[in_slot_term(p, slot)]
end

function _needed_parts(s::State)::Vector{Symbol}
    need = Symbol[]
    for p in REQUIRED_PARTS[]
        if !_part_placed(s, p)
            push!(need, p)
        end
    end
    return need
end

function _bin_of_part(s::State, p::Symbol)::Union{Symbol,Nothing}
    for b in BIN_SLOTS[]
        if s[part_on_term(p, b)]
            return b
        end
    end
    return nothing
end

function _agv_home_dest(s::State)::Union{Symbol,Nothing}
    if AGV_SYM[] === nothing
        return nothing
    end
    a = AGV_SYM[]::Symbol
    for d in AGV_DESTS[]
        if s[home_term(a, d)] && s[agv_at_term(a, d)]
            return d
        end
    end
    return nothing
end


# 说明：_agv_home_dest(s) 返回“AGV 当前所在且也是 home 的对接点”，
# 但当 AGV 被移动到 warehouse 后它会返回 nothing。这里补两个更稳健的 helper：
function _agv_home_loc(s::State)::Union{Symbol,Nothing}
    if AGV_SYM[] === nothing
        return nothing
    end
    a = AGV_SYM[]::Symbol
    for d in AGV_DESTS[]
        if s[home_term(a, d)]
            return d
        end
    end
    return nothing
end

function _agv_current_dest(s::State)::Union{Symbol,Nothing}
    if AGV_SYM[] === nothing
        return nothing
    end
    a = AGV_SYM[]::Symbol
    for d in AGV_DESTS[]
        if s[agv_at_term(a, d)]
            return d
        end
    end
    return nothing
end

function _tray_slot_in_state(s::State)::Union{Symbol,Nothing}
    if TRAY_SYM[] === nothing
        return nothing
    end
    t = TRAY_SYM[]::Symbol
    for ts in TRAY_SLOTS[]
        if s[tray_on_slot_term(t, ts)]
            return ts
        end
    end
    return nothing
end

function _all_required_parts_placed(s::State)::Bool
    _task_ready() || return false
    for p in REQUIRED_PARTS[]
        sl = get(TARGET_SLOT_FOR_PART[], p, :none)
        sl == :none && return false
        if !s[in_slot_term(p, sl)]
            return false
        end
    end
    return true
end

function _allowed_move_dests(s::State)::Set{Symbol}
    dests = Set{Symbol}()

    # 永远允许去夹爪站（可能需要换夹爪）
    union!(dests, GRIPPER_STATIONS[])

    # 若任务信息未加载，则退化：用 PRIORITY_LOCS + BIN_SLOTS（再由 PW 控分支）
    if !_task_ready()
        union!(dests, PRIORITY_LOCS[])
        union!(dests, BIN_SLOTS[])
        push!(dests, :floor_init)
        return dests
    end

    home = _agv_home_loc(s)
    tray_on = _tray_on_agv(s)
    heldp = _held_part(s)
    holding_tray = (TRAY_SYM[] !== nothing) && s[holding_tray_term(TRAY_SYM[]::Symbol)]
    tray_slot = _tray_slot_in_state(s)

    # Phase A：托盘未上 AGV —— 只允许“拿托盘/装托盘”相关移动
    if !tray_on
        if holding_tray
            # 拿着托盘：只需要去 AGV home 对接点装托盘
            home !== nothing && push!(dests, home)
        else
            # 没拿托盘：需要先拿到托盘夹爪，再去托盘槽位取托盘
            if s[has_tray_gripper_term()]
                tray_slot !== nothing && push!(dests, tray_slot)
            end
            # home 也允许（方便等待/对接）
            home !== nothing && push!(dests, home)
        end
        push!(dests, :floor_init)
        return dests
    end

    # Phase B：托盘已上 AGV —— 才进入“抓零件/放槽位/质检/提交”阶段

    # 若手里拿着 required part：只允许去它的目标槽位
    if heldp !== nothing
        sl = get(TARGET_SLOT_FOR_PART[], heldp, :none)
        sl != :none && push!(dests, sl)
        home !== nothing && push!(dests, home)   # 兜底：回 home 方便质检/调整
        push!(dests, :floor_init)
        return dests
    end

    # 未拿零件：只允许去“仍未放到位”的 required part 所在料箱
    for p in REQUIRED_PARTS[]
        sl = get(TARGET_SLOT_FOR_PART[], p, :none)
        sl == :none && continue
        if !s[in_slot_term(p, sl)]
            b = _bin_of_part(s, p)
            b !== nothing && push!(dests, b)
        end
    end

    # 质检/提交都依赖 AGV 在 home/warehouse，保留 home 与 floor_init 兜底
    home !== nothing && push!(dests, home)
    push!(dests, :floor_init)
    return dests
end

function _is_relevant_action(s::State, a::Term)::Bool
    a isa Compound || return true
    aname = (a::Compound).name

    # 如果还没有任务信息，先不剪枝非移动动作（只控 move_floor）
    if !_task_ready()
        return true
    end

    tray_on = _tray_on_agv(s)
    parts_placed = _all_required_parts_placed(s)
    holding_tray = (TRAY_SYM[] !== nothing) && s[holding_tray_term(TRAY_SYM[]::Symbol)]
    heldp = _held_part(s)

    # 信息动作（inspect）：只在“进入抓零件阶段”时开放，并且只允许对 bin 做 inspect
    if aname == :inspect
        loc = (a::Compound).args[2].name
        return tray_on && (loc in BIN_SLOTS[]) && (heldp === nothing) && !holding_tray
    end

    # 换夹爪：用阶段约束防止来回切换
    if aname == :change_gripper
        ge = s[gripper_empty_term()]
        if !ge
            return false
        end
        if !tray_on
            # 托盘阶段：只允许从 part->tray（拿托盘夹爪），拿到后不允许再切回
            return s[has_part_gripper_term()]
        else
            # 零件阶段：只允许从 tray->part（拿零件夹爪）
            return s[has_tray_gripper_term()]
        end
    end

    # move_agv：严格阶段化（避免 warehouse <-> home 来回抖动）
    if aname == :move_agv
        agv = (a::Compound).args[1].name
        agv == (AGV_SYM[]::Symbol) || return false
        d2  = (a::Compound).args[3].name

        home = _agv_home_loc(s)
        cur  = _agv_current_dest(s)

        # Phase A/B 未完成：必须保持/回到 home（否则无法 place_tray / place_part / QC）
        if !tray_on || !parts_placed
            return (home !== nothing) && (cur !== nothing) && (cur != home) && (d2 == home)
        end

        # Phase C 已完成装配：才允许去 warehouse（为 submit 做准备）
        return (cur !== nothing) && (cur != :warehouse) && (d2 == :warehouse)
    end

    # 只保留“对目标订单有意义”的动作实例
    if aname == :floor_pick_tray
        t = (a::Compound).args[2].name
        return !tray_on && t == (TRAY_SYM[]::Symbol)
    elseif aname == :floor_place_tray
        t = (a::Compound).args[2].name
        agv = (a::Compound).args[3].name
        o = (a::Compound).args[4].name
        return !tray_on && holding_tray &&
               t == (TRAY_SYM[]::Symbol) && agv == (AGV_SYM[]::Symbol) && o == (ORDER_SYM[]::Symbol)
    elseif aname == :floor_pick_part
        p = (a::Compound).args[2].name
        return tray_on && (p in REQUIRED_PARTS[]) && (heldp === nothing) && !holding_tray
    elseif aname == :floor_place_part
        p = (a::Compound).args[2].name
        slot = (a::Compound).args[3].name
        return tray_on && (p in REQUIRED_PARTS[]) && get(TARGET_SLOT_FOR_PART[], p, :none) == slot
    elseif aname == :perform_quality_check
        p = (a::Compound).args[2].name
        return tray_on && (p in REQUIRED_PARTS[]) && parts_placed
    elseif aname == :submit_order
        o = (a::Compound).args[1].name
        agv = (a::Compound).args[2].name
        return tray_on && parts_placed && o == (ORDER_SYM[]::Symbol) && agv == (AGV_SYM[]::Symbol)
    end

    return true
end

function _filter_actions_pw(s::State, acts::Vector{Term})::Vector{Term}
    # 统计该“宏观状态”在规划过程中的访问次数，用于近似 progressive widening。
    # 重要：belief 粒子在 part_on / good/bad 上高度分散，直接 hash(s) 几乎不会重复，
    # 会导致 v≈1 恒成立，从而每个状态只保留极少数 move_floor，表现为“随机游走”。
    # 因此这里用更粗粒度的 key（机器人位置 + 夹爪/托盘宏观状态）来累计访问次数。
    loc = _robot_loc(s)
    hp  = s[has_part_gripper_term()]
    ht  = s[has_tray_gripper_term()]
    ge  = s[gripper_empty_term()]
    ta  = _tray_on_agv(s)
    h = hash((loc, hp, ht, ge, ta))
    v = get!(PW_STATE_VISITS, h, 0) + 1
    PW_STATE_VISITS[h] = v

    non_move = Term[]
    move_floor = Term[]
    for a in acts
        if a isa Compound && (a::Compound).name == :move_floor
            push!(move_floor, a)
        else
            _is_relevant_action(s, a) && push!(non_move, a)
        end
    end

    allowed = _allowed_move_dests(s)
    move_floor = [a for a in move_floor if (_move_floor_dest(a) !== nothing && (_move_floor_dest(a)::Symbol) in allowed)]
    isempty(move_floor) && return non_move

    move_budget = min(length(move_floor), max(PW_MOVE_MIN, Int(floor(PW_K * v^PW_ALPHA))))
    sort!(move_floor, by=a -> begin
        d = _move_floor_dest(a)::Symbol
        return (_move_rank(d), string(a))
    end)

    return vcat(non_move, move_floor[1:move_budget])
end

function POMDPs.actions(p::KittingPOMDP, s::State)
    # 复用 SymbolicMDP 的 action cache：actions(mdp, s) 内部会用 a_cache 缓存 PDDL.available 结果
    acts = Vector{Term}(POMDPs.actions(p.env_mdp, s))
    return _filter_actions_pw(s, acts)
end

POMDPs.discount(p::KittingPOMDP) = POMDPs.discount(p.env_mdp)
POMDPs.isterminal(p::KittingPOMDP, s) = POMDPs.isterminal(p.env_mdp, s)
POMDPs.initialstate(p::KittingPOMDP) = Deterministic(deepcopy(p.env_mdp.init))

function POMDPs.transition(p::KittingPOMDP, s, a)
    # 复用缓存后的合法动作集合（避免反复 PDDL.available）
    legal = POMDPs.actions(p.env_mdp, s)
    is_legal = any(_act_term(act) == a for act in legal)
    if is_legal
        return POMDPs.transition(p.env_mdp, s, a)
    else
        return Deterministic(deepcopy(s))
    end
end

########################
# Reward / Observation  #
########################

const STEP_COST           = -1.0
const MOVE_FLOOR_COST     = -5.0
const MOVE_AGV_COST       = -6.0
const PLACE_TRAY_REWARD   = 1.0
const PLACE_PART_REWARD   = 1.0
const PICK_TRAY_REWARD    = 1.0
const PICK_PART_REWARD    = 1.0
const CHANGE_GRIPPER_REWARD = 0.0
const INSPECT_ACTION_COST = -2.0
const QC_ACTION_COST      = -3.0
const REPAIR_ACTION_COST  = 0.0
const INVALID_ACTION_PEN  = -10.0
const GOAL_REWARD         = 200.0

# 任务相关的额外奖励塑形：只奖励“抓对件/放对槽位/上对托盘/提交”
const PICK_REQUIRED_BONUS      = 8.0
const PICK_IRRELEVANT_PEN      = -8.0
const PLACE_REQUIRED_BONUS     = 20.0
const PLACE_WRONG_SLOT_PEN     = -10.0
const PLACE_TRAY_BONUS         = 10.0
const SUBMIT_BONUS             = 300.0

POMDPs.initialobs(::KittingPOMDP, s) = QualityObs(:none, QNone)

function POMDPs.reward(p::KittingPOMDP, s::State, a::Term, sp::State)
    # 默认每步代价为 STEP_COST，但对关键动作做“动作特定代价/奖励”以塑形：
    # - move_floor: 代价更高（见 MOVE_FLOOR_COST），减少无意义走动
    # - floor_place_tray / floor_place_part: 给正奖励（+1），鼓励尽快完成关键装配动作
    r = STEP_COST
    if a isa Compound
        aname = (a::Compound).name
        if aname == :move_floor
            r = MOVE_FLOOR_COST
        elseif aname == :move_agv
            r = MOVE_AGV_COST
        elseif aname == :floor_place_tray
            r = PLACE_TRAY_REWARD
        elseif aname == :floor_place_part
            r = PLACE_PART_REWARD
        elseif aname == :floor_pick_tray
            r = PICK_TRAY_REWARD
        elseif aname == :floor_pick_part
            r = PICK_PART_REWARD
        elseif aname == :change_gripper
            r = CHANGE_GRIPPER_REWARD
        end
    end

    # 复用缓存后的合法动作集合（避免反复 PDDL.available）
    legal_actions = POMDPs.actions(p.env_mdp, s)
    is_legal = any(_act_term(act) == a for act in legal_actions)
    if !is_legal
        return r + INVALID_ACTION_PEN
    end

    if is_inspect_action(a)
        r += INSPECT_ACTION_COST
    elseif is_qc_action(a)
        r += QC_ACTION_COST
    end

    # 任务相关 shaping：避免“抓错件也得分”的策略缺陷
    if _task_ready() && a isa Compound
        aname = (a::Compound).name
        # 额外策略 shaping：强制“先装托盘再抓零件”，并减少托盘之前的无效移动/感知
        if aname == :move_floor
            d = _move_floor_dest(a)
            if d !== nothing && !_tray_on_agv(s) && (d::Symbol) in BIN_SLOTS[]
                r += -2.0  # 托盘未上车前去料箱通常是浪费时间
            end
        elseif aname == :inspect
            if !_tray_on_agv(s)
                r += -1.0  # 托盘阶段不鼓励 inspect
            end
        elseif aname == :move_agv
            # 在装配完成前移动 AGV 会直接破坏 place_tray/place_part/QC 的前置条件
            if !_tray_on_agv(s) || !_all_required_parts_placed(s)
                r += -25.0
            end
        elseif aname == :change_gripper
            if !_tray_on_agv(s) && s[has_part_gripper_term()] && s[gripper_empty_term()]
                r += 2.0  # 轻度鼓励切换到托盘夹爪（抵消一部分动作成本）
            end
        elseif aname == :floor_pick_tray
            if !_tray_on_agv(s)
                r += 2.0
            end
        end

        if aname == :floor_pick_part
            part = (a::Compound).args[2].name
            if !_tray_on_agv(s)
                # 关键：托盘未上 AGV 时抓零件会导致“抱着零件无法装托盘”的死局，强惩罚以逼迫先装托盘
                r += -50.0
            end
            r += (part in REQUIRED_PARTS[]) ? PICK_REQUIRED_BONUS : PICK_IRRELEVANT_PEN
        elseif aname == :floor_place_part
            part = (a::Compound).args[2].name
            slot = (a::Compound).args[3].name
            if (part in REQUIRED_PARTS[]) && get(TARGET_SLOT_FOR_PART[], part, :none) == slot
                r += PLACE_REQUIRED_BONUS
            else
                r += PLACE_WRONG_SLOT_PEN
            end
        elseif aname == :floor_place_tray
            t = (a::Compound).args[2].name
            if TRAY_SYM[] !== nothing && t == (TRAY_SYM[]::Symbol)
                r += PLACE_TRAY_BONUS
            end
        elseif aname == :submit_order
            r += SUBMIT_BONUS
        end
    end

    was_terminal = POMDPs.isterminal(p.env_mdp, s)
    now_terminal = POMDPs.isterminal(p.env_mdp, sp)
    if now_terminal && !was_terminal
        r += GOAL_REWARD
    end
    return r
end

function POMDPs.observation(p::KittingPOMDP, a::Term, sp::State)
    if is_qc_action(a)
        part = checked_part(a)
        is_good = sp[good_term(part)]
        if is_good
            return SparseCat([QualityObs(part, QGood), QualityObs(part, QBad)], [0.9, 0.1])
        else
            return SparseCat([QualityObs(part, QBad), QualityObs(part, QGood)], [0.9, 0.1])
        end
    elseif is_inspect_action(a)
        loc = inspected_location(a)
        if !(loc in BIN_SLOTS[])
            return SparseCat([InspectObs(loc, :none)], [1.0])
        end
        true_part = part_at_bin(sp, loc)
        if true_part == :none
            return SparseCat([InspectObs(loc, :none)], [1.0])
        else
            # 简化噪声模型：99% 看到真实零件，1% 漏检成 none
            return SparseCat([InspectObs(loc, true_part), InspectObs(loc, :none)], [0.99, 0.01])
        end
    else
        return SparseCat([QualityObs(:none, QNone)], [1.0])
    end
end

########################
# Initial belief (粒子) #
########################

sample_symbol(dist::Dict{Symbol, Float64}) = begin
    r = rand()
    acc = 0.0
    last_key = first(keys(dist))
    for (k, p) in dist
        acc += p
        if r <= acc + 1e-12
            return k
        end
        last_key = k
    end
    return last_key
end

function extract_parts_and_bins(mdp_env)::Tuple{Vector{Symbol},Vector{Symbol}}
    parts = Set{Symbol}()
    bins  = Set{Symbol}()
    for f in mdp_env.states.fluents
        if f isa Compound && (f::Compound).name == :part_on
            push!(parts, (f::Compound).args[1].name)
            push!(bins,  (f::Compound).args[2].name)
        elseif f isa Compound && ((f::Compound).name == :good || (f::Compound).name == :bad)
            push!(parts, (f::Compound).args[1].name)
        end
    end
    return sort!(collect(parts)), sort!(collect(bins))
end

function guess_loc_from_state(s::State, p::Symbol, bins::Vector{Symbol})::Union{Symbol,Nothing}
    for b in bins
        if s[part_on_term(p, b)]
            return b
        end
    end
    return nothing
end

function prior_loc_from_vlm(mdp_vlm, parts::Vector{Symbol}, bins::Vector{Symbol};
                            p_high::Float64=0.8)
    s_vlm = rand(POMDPs.initialstate(mdp_vlm))
    prior_loc = Dict{Symbol, Dict{Symbol, Float64}}()
    for p in parts
        b_v = guess_loc_from_state(s_vlm, p, bins)
        probs = Dict{Symbol, Float64}()
        if b_v === nothing || length(bins) <= 1
            for b in bins
                probs[b] = 1.0 / length(bins)
            end
        else
            p_rest = (1.0 - p_high) / (length(bins) - 1)
            for b in bins
                probs[b] = (b == b_v) ? p_high : p_rest
            end
        end
        prior_loc[p] = probs
    end
    return prior_loc
end

function prior_good_from_vlm(mdp_vlm, parts::Vector{Symbol}; p_high::Float64=0.8)
    s_vlm = rand(POMDPs.initialstate(mdp_vlm))
    prior_good = Dict{Symbol, Float64}()
    for p in parts
        if s_vlm[good_term(p)]
            prior_good[p] = p_high
        elseif s_vlm[bad_term(p)]
            prior_good[p] = 1.0 - p_high
        else
            prior_good[p] = 0.5
        end
    end
    return prior_good
end

function sample_initial_state(mdp_env, prior_loc, prior_good)
    s = deepcopy(rand(POMDPs.initialstate(mdp_env)))
    for p in PARTS[]
        b_sample = sample_symbol(prior_loc[p])
        set_part_loc!(s, p, b_sample)
        set_quality!(s, p, rand() < prior_good[p])
    end
    return s
end

function make_initial_belief(mdp_env, mdp_vlm; n_particles::Int=300)
    prior_loc  = prior_loc_from_vlm(mdp_vlm, PARTS[], BIN_SLOTS[]; p_high=0.8)
    prior_good = prior_good_from_vlm(mdp_vlm, PARTS[]; p_high=0.8)
    particles = State[]
    for _ in 1:n_particles
        push!(particles, sample_initial_state(mdp_env, prior_loc, prior_good))
    end
    return ParticleFilters.ParticleCollection(particles)
end

########################
# Env loading / main    #
########################

function load_models()
    domain_path = joinpath(@__DIR__, "domain.pddl")
    real_path   = joinpath(@__DIR__, "p_real.pddl")
    vlm_path    = joinpath(@__DIR__, "p_vlm.pddl")

    println("加载 PDDL 文件：")
    println("  domain : $domain_path")
    println("  real   : $real_path")
    println("  vlm    : $vlm_path")

    domain   = PDDL.load_domain(domain_path)
    prob_env = PDDL.load_problem(real_path)
    prob_vlm = PDDL.load_problem(vlm_path)

    mdp_env = SymbolicMDP(domain, prob_env; discount=0.95)
    mdp_vlm = SymbolicMDP(domain, prob_vlm; discount=0.95)

    parts, bins = extract_parts_and_bins(mdp_env)
    PARTS[] = parts
    BIN_SLOTS[] = bins
    PRIORITY_LOCS[] = _priority_locs_from_problem(prob_env)
    GRIPPER_STATIONS[], TRAY_SLOTS[], AGV_SLOTS[], AGV_DESTS[] = _extract_location_sets_from_problem(prob_env)

    # 抽取任务（目标订单/AGV/托盘/需要的零件及其目标槽位）
    o, agv, tray, req = _extract_task_from_init(mdp_env)
    ORDER_SYM[] = o
    AGV_SYM[] = agv
    TRAY_SYM[] = tray
    REQUIRED_PARTS[] = Set(keys(req))
    TARGET_SLOT_FOR_PART[] = req
    if _task_ready()
        println("✓ 任务抽取: order=$(ORDER_SYM[]), agv=$(AGV_SYM[]), tray=$(TRAY_SYM[])")
        println("✓ 订单需求零件: ", collect(REQUIRED_PARTS[]))
        println("✓ 目标槽位映射: ", TARGET_SLOT_FOR_PART[])
    else
        @warn "未能从初始状态抽取订单需求（order_needs_part/order_uses_tray），将退化为通用过滤策略。"
    end

    # 为 POMCP 构建“裁剪后的全局动作集合”（避免 solver 使用全局动作导致随机游走）
    GLOBAL_ACTIONS[] = build_global_actions(mdp_env)
    println("✓ 裁剪后全局动作数: ", length(GLOBAL_ACTIONS[]))

    println("✓ 真实 MDP 类型: ", typeof(mdp_env))
    println("✓ VLM   MDP 类型: ", typeof(mdp_vlm))
    println("✓ 抽取到零件数: ", length(PARTS[]), "；bin_slot 数: ", length(BIN_SLOTS[]))
    return domain, mdp_env, mdp_vlm
end

"防止粒子滤波退化导致 belief 为空，从而让 POMCP 在 rand(b) 时崩溃。"
struct RobustUpdater{U} <: POMDPs.Updater
    base::U
end

function POMDPs.update(up::RobustUpdater, b, a, o)
    b2 = try
        POMDPs.update(up.base, b, a, o)
    catch err
        @warn "Belief update 失败，回退到上一步 belief（避免崩溃）" exception=(err, catch_backtrace()) a=a o=o
        return b
    end
    if ParticleFilters.n_particles(b2) == 0
        @warn "Particle depletion：更新后粒子为 0，回退到上一步 belief（避免崩溃）" a=a o=o
        return b
    end
    return b2
end

function main_pomdp()
    t_total_start = time()
    println("="^60)
    println("ARIAC kitting：p_real 环境 + p_vlm 先验 的 POMDP (POMCP)")
    println("="^60 * "\n")

    _, mdp_env, mdp_vlm = load_models()

    println("\n构建初始 belief（粒子）...")
    init_belief = make_initial_belief(mdp_env, mdp_vlm; n_particles=200)
    println("粒子数: ", ParticleFilters.n_particles(init_belief))

    pomdp = KittingPOMDP(mdp_env, init_belief)
    all_acts = collect(POMDPs.actions(pomdp))
    n_inspect = sum(is_inspect_action, all_acts)
    n_qc      = sum(is_qc_action, all_acts)
    n_repair  = 0
    # 注意：actions(pomdp) 是“全局 ground 动作集合”（包含很多在当前状态不满足 precondition 的实例）
    # 真正用于规划的是 actions(pomdp, s)：它会调用 cached 的 PDDL.available，只返回满足 precondition 的动作。
    println("\n全局动作数(ground, 未做 precondition 过滤): ", length(all_acts))
    println("  其中 inspect 动作数量: ", n_inspect)
    println("  其中 quality_check 动作数量: ", n_qc)
    println("  其中 repair  动作数量: ", n_repair)

    # 额外打印：初始状态下“满足 precondition 的合法动作数”（以及 PW 过滤后用于规划的动作数）
    s0_env = deepcopy(rand(POMDPs.initialstate(mdp_env)))
    n_legal_raw = length(collect(POMDPs.actions(mdp_env, s0_env))) # cached available
    n_legal_pw  = length(collect(POMDPs.actions(pomdp, s0_env)))   # cached available + PW 过滤
    println("\n初始状态合法动作数(满足 precondition): ", n_legal_raw)
    println("初始状态合法动作数(渐进扩展后用于规划): ", n_legal_pw)

    println("\n使用 POMCP 进行 POMDP 在线规划 ...")
    solver = POMCPSolver(
        tree_queries  = 2000,
        max_depth     = 40,
        c             = 1.0,
        estimate_value = 0.0,
    )
    t_solve = @elapsed policy = solve(solver, pomdp)
    println("✓ POMCP 求解完成。\n")

    println("（提示）本 demo 的 stepthrough 依赖 POMDPSimulators；为避免 Julia 版本/依赖冲突，这里不再自动模拟轨迹。")
    println("你仍可在 Python 侧通过 `JuliaInfoClient.solve_*` 调用小 POMDP 触发器。")
    t_total = time() - t_total_start
    println("\n耗时统计：")
    println("  solve() 耗时: ", round(t_solve, digits=3), " 秒")
    println("  总耗时      : ", round(t_total, digits=3), " 秒")
end

############################################################
# 3) Hybrid controller API (Python/UP ground truth + Julia info)
#    - 只提供“小信息POMDP”触发点决策：inspect / QC / continue
#    - 以及从 p_vlm 初始化先验、与基于观测的轻量 belief 更新
############################################################

"""
    init_prior(domain_path, p_vlm_path; p_high=0.8)

从 `p_vlm.pddl` 初始化先验：
- `prior_loc[p][b] = P(loc(p)=b)`（对每个 part 一个离散分布）
- `prior_good[p]   = P(good(p))`

返回 NamedTuple：`(; prior_loc, prior_good, parts, bins)`, keys 为 String，便于 Python 互操作。
"""
function init_prior(domain_path::AbstractString, p_vlm_path::AbstractString; p_high::Float64=0.8)
    domain = PDDL.load_domain(domain_path)
    prob_vlm = PDDL.load_problem(p_vlm_path)
    s = PDDL.initstate(domain, prob_vlm)

    # Extract parts/bins robustly from grounded fluents in the domain
    parts = Set{String}()
    bins  = Set{String}()
    for args in PDDL.groundargs(domain, s, :part_on)
        push!(parts, String(args[1].name))
        push!(bins,  String(args[2].name))
    end
    for fname in (:good, :bad)
        for args in PDDL.groundargs(domain, s, fname)
            push!(parts, String(args[1].name))
        end
    end
    parts = sort!(collect(parts))
    bins  = sort!(collect(bins))

    prior_loc  = Dict{String, Dict{String, Float64}}()
    prior_good = Dict{String, Float64}()

    # Helper: find argmax location in VLM state
    for p in parts
        # location prior
        probs = Dict{String, Float64}()
        b_vlm = nothing
        for b in bins
            if s[part_on_term(Symbol(p), Symbol(b))]
                b_vlm = b
                break
            end
        end
        if b_vlm === nothing || length(bins) <= 1
            for b in bins
                probs[b] = 1.0 / max(1, length(bins))
            end
        else
            p_rest = (1.0 - p_high) / (length(bins) - 1)
            for b in bins
                probs[b] = (b == b_vlm) ? p_high : p_rest
            end
        end
        prior_loc[p] = probs

        # quality prior
        if s[good_term(Symbol(p))]
            prior_good[p] = p_high
        elseif s[bad_term(Symbol(p))]
            prior_good[p] = 1.0 - p_high
        else
            prior_good[p] = 0.5
        end
    end

    return (; prior_loc, prior_good, parts, bins)
end

"""
    update_belief_loc_target(prior_loc, target_part, inspected_location, saw_target;
                             p_detect=0.99)

对 `target_part` 的位置分布做一次 Bayes 更新（只用“是否看到目标件”的二值观测）。
"""
function update_belief_loc_target(prior_loc::Dict{String,Dict{String,Float64}},
                                  target_part::AbstractString,
                                  inspected_location::AbstractString,
                                  saw_target::Bool; p_detect::Float64=0.99)
    dist = get(prior_loc, String(target_part), Dict{String,Float64}())
    isempty(dist) && return prior_loc
    loc = String(inspected_location)

    # Likelihood model: no false positives; miss with (1-p_detect) if actually there
    for (b, p) in collect(dist)
        if b == loc
            dist[b] = saw_target ? (p * p_detect) : (p * (1.0 - p_detect))
        else
            dist[b] = saw_target ? 0.0 : p
        end
    end

    z = sum(values(dist))
    if z <= 0
        # Degenerate: fall back to uniform
        n = max(1, length(dist))
        for b in keys(dist)
            dist[b] = 1.0 / n
        end
    else
        for b in keys(dist)
            dist[b] /= z
        end
    end

    prior_loc[String(target_part)] = dist
    return prior_loc
end

# PythonCall/juliacall interop helper:
# Python `dict` often arrives as `PythonCall.PyDict{Any,Any}` (or `Dict{Any,Any}`),
# which will not dispatch to the typed method above. Provide a permissive wrapper
# that coerces into `Dict{String,Dict{String,Float64}}` first.
function update_belief_loc_target(prior_loc_any,
                                  target_part::AbstractString,
                                  inspected_location::AbstractString,
                                  saw_target::Bool; p_detect::Float64=0.99)
    prior_loc = prior_loc_any isa Dict{String,Dict{String,Float64}} ?
        prior_loc_any : begin
            coerced = Dict{String,Dict{String,Float64}}()
            for (p, dist_any) in pairs(prior_loc_any)
                dist = Dict{String,Float64}()
                for (b, pr) in pairs(dist_any)
                    dist[String(b)] = Float64(pr)
                end
                coerced[String(p)] = dist
            end
            coerced
        end

    # Call the typed method (avoid recursion via invoke)
    return invoke(update_belief_loc_target,
                  Tuple{Dict{String,Dict{String,Float64}},AbstractString,AbstractString,Bool},
                  prior_loc, target_part, inspected_location, saw_target; p_detect=p_detect)
end

"""
    update_belief_quality(p_good, obs_label; p_correct=0.9)

对二元质量变量 good/bad 做一次 Bayes 更新。
"""
function update_belief_quality(p_good::Float64, obs_label::AbstractString; p_correct::Float64=0.9)
    p_good = clamp(p_good, 0.0, 1.0)
    lbl = String(obs_label)
    if lbl == "good"
        # P(o=good|good)=p_correct, P(o=good|bad)=1-p_correct
        num = p_correct * p_good
        den = num + (1.0 - p_correct) * (1.0 - p_good)
        return den <= 0 ? p_good : (num / den)
    elseif lbl == "bad"
        # P(o=bad|bad)=p_correct, P(o=bad|good)=1-p_correct
        num = (1.0 - p_correct) * p_good
        den = num + p_correct * (1.0 - p_good)
        return den <= 0 ? p_good : (num / den)
    else
        return p_good
    end
end

# -----------------------
# Small trigger POMDPs
# -----------------------

struct LocTriggerPOMDP <: POMDP{Int,Int,Bool}
    probs::Vector{Float64}      # belief over true location index
    chosen_idx::Int             # MAP-chosen bin index for continue
    inspect_cost::Float64
    miss_penalty::Float64
    p_detect::Float64
end

POMDPs.discount(::LocTriggerPOMDP) = 1.0
POMDPs.actions(m::LocTriggerPOMDP) = collect(1:(length(m.probs) + 1)) # 1..K inspect_i, K+1 continue
POMDPs.initialstate(m::LocTriggerPOMDP) = Deterministic(argmax(m.probs)) # placeholder (not used directly)
POMDPs.isterminal(::LocTriggerPOMDP, s::Int) = false

function POMDPs.transition(m::LocTriggerPOMDP, s::Int, a::Int)
    # inspect does not change hidden location; continue keeps terminal handled via depth
    return Deterministic(s)
end

function POMDPs.observation(m::LocTriggerPOMDP, a::Int, sp::Int)
    k = length(m.probs)
    if a <= k
        # observe saw_target? (true/false)
        if sp == a
            return SparseCat(Bool[true, false], [m.p_detect, 1.0 - m.p_detect])
        else
            return SparseCat(Bool[false], [1.0])
        end
    else
        return SparseCat(Bool[false], [1.0])
    end
end

function POMDPs.reward(m::LocTriggerPOMDP, s::Int, a::Int, sp::Int)
    k = length(m.probs)
    if a <= k
        return -m.inspect_cost
    else
        return (sp == m.chosen_idx) ? 0.0 : -m.miss_penalty
    end
end

struct QualityTriggerPOMDP <: POMDP{Bool,Symbol,Symbol}
    p_good::Float64
    qc_cost::Float64
    repair_cost::Float64
    fail_penalty::Float64
    p_correct::Float64
end

POMDPs.discount(::QualityTriggerPOMDP) = 1.0
POMDPs.actions(::QualityTriggerPOMDP) = Symbol[:qc, :continue]
POMDPs.initialstate(m::QualityTriggerPOMDP) = Deterministic(m.p_good >= 0.5)
POMDPs.isterminal(::QualityTriggerPOMDP, s::Bool) = false

function POMDPs.transition(m::QualityTriggerPOMDP, s::Bool, a::Symbol)
    return Deterministic(s)
end

function POMDPs.observation(m::QualityTriggerPOMDP, a::Symbol, sp::Bool)
    if a == :qc
        # obs ∈ {:good,:bad}
        if sp
            return SparseCat(Symbol[:good, :bad], [m.p_correct, 1.0 - m.p_correct])
        else
            return SparseCat(Symbol[:bad, :good], [m.p_correct, 1.0 - m.p_correct])
        end
    else
        return SparseCat(Symbol[:none], [1.0])
    end
end

function POMDPs.reward(m::QualityTriggerPOMDP, s::Bool, a::Symbol, sp::Bool)
    if a == :qc
        return -m.qc_cost
    elseif a == :continue
        return sp ? 0.0 : -m.fail_penalty
    else
        return 0.0
    end
end

"""
    solve_trigger(trigger_type; kwargs...)

统一入口：根据触发点上下文返回 `action_str`（String）。
支持 trigger_type:\n
- \"pre_pick\": 位置不确定 -> inspect(bin_i) 或 continue\n
- \"after_place\" / \"before_submit\": 质量不确定 -> QC / continue\n

kwargs 约定（从 Python 传入）：\n
pre_pick:\n
- part::String, robot::String, candidate_bins::Vector{String}, loc_probs::Vector{Float64}\n
- inspect_cost, miss_penalty, p_detect, tree_queries, max_depth, c\n
quality:\n
- order, part, slot, agv (for action string)\n
- p_good, qc_cost, repair_cost, fail_penalty, p_correct, tree_queries, max_depth, c\n
"""
function solve_trigger(trigger_type::AbstractString; kwargs...)
    t = String(trigger_type)
    if t == "pre_pick"
        robot = String(get(kwargs, :robot, "floor"))
        bins  = Vector{String}(get(kwargs, :candidate_bins, String[]))
        probs = Vector{Float64}(get(kwargs, :loc_probs, Float64[]))
        isempty(bins) && return "continue"
        isempty(probs) && (probs = fill(1.0 / length(bins), length(bins)))

        inspect_cost = Float64(get(kwargs, :inspect_cost, 2.0))
        miss_penalty = Float64(get(kwargs, :miss_penalty, 25.0))
        p_detect     = Float64(get(kwargs, :p_detect, 0.99))
        tree_queries = Int(get(kwargs, :tree_queries, 200))
        max_depth    = Int(get(kwargs, :max_depth, 3))
        c            = Float64(get(kwargs, :c, 1.0))

        chosen_idx = argmax(probs)
        pomdp = LocTriggerPOMDP(probs, chosen_idx, inspect_cost, miss_penalty, p_detect)
        solver = POMCPSolver(tree_queries=tree_queries, max_depth=max_depth, c=c, estimate_value=0.0)
        policy = solve(solver, pomdp)

        # Build particle belief from probs
        n = max(50, min(200, tree_queries))
        parts = Int[]
        for i in 1:n
            push!(parts, sample(1:length(probs), Weights(probs)))
        end
        b = ParticleFilters.ParticleCollection(parts)
        a = action(policy, b)

        if a <= length(bins)
            return "inspect($robot, $(bins[a]))"
        else
            return "continue"
        end

    elseif t == "after_place" || t == "before_submit"
        order = String(get(kwargs, :order, ""))
        part  = String(get(kwargs, :part, ""))
        slot  = String(get(kwargs, :slot, ""))
        agv   = String(get(kwargs, :agv, ""))

        p_good       = Float64(get(kwargs, :p_good, 0.5))
        qc_cost      = Float64(get(kwargs, :qc_cost, 3.0))
        repair_cost  = Float64(get(kwargs, :repair_cost, 5.0))
        fail_penalty = Float64(get(kwargs, :fail_penalty, 300.0))
        p_correct    = Float64(get(kwargs, :p_correct, 0.9))
        tree_queries = Int(get(kwargs, :tree_queries, 200))
        max_depth    = Int(get(kwargs, :max_depth, 3))
        c            = Float64(get(kwargs, :c, 1.0))

        pomdp = QualityTriggerPOMDP(p_good, qc_cost, repair_cost, fail_penalty, p_correct)
        solver = POMCPSolver(tree_queries=tree_queries, max_depth=max_depth, c=c, estimate_value=0.0)
        policy = solve(solver, pomdp)

        # Particles from p_good
        n = max(50, min(200, tree_queries))
        parts = Bool[]
        for _ in 1:n
            push!(parts, rand() < p_good)
        end
        b = ParticleFilters.ParticleCollection(parts)
        a = action(policy, b)

        if a == :qc
            isempty(order) && return "continue"
            return "perform_quality_check($order, $part, $slot, $agv)"
        else
            return "continue"
        end
    else
        return "continue"
    end
end

end # module AriacKitting

# If run as a script, execute the original demo; when included (e.g., via juliacall), do nothing.
if abspath(PROGRAM_FILE) == @__FILE__
    Base.invokelatest(AriacKitting.main_pomdp)
end