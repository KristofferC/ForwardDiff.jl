immutable GradientNum{N,T,C} <: ForwardDiffNum{N,T,C}
    value::T
    grad::C
    GradientNum(value::T, grad::NTuple{N,T}) = new(value, grad)
    GradientNum(value::T, grad::Vector{T}) = new(value, grad)
    GradientNum(value, grad::Tuple) = GradientNum{N,T,C}(convert(T,value), convert(NTuple{N,T}, grad))
    GradientNum(value, grad::Vector) = GradientNum{N,T,C}(convert(T,value), convert(Vector{T}, grad))
end

typealias GradNumTup{N,T} GradientNum{N,T,NTuple{N,T}}
typealias GradNumVec{N,T} GradientNum{N,T,Vector{T}}

function GradientNum{N,T}(value, grad::NTuple{N,T})
    S = promote_type(typeof(value), T)
    return GradientNum{N,S,NTuple{N,S}}(convert(S, value), convert(NTuple{N,S},grad))
end

GradientNum{N,T}(value::T, grad::NTuple{N,T}) = GradientNum{N,T,NTuple{N,T}}(value, grad)
GradientNum{T}(value::T, grad::T...) = GradientNum(value, grad)

##############################
# Utility/Accessor Functions #
##############################
value(g::GradientNum) = g.value

grad(g::GradientNum) = g.grad
hess(g::GradientNum) = error("GradientNums do not store Hessian values")
tens(g::GradientNum) = error("GradientNums do not store tensor values")

eltype{N,T,C}(::Type{GradientNum{N,T,C}}) = T
eltype{N,T}(::GradientNum{N,T}) = T

npartials{N,T,C}(::Type{GradientNum{N,T,C}}) = N
npartials{N}(::GradientNum{N}) = N

zero_partials{N,T,C<:Tuple}(::Type{GradientNum{N,T,C}}) = zero_tuple(C)
zero_partials{N,T,C<:Vector}(::Type{GradientNum{N,T,C}}) = zeros(T, N)
rand_partials{N,T,C<:Tuple}(::Type{GradientNum{N,T,C}}) = rand_tuple(C)
rand_partials{N,T,C<:Vector}(::Type{GradientNum{N,T,C}}) = rand(T, N)

#####################
# Generic Functions #
#####################
isconstant(g::GradientNum) = any(x -> x == 0, grad(g))
isconstant(::GradientNum{0}) = true

==(a::GradientNum, b::GradientNum) = value(a) == value(b) && grad(a) == grad(b)

isequal(a::GradientNum, b::GradientNum) = isequal(value(a), value(b)) && isequal(grad(a), grad(b))

hash(g::GradientNum) = isconstant(g) ? hash(value(g)) : hash(value(g), hash(grad(g)))
hash(g::GradientNum, hsh::Uint64) = hash(hash(g), hsh)

read_partials{N,T}(io::IO, n::Int, ::Type{NTuple{N,T}}) = ntuple(n->read(io, T), Val{N})
read_partials{T}(io::IO, n::Int, ::Type{Vector{T}}) = [read(io, T) for i in 1:n]

function read{N,T,C}(io::IO, ::Type{GradientNum{N,T,C}})
    value = read(io, T)
    partials = read_partials(io, N, C)
    return GradientNum{N,T,C}(value, partials)
end

function write(io::IO, g::GradientNum)
    write(io, value(g))
    for du in grad(g)
        write(io, du)
    end
end

########################
# Conversion/Promotion #
########################
zero{N,T,C}(G::Type{GradientNum{N,T,C}}) = G(zero(T), zero_partials(G))
one{N,T,C}(G::Type{GradientNum{N,T,C}}) = G(one(T), zero_partials(G))
rand{N,T,C}(G::Type{GradientNum{N,T,C}}) = G(rand(T), rand_partials(G))

for F in (:GradNumVec, :GradNumTup)
    @eval begin
        convert{N,T}(::Type{$(F){N,T}}, x::Real) = $(F){N,T}(x, zero_partials($(F){N,T}))
        convert{N,A,B}(::Type{$(F){N,A}}, g::$(F){N,B}) = $(F){N,A}(value(g), grad(g))
        convert{N,T}(::Type{$(F){N,T}}, g::$(F){N,T}) = g

        promote_rule{N,A,B}(::Type{$(F){N,A}}, ::Type{$(F){N,B}}) = $(F){N,promote_type(A,B)}
        promote_rule{N,A,B}(::Type{$(F){N,A}}, ::Type{B}) = $(F){N,promote_type(A,B)}
    end
end

convert{T<:Real}(::Type{T}, g::GradientNum{0}) = convert(T, value(g))
convert{T<:Real}(::Type{T}, g::GradientNum) = isconstant(g) ? convert(T, value(g)) : throw(InexactError())
convert(::Type{GradientNum}, g::GradientNum) = g
convert(::Type{GradientNum}, x::Real) = GradientNum(x)

#########################
# Math with GradientNum #
#########################
# helper function to force use of NaNMath 
# functions in derivative calculations
function to_nanmath(x::Expr)
    if x.head == :call
        funsym = Expr(:.,:NaNMath,Base.Meta.quot(x.args[1]))
        return Expr(:call,funsym,[to_nanmath(z) for z in x.args[2:end]]...)
    else
        return Expr(:call,[to_nanmath(z) for z in x.args]...)
    end
end
to_nanmath(x) = x

abs(g::GradientNum) = (value(g) >= 0) ? g : -g
abs2(g::GradientNum) = g*g

include("gradnum_math/gradtup_math.jl")
include("gradnum_math/gradvec_math.jl")
