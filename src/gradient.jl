########################
# @gradient!/@gradient #
########################

const GRADIENT_KWARG_ORDER = (:all, :chunk, :input_length, :multithread)
const GRADIENT_F_KWARG_ORDER =  (:all, :chunk, :input_length, :multithread, :output_mutates)

macro gradient!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, GRADIENT_KWARG_ORDER)
    return esc(:(ForwardDiff.gradient!($(args...), $(arranged_kwargs...))))
end

macro gradient(args...)
    args, kwargs = separate_kwargs(args)
    if length(args) == 1
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, GRADIENT_F_KWARG_ORDER)
    else
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, GRADIENT_KWARG_ORDER)
    end
    return esc(:(ForwardDiff.gradient($(args...), $(arranged_kwargs...))))
end

######################
# gradient!/gradient #
######################

@generated function gradient!{S,A,L}(f, output::Vector{S}, x::Vector, ::Type{Val{A}}, N::DataType, ::Type{Val{L}}, M::DataType)
    return quote
        val, grad = call_gradient!(f, output, x, N, Val{$(L == nothing ? :(length(x)) : L)}, M)
        return $(A ? :(val::S, output::Vector{S}) : :(output::Vector{S}))
    end
end

function gradient(f, x::Vector, A::DataType, N::DataType, L::DataType, M::DataType)
    return gradient!(f, similar(x), x, A, N, L, M)
end

@generated function gradient{A,output_mutates}(f, ::Type{Val{A}}, N::DataType, L::DataType, M::DataType, ::Type{Val{output_mutates}})
    R = A ? :(Tuple{S,Vector{S}}) : :(Vector{S})
    if output_mutates
        return quote
            g!{S}(output::Vector{S}, x::Vector) = gradient!(f, output, x, Val{A}, N, L, M)::$(R)
            return g!
        end
    else
        return quote
            g{S}(x::Vector{S}) = gradient(f, x, Val{A}, N, L, M)::$(R)
            return g
        end
    end
end

#########
# cache #
#########

immutable GradientCache{N, T} <: ForwardDiffCache
    workvec::Vector{DiffNumber{N, T}}
    partials::Vector{Partials{N, T}}
    partials_remainder::Vector{Partials{N, T}}
end

function GradientCache{N, T}(Z, L, ::Type{Val{N}}, ::Type{Val{T}})
    V_diffnumber = Vector{DiffNumber{N,T}}(L)
    V_partials = Vector{Partials{N,T}}(N)
    V_partials_remainder = Vector{Partials{N,T}}(Z)

    x = one(T)
    for i in 1:N
        V_partials[i] = setindex(zero(Partials{N,T}), x, i)
        if i <= Z
            V_partials_remainder[i] = setindex(zero(Partials{N,T}), x, i)
        end
    end

    return GradientCache{N, T}(V_diffnumber, V_partials, V_partials_remainder)
end

##################
# calc_gradient! #
##################
# The below code is pretty ugly, so here's an overview:
#
# `call_gradient!` is the entry point that is called by the API functions. If a chunk size
# isn't given by an upstream caller, `call_gradient!` picks one based on the input length.
#
# `calc_gradient!` is the workhorse function - it generates code for calculating the
# gradient in chunk-mode or vector-mode, depending on the input length and chunk size.
#
# `multi_calc_gradient!` is just like `_calc_gradient!`, but uses Julia's multithreading
# capabilities when performing calculations in chunk-mode.
#
# `VEC_MODE_EXPR` is a constant expression for vector-mode that provides the function body
# for `calc_gradient!` and `multi_calc_gradient!` when chunk size equals input length.
#
# `calc_gradient_expr` takes in a vector-mode expression body or chunk-mode expression body
# and returns a completed function body with the input body injected in the correct place.

@generated function call_gradient!{S,N,L,M}(f, output::Vector{S}, x::Vector, ::Type{Val{N}}, ::Type{Val{L}}, ::Type{Val{M}})
    gradf! = M ? :multi_calc_gradient! : :calc_gradient!
    return :($(gradf!)(f, output, x, Val{$(N == nothing ? pick_chunk(L) : N)}, Val{L})::Tuple{S, Vector{S}})
end

@generated function calc_gradient!{S,T,N,L}(f, output::Vector{S}, x::Vector{T}, ::Type{Val{N}}, ::Type{Val{L}})
    if L == N
        remainder = 0
    else
        remainder = L % N == 0 ? N : L % N
    end

    if N == L
        body = quote
                cache_vec = cachefetch!(Val{N}, Val{T}, Val{L}, Val{$remainder})
                seed_partials::Vector{Partials{N,T}} = cache_vec[tid].partials
                $VEC_MODE_EXPR
            end
    else
        fill_length = L - remainder
        body = quote
            cache_vec = cachefetch!(Val{N}, Val{T}, Val{L}, Val{$remainder})
            seed_partials::Vector{Partials{N,T}} = cache_vec[tid].partials
            cache = cache_vec[tid]
            workvec::Vector{DiffNumber{N,T}} = cache.workvec
            pzeros = zero(Partials{N,T})

            @simd for i in 1:L
                @inbounds workvec[i] = DiffNumber{N,T}(x[i], pzeros)
            end

            for c in 1:$(N):$(fill_length)
                @simd for i in 1:N
                    j = i + c - 1
                    @inbounds workvec[j] = DiffNumber{N,T}(x[j], seed_partials[i])
                end
                local result::DiffNumber{N,S} = f(workvec)
                @simd for i in 1:N
                    j = i + c - 1
                    @inbounds output[j] = partials(result, i)
                    @inbounds workvec[j] = DiffNumber{N,T}(x[j], pzeros)
                end
            end

            # Performing the final chunk manually seems to triggers some additional
            # optimization heuristics, which results in more efficient memory allocation
            @simd for i in 1:$(remainder)
                j = $(fill_length) + i
                @inbounds workvec[j] = DiffNumber{N,T}(x[j], cache.partials_remainder[i])
            end
            result::DiffNumber{N,S} = f(workvec)
            @simd for i in 1:$(remainder)
                j = $(fill_length) + i
                @inbounds output[j] = partials(result, i)
                @inbounds workvec[j] = DiffNumber{N,T}(x[j], pzeros)
            end
        end
    end
    return calc_gradient_expr(body)
end

if VERSION >= THREAD_VERSION
    @generated function multi_calc_gradient!{S,T,N,L}(f, output::Vector{S}, x::Vector{T}, ::Type{Val{N}}, ::Type{Val{L}})
        if N == L
            body = VEC_MODE_EXPR
        else
            nthreads = Threads.nthreads()
            remainder = L % N == 0 ? N : L % N
            fill_length = L - remainder
            body = quote
                pzeros = zero(Partials{N,T})

                Threads.@threads for t in 1:$(nthreads)
                    # must be local, see https://github.com/JuliaLang/julia/issues/14948
                    local workvec = cache_vec[t].workvec
                    @simd for i in 1:L
                        @inbounds workvec[i] = DiffNumber{N,T}(x[i], pzeros)
                    end
                end

                Threads.@threads for c in 1:$(N):$(fill_length)
                    local workvec = cache[Threads.threadid()].workvec
                    @simd for i in 1:N
                        j = i + c - 1
                        @inbounds workvec[j] = DiffNumber{N,T}(x[j], seed_partials[i])
                    end
                    local result::DiffNumber{N,S} = f(workvec)
                    @simd for i in 1:N
                        j = i + c - 1
                        @inbounds output[j] = partials(result, i)
                        @inbounds workvec[j] = DiffNumber{N,T}(x[j], pzeros)
                    end
                end

                # Performing the final chunk manually seems to triggers some additional
                # optimization heuristics, which results in more efficient memory allocation
                workvec = cache_vec[tid].workvec
                @simd for i in 1:$(remainder)
                    j = $(fill_length) + i
                    @inbounds workvec[j] = DiffNumber{N,T}(x[j], cache_vec[tid].partials_remainder[i])
                end
                result::DiffNumber{N,S} = f(workvec)
                @simd for i in 1:$(remainder)
                    j = $(fill_length) + i
                    @inbounds output[j] = partials(result, i)
                    @inbounds workvec[j] = DiffNumber{N,T}(x[j], pzeros)
                end
            end
        end
        return calc_gradient_expr(body)
    end
end

const VEC_MODE_EXPR = quote
    workvec::Vector{DiffNumber{N,T}} = cache_vec[tid].workvec
    @simd for i in 1:L
        @inbounds workvec[i] = DiffNumber{N,T}(x[i], seed_partials[i])
    end
    result::DiffNumber{N,S} = f(workvec)
    @simd for i in 1:L
        @inbounds output[i] = partials(result, i)
    end
end

function calc_gradient_expr(body)

    return quote
        @assert L == length(x) == length(output)
        tid = Threads.threadid()
        $(body)
        return (value(result)::S, output::Vector{S})
    end
end
