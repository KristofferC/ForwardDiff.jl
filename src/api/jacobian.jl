####################
# Taking Jacobians #
####################

type JacobianCache{T, Q}
    workvec::Vector{T}
    output::Vector{T}
    zeros::Vector{T}
    partials::Vector{Q}
end

function JacobianCache{T}(Tv::Type{T}, intput_length, output_length)
    workvec = zeros(Tv, output_length)
    output = zeros(Tv, intput_length)
    _zeros = zeros(Tv, output_length)
    partials = build_partials(T)
    return JacobianCache(workvec, output, _zeros, partials)
end

get_output(cache::JacobianCache) = cache.output
get_zeros(cache::JacobianCache) = cache._zeros
get_workvec(cache::JacobianCache) = cache.workvec
get_partials(cache::JacobianCache) = cache.partials

# Exposed API methods #
#---------------------#
@generated function jacobian!{T,A}(output::Matrix{T}, f, x::Vector, ::Type{A}=Void;
                                   chunk_size::Int=default_chunk_size,
                                   cache::JacobianCache=dummy_cache)
    if A <: Void
        return_stmt = :(jacobian!(output, result)::Matrix{T})
    elseif A <: AllResults
        return_stmt = :(jacobian!(output, result)::Matrix{T}, result)
    else
        error("invalid argument $A passed to FowardDiff.jacobian")
    end

    return quote
        result = _calc_jacobian(f, x, T, chunk_size, cache)
        return $return_stmt
    end
end

@generated function jacobian{T,A}(f, x::Vector{T}, ::Type{A}=Void;
                                  chunk_size::Int=default_chunk_size,
                                  cache::JacobianCache=dummy_cache)
    if A <: Void
        return_stmt = :(jacobian(result)::Matrix{T})
    elseif A <: AllResults
        return_stmt = :(jacobian(result)::Matrix{T}, result)
    else
        error("invalid argument $A passed to FowardDiff.jacobian")
    end

    return quote
        result = _calc_jacobian(f, x, T, chunk_size, cache)
        return $return_stmt
    end
end

function jacobian{A}(f, input_length::Int, output_length::Int, ::Type{A}=Void;
                     mutates::Bool=false,
                     )

    G = workvec_eltype(GradientNumber, Float64, Val{input_length}, Val{input_length})
    cache = JacobianCache(G, input_length, output_length)

    function newf{G<:GradientNumber}(x::Vector{G})
        output = get_output(cache)
        f(output, x)
        return output
    end

    if mutates
        function j!(output::Matrix, x::Vector)
            return ForwardDiff.jacobian!(output, newf, x, A;
                                         chunk_size=length(x),
                                         cache=cache)
        end
        return j!
    else
        function j(x::Vector)
            return ForwardDiff.jacobian(newf, x, A;
                                        chunk_size=length(x),
                                        cache=cache)
        end
        return j
    end
end

# Calculate Jacobian of a given function #
#----------------------------------------#
function _calc_jacobian{S}(f, x::Vector, ::Type{S},
                           chunk_size::Int,
                           cache::JacobianCache)
    X = Val{length(x)}
    C = Val{chunk_size}
    return _calc_jacobian(f, x, S, X, C, cache)
end

@generated function _calc_jacobian{T,S,xlen,chunk_size}(f, x::Vector{T}, ::Type{S},
                                                        X::Type{Val{xlen}},
                                                        C::Type{Val{chunk_size}},
                                                        cache::JacobianCache)
    check_chunk_size(xlen, chunk_size)
    G = workvec_eltype(GradientNumber, T, Val{xlen}, Val{chunk_size})
    if chunk_size_matches_vec_mode(xlen, chunk_size)
        # Vector-Mode
        ResultType = switch_eltype(G, S)
        body = quote
            @simd for i in eachindex(x)
                @inbounds gradvec[i] = G(x[i], partials[i])
            end

            result::Vector{$ResultType} = f(gradvec)
        end
    else
        # Chunk-Mode
        ChunkType = switch_eltype(G, S)
        ResultType = GradientNumber{xlen,S,Vector{S}}
        body = quote
            N = npartials(G)
            gradzeros = get_zeros!(cache, G)

            @simd for i in eachindex(x)
                @inbounds gradvec[i] = G(x[i], gradzeros)
            end

            # Perform the first chunk "manually" to retrieve
            # the info we need to build our output

            @simd for i in 1:N
                @inbounds gradvec[i] = G(x[i], partials[i])
            end

            first_result::Vector{$ChunkType} = f(gradvec)

            nrows, ncols = length(first_result), xlen
            output = Vector{S}[Vector{S}(ncols) for i in 1:nrows]

            for j in 1:N
                @simd for i in 1:nrows
                    @inbounds output[i][j] = grad(first_result[i], j)
                end
                @inbounds gradvec[j] = G(x[j], gradzeros)
            end

            # Perform the rest of the chunks, filling in the output as we go

            local chunk_result::Vector{$ChunkType}

            for i in (N+1):N:xlen
                offset = i-1

                @simd for j in 1:N
                    m = j+offset
                    @inbounds gradvec[m] = G(x[m], partials[j])
                end

                chunk_result = f(gradvec)

                for j in 1:N
                    m = j+offset
                    @simd for n in 1:nrows
                        @inbounds output[n][m] = grad(chunk_result[n], j)
                    end
                    @inbounds gradvec[m] = G(x[m], gradzeros)
                end
            end

            result = Vector{$ResultType}(nrows)

            @simd for i in eachindex(result)
                @inbounds result[i] = ($ResultType)(value(chunk_result[i]), output[i])
            end
        end
    end

    return quote
        G = $G
        gradvec = get_workvec(cache)
        partials = get_partials(cache)

        $body

        return ForwardDiffResult(result)
    end
end
