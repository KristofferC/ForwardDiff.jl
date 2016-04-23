####################
# Taking Gradients #
####################

# Exposed API methods #
#---------------------#
@generated function gradient!{T,A}(output::AbstractVector{T}, f, x, ::Type{A}=Void;
                                   chunk_size::Int=default_chunk_size,
                                   cache::ForwardDiffCache=dummy_cache)
    if A <: Void
        return_stmt = :(gradient!(output, result)::typeof(output))
    elseif A <: AllResults
        return_stmt = :(gradient!(output, result)::typeof(output), result)
    else
        error("invalid argument $A passed to FowardDiff.gradient")
    end

    return quote
        result = _calc_gradient(f, x, T, chunk_size, cache)
        return $return_stmt
    end
end

@generated function gradient{A}(f, x, ::Type{A}=Void;
                                  chunk_size::Int=default_chunk_size,
                                  cache::ForwardDiffCache=dummy_cache)
    if A <: Void
        return_stmt = :(gradient(result))
    elseif A <: AllResults
        return_stmt = :(gradient(result), result)
    else
        error("invalid argument $A passed to FowardDiff.gradient")
    end

    return quote
        result = _calc_gradient(f, x, eltype(x), chunk_size, cache)
        return $return_stmt
    end
end

function gradient{A}(f, ::Type{A}=Void;
                     mutates::Bool=false,
                     chunk_size::Int=default_chunk_size,
                     cache::ForwardDiffCache=ForwardDiffCache())
    if mutates
        function g!(output::AbstractVector, x)
            return ForwardDiff.gradient!(output, f, x, A;
                                         chunk_size=chunk_size,
                                         cache=cache)
        end
        return g!
    else
        function g(x)
            return ForwardDiff.gradient(f, x, A;
                                        chunk_size=chunk_size,
                                        cache=cache)
        end
        return g
    end
end

# Calculate gradient of a given function #
#----------------------------------------#
function _calc_gradient{S}(f, x, ::Type{S},
                           chunk_size::Int,
                           cache::ForwardDiffCache)
    X = Val{length(x)}
    C = Val{chunk_size}
    return _calc_gradient(f, x, S, X, C, cache)
end

@generated function _calc_gradient{S,xlen,chunk_size}(f, x, ::Type{S},
                                                        X::Type{Val{xlen}},
                                                        C::Type{Val{chunk_size}},
                                                        cache::ForwardDiffCache)

    check_chunk_size(xlen, chunk_size)
    G = workvec_eltype(GradientNumber, eltype(x), Val{xlen}, Val{chunk_size})
    gradvec = build_workvec(G, xlen)

    partials = build_partials(G)
    gradzeros = build_zeros(G)
    
    if chunk_size_matches_vec_mode(xlen, chunk_size)
        # Vector-Mode
        ResultType = switch_eltype(G, S)
        body = quote
            @simd for i in 1:xlen
                @inbounds $gradvec[i] = G(x[i], $partials[i])
            end

            result::$ResultType = f($gradvec)
        end
    else
        # Chunk-Mode
        ChunkType = switch_eltype(G, S)
        ResultType = GradientNumber{xlen,S,Vector{S}}
        body = quote
            output = Vector{S}(xlen)

            @simd for i in 1:xlen
                @inbounds $gradvec[i] = G(x[i], $gradzeros)
            end

            local chunk_result::$ChunkType

            for i in 1:chunk_size:xlen
                offset = i-1

                @simd for j in 1:chunk_size
                    q = j+offset
                    @inbounds $gradvec[q] = G(x[q], $partials[j])
                end

                chunk_result = f($gradvec)

                @simd for j in 1:chunk_size
                    q = j+offset
                    @inbounds output[q] = grad(chunk_result, j)
                    @inbounds $gradvec[q] = G(x[q], $gradzeros)
                end
            end

            result::$ResultType = ($ResultType)(value(chunk_result), output)
        end
    end

    return quote
        G = $G
        T = eltype(x)

        $body

        return ForwardDiffResult(result)
    end
end
