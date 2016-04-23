####################
# Taking Jacobians #
####################

# Exposed API methods #
#---------------------#
@generated function jacobian!{T,A}(output::Matrix{T}, f, x::Vector, ::Type{A}=Void;
                                   chunk_size::Int=default_chunk_size,
                                   output_length::Int=0,
                                   cache::ForwardDiffCache=dummy_cache)
    if A <: Void
        return_stmt = :(jacobian!(output, result)::Matrix{T})
    elseif A <: AllResults
        return_stmt = :(jacobian!(output, result)::Matrix{T}, result)
    else
        error("invalid argument $A passed to FowardDiff.jacobian")
    end

    return quote
        result = _calc_jacobian(f, x, T, chunk_size, output_length, cache)
        return $return_stmt
    end
end

@generated function jacobian{T,A}(f, x::Vector{T}, ::Type{A}=Void;
                                  chunk_size::Int=default_chunk_size,
                                  output_length::Int=0,
                                  cache::ForwardDiffCache=dummy_cache)
    if A <: Void
        return_stmt = :(jacobian(result)::Matrix{T})
    elseif A <: AllResults
        return_stmt = :(jacobian(result)::Matrix{T}, result)
    else
        error("invalid argument $A passed to FowardDiff.jacobian")
    end

    return quote
        result = _calc_jacobian(f, x, T, chunk_size, output_length, cache)
        return $return_stmt
    end
end

function jacobian{A}(f, ::Type{A}=Void;
                     mutates::Bool=false,
                     chunk_size::Int=default_chunk_size,
                     cache::ForwardDiffCache=ForwardDiffCache(),
                     output_length::Int=0)
    # if output_length > 0, assume that f is of
    # the form f!(output, x), and generate the
    # appropriate closure
    if output_length > 0
        #output_cache = ForwardDiffCache()
        newf = (output::Vector, x::Vector) -> begin
            f(output, x)
            return output
        end
    else
        newf = f
    end

    if mutates
        function j!(output::Matrix, x::Vector)
            return ForwardDiff.jacobian!(output, newf, x, A;
                                         chunk_size=chunk_size,
                                         output_length=output_length,
                                         cache=cache)
        end
        return j!
    else
        function j(x::Vector)
            return ForwardDiff.jacobian(newf, x, A;
                                        chunk_size=chunk_size,
                                        output_length=output_length,
                                        cache=cache)
        end
        return j
    end
end

# Calculate Jacobian of a given function #
#----------------------------------------#
function _calc_jacobian{S}(f, x::Vector, ::Type{S},
                           chunk_size::Int,
                           output_length::Int,
                           cache::ForwardDiffCache)
    X = Val{length(x)}
    C = Val{chunk_size}
    output_length = Val{output_length}
    return _calc_jacobian(f, x, S, X, C, output_length, cache)
end

@generated function _calc_jacobian{T,S,xlen,chunk_size, output_length}(f, x::Vector{T}, ::Type{S},
                                                        X::Type{Val{xlen}},
                                                        C::Type{Val{chunk_size}},
                                                        ::Type{Val{output_length}},
                                                        cache::ForwardDiffCache)
    check_chunk_size(xlen, chunk_size)
    G = workvec_eltype(GradientNumber, T, Val{xlen}, Val{chunk_size})
    gradvec = build_workvec(G, xlen)
    output = build_workvec(G, output_length)
    partials = build_partials(G)
    gradzeros = build_zeros(G)

    if chunk_size_matches_vec_mode(xlen, chunk_size)
        # Vector-Mode
        ResultType = switch_eltype(G, S)
        body = quote
            @simd for i in 1:xlen
                @inbounds $gradvec[i] = G(x[i], $partials[i])
            end

            if $output_length == 0
                result = f($gradvec)
            else
                result = f($output, $gradvec)
            end
        end
    else
        # Chunk-Mode
        ChunkType = switch_eltype(G, S)
        ResultType = GradientNumber{xlen,S,Vector{S}}
        body = quote
            @simd for i in 1:xlen
                @inbounds $gradvec[i] = G(x[i], $gradzeros)
            end

            # Perform the first chunk "manually" to retrieve
            # the info we need to build our output

            @simd for i in 1:chunk_size
                @inbounds $gradvec[i] = G(x[i], $partials[i])
            end

            first_result::Vector{$ChunkType} = f($gradvec)

            nrows, ncols = length(first_result), xlen
            output = Vector{S}[Vector{S}(ncols) for i in 1:nrows]

            for j in 1:chunk_size
                @simd for i in 1:nrows
                    @inbounds output[i][j] = grad(first_result[i], j)
                end
                @inbounds $gradvec[j] = G(x[j], $gradzeros)
            end

            # Perform the rest of the chunks, filling in the output as we go

            local chunk_result::Vector{$ChunkType}

            for i in (chunk_size+1):chunk_size:xlen
                offset = i-1

                @simd for j in 1:chunk_size
                    m = j+offset
                    @inbounds $gradvec[m] = G(x[m], $partials[j])
                end

                if $output_length == 0
                    chunk_result = f($gradvec)
                else
                    chunk_result = f($output, $gradvec)
                end

                for j in 1:chunk_size
                    m = j+offset
                    @simd for n in 1:nrows
                        @inbounds output[n][m] = grad(chunk_result[n], j)
                    end
                    @inbounds $gradvec[m] = G(x[m], $gradzeros)
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
        $body

        return ForwardDiffResult(result)
    end
end
