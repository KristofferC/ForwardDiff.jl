const CACHE = Dict{Tuple{Int, DataType, Int, Int}, Vector}()

function clearcache!()
    empty(CACHE)
end

abstract ForwardDiffCache

function cachefetch!{N,T,L,Z}(::Type{Val{N}}, ::Type{Val{T}}, ::Type{Val{L}}, ::Type{Val{Z}})
    K = (N, T, L, Z)
    V =  Vector{GradientCache{N, T}}
    if haskey(CACHE, K)
        cache_vec = CACHE[K]::V
    else
        cache_vec = V(NTHREADS)
        for i in 1:NTHREADS
            cache_vec[i] = GradientCache(Z, L, Val{N}, Val{T})
        end
        CACHE[K] = cache_vec
    end
    return cache_vec::V
end
