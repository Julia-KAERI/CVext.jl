
module CVext
    using OpenCV
    using Reexport
    @reexport using OpenCV

    using Distributions
    export arr2mat, img2arr, img2mat, mat2arr, histogram1d, gaussian_noise, salt_pepper_noise, Point, Size, ConvertTo, fftshift, ifftshift, fftfreq

    Mat = OpenCV.Mat
    dft = OpenCV.dft
    idft = OpenCV.idft

    export Mat, dft, idft

    """
    `arr2mat(arr::Matrix{T})`
    
    Convert julia's Array{T} to OpenCV.Mat{T}
    """
    function arr2mat(arr::Matrix{T}) where T<:Union{Float32, Float64, Int16, Int32, Int8, UInt16, UInt8}
        OpenCV.Mat(permutedims(stack([arr, ]), [3,2,1]))
    end

    function arr2mat(arr::Array{T, 3}) where T<:Union{Float32, Float64, Int16, Int32, Int8, UInt16, UInt8}
        OpenCV.Mat(permutedims(stack([arr, ]), [3,2,1]))
    end

    """
    `img2arr(img)`
    
    Convert Images.jl's image to Array
    """
    function img2arr(img)
        T = typeof(img[1, 1].val.i)
        broadcast(q->T(q.val.i), img)
    end

    """
    `img2mat(img)` 

    Convert Images.jl's image to OpenCV.Mat
    """
    function img2mat(img) 
        T = typeof(img[1, 1].val.i)
        tm = broadcast(q->T(q.val.i),img)
        OpenCV.Mat(permutedims(stack([tm, ]), [3,2,1]))
    end

    """
    `mat2arr(mat::OpenCV.Mat)`

    Convert OpenCV.Mat{T} to Array{T, 2} or Array{T, 3}
    """
    function mat2arr(mat::OpenCV.Mat)
        return permutedims(mat.data, [3,2,1])
    end


    """
    `histogram1d(mat::OpenCV.Mat{T})`

    Generate histogram of OpenCV.Mat
    """
    function histogram1d(mat::OpenCV.Mat{T}) where T<:Union{UInt8, UInt16}
        tm = Int32(typemax(T))

        v = OpenCV.calcHist(OpenCV.InputArray[mat,], Int32[0], fill(UInt8(1), size(mat)), Int32[tm+1], Float32[0, tm+1])
        return (0:1:tm, Int64.(v[1,1,:]))    
    end


    """
    `gaussian_noise(img::OpenCV.Mat{T}, μ, σ, N)`

    Generate Gaussian noise added image(OpenCV.Mat type) from img.
    
    Parameters
    ==========
    μ, σ is mean and std of gaussian distribution
    N : number of noise point
    """
    function gaussian_noise(img::OpenCV.Mat{T}, μ, σ, N) where T<:Union{UInt8, UInt16}
        m, n = size(img)[2:3]
        MM = typemax(T)
        Y, X, V = rand(1:m, N), rand(1:n, N), round.(T, rand(truncated(Normal(μ, σ), 0, MM), N))
        img2 = copy(img)
        for (y, x, v) ∈ zip(Y, X, V)
            img2[1, y, x] = v
        end
        return img2
    end

    """
    `salt_pepper_noise(img::OpenCV.Mat{T}, N)`

    Generate salt-papper noise image(OpenCV.Mat type) from img.
    
    Parameters
    ==========
    N : number of noise point
    """
    function salt_pepper_noise(img::OpenCV.Mat{T}, N) where T<:Union{UInt8, UInt16}
        m, n = size(img)[2:3]
        MM = typemax(T)
        Y, X, V = rand(1:m, N), rand(1:n, N), sample([0, MM], N)
        img2 = copy(img)
        for (y, x, v) ∈ zip(Y, X, V)
            img2[1, y, x] = v
        end
        return img2
    end

    """
    `Point(x, y)`

    Return OpenCV.Point{T}(x, y)
    """
    function Point(x::T1, y::T2) where {T1<:Real, T2<:Real}
        T = promote_type(T1, T2)
        return OpenCV.Point{T}(T(x), T(y))
    end

    """
    `Point(x, y, z)`

    Return OpenCV.Point{T}(x, y, z)
    """
    function Point(x::T1, y::T2, z::T3) where {T1<:Real, T2<:Real, T3<:Real}
        T = promote_type(T1, T2, T3)
        return OpenCV.Point3{T}(T(x), T(y), T(z))
    end

    """
    `Size(w, h)`

    Return OpenCV.Size{Int32}(w, h) 
    """
    function Size(w::T1, h::T2) where {T1<:Integer, T2<:Integer}
        return OpenCV.Size(Int32(w), Int32(h))
    end

    """
    `ConvertTo(mat::OpenCV.Mat, t::T)`

    Convert element type of mat to t
    """
    function ConvertTo(mat::OpenCV.Mat, t::T) where T<:Type
        return OpenCV.Mat(convert.(t, mat.data))
    end
    
    """
    `fftshift(v::Vector)`

    Google `fftshift` (for example numpy ...)
    """
    function fftshift(v::Vector)
        m = length(v)
        n = iseven(m) ? div(m, 2) : div(m, 2)+1
        return [v[n+1:end]; v[1:n] ]
    end 


    function fftshift(A::OpenCV.Mat)
        c, h, w = size(A)
        m = iseven(h) ? div(h, 2) : div(h, 2)+1
        n = iseven(w) ? div(w, 2) : div(w, 2)+1
        return OpenCV.Mat(stack([[A[i, m+1:end, n+1:end] A[i, m+1:end, 1:n] ; A[i, 1:m, n+1:end] A[i, 1:m, 1:n] ] for i in 1:c], dims=1))
    end


    """
    `ifftshift(v::Vector)`

    Google `ifftshift` (for example numpy ...)
    """
    function ifftshift(v::Vector)
        m = length(v)
        n = iseven(m) ? div(m, 2) : div(m, 2)
        return [v[n+1:end]; v[1:n] ]
    end 
    
    
    function ifftshift(A::Matrix)
        h, w = size(A)
        m = iseven(h) ? div(h, 2) : div(h, 2)
        n = iseven(w) ? div(w, 2) : div(w, 2)
        return [A[m+1:end, n+1:end] A[m+1:end, 1:n] ; A[1:m, n+1:end] A[1:m, 1:n] ]
    end

    function ifftshift(A::OpenCV.Mat)
        c, h, w = size(A)
        m = iseven(h) ? div(h, 2) : div(h, 2)
        n = iseven(w) ? div(w, 2) : div(w, 2)
        return OpenCV.Mat(stack([[A[i, m+1:end, n+1:end] A[i, m+1:end, 1:n] ; A[i, 1:m, n+1:end] A[i, 1:m, 1:n] ] for i in 1:c], dims=1))
    end

    """
    `fftfreq(n::Integer, d::Real =1)`

    Google `fffreq` (for example numpy ...)
    """
    function fftfreq(n::Integer, d::Real =1)
        @assert n>0
        m = n>>1
        if isodd(n)
            return [collect(0:m); collect(-(m):-1)]./(n*d)
        else
            
            return [collect(0:m-1); collect(-(m):-1)]./(n*d)
        end
    end

end;



