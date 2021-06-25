using CUDA
#using CairoMakie
using Makie

function cuda_mandelbrot!(pic, z, c, xrange, yrange, n = 150, b=4)
    z[:] .= 0
    pic[:] .= -1
    for (i, x) in enumerate(xrange)
        c[i, :] .= x
    end
    for (i, y) in enumerate(yrange)
        c[:, i] .+= im*y
    end
    for x=1:n
        @cuda threads=256 blocks=cld(length(z), 256) cuda_kernel!(pic, z, c, b, x)
    end
    pic
end


function cuda_kernel!(pic, z, c, b, val)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for id = index:stride:length(z)
        if c[id] !=0 && abs2(z[id]) > b
            z[id] = 0
            c[id] = 0
            pic[id] = val
        elseif c[id] != 0
            
            z[id] = z[id]*z[id] +c[id]  
            
        end
    end
    return nothing
end

function thread_mandelbrot!(pic, z, c, xrange, yrange, n = 50, b=4)
    pic[:] .= 0
    z[:] .= 0
    for (i, x) in enumerate(xrange)
        c[i, :] .= x
    end
    for (i, y) in enumerate(yrange)
        c[:, i] .+= im*y
    end
    for x=1:n
        thread_kernel!(pic, z, c, b, x)
    end
    pic
end

function thread_kernel!(pic, z, c, b, val)
    Threads.@threads for id=1:length(z)
        if c[id] !=0 && abs2(z[id]) > b
            z[id] = 0
            c[id] = 0
            pic[id] = val
        elseif c[id] != 0
            z[id] = z[id]*z[id] +c[id]  
        end
    end
    return nothing
end

function thread_julia!(f, pic, z, c, xrange, yrange, b=1000, n = 100)
    pic[:] .= 0
    for (i, x) in enumerate(xrange)
        c[i, :] .= x
    end
    for (i, y) in enumerate(yrange)
        c[:, i] .+= im*y
    end
    z[:] = c[:]
    for x=1:n
        thread_julia_kernel!(f, pic, z, b, x)
    end
    pic
end

function thread_julia_kernel!(f, pic, z, b, val)
    Threads.@threads for id=1:length(z)
        if pic[id] == 0 && abs2(z[id]) > b
            z[id] = 0
            pic[id] = val
        elseif pic[id] == 0
            z[id] = f(z[id])
        end
    end
    return nothing
end

function setup(xpix, ypix; xlow=-2, xhigh=1, ylow=-1, yhigh=1)
    xrange = range(xlow, xhigh, length=xpix)
    yrange = range(ylow, yhigh, length=ypix)
    pic = zeros(Int32, length(xrange), length(yrange))
    z = zeros(Complex{Float64}, length(xrange), length(yrange))
    c = zeros(Complex{Float64}, length(xrange), length(yrange))
    (pic, z, c, xrange, yrange)
end

function cuda_setup(xpix, ypix)
    xrange = range(-2, 1, length=xpix)
    yrange = range(-1, 1, length=ypix)
    pic = CUDA.zeros(Int32, length(xrange), length(yrange))
    z = CUDA.zeros(ComplexF64, length(xrange), length(yrange))
    c = CUDA.zeros(ComplexF64, length(xrange), length(yrange))
    (pic, z, c, xrange, yrange)
end

function interactive_sim()
    (pic, z, c, xrange, yrange) = setup(1600, 800)
    thread_mandelbrot!(pic, z, c, xrange, yrange)
    p = Node(Array(pic))
    fig = Makie.heatmap(xrange, yrange, p, figure=(resolution=(1600, 800), ))
    on(events(fig).keyboardbutton) do event
        if event.action in (Keyboard.press, Keyboard.repeat)
            if event.key == Keyboard.left 
                (xrange, yrange) = shift(xrange, yrange, -0.05, 0)
            elseif event.key == Keyboard.up     
                (xrange, yrange) = shift(xrange, yrange, 0, 0.05 )
            elseif event.key == Keyboard.right 
                (xrange, yrange) = shift(xrange, yrange, 0.05, 0)
            elseif event.key == Keyboard.down  
                (xrange, yrange) = shift(xrange, yrange, 0, -0.05)
            elseif event.key == Keyboard.kp_add 
                (xrange, yrange) = zoom(xrange, yrange, 0.95)
            elseif event.key == Keyboard.kp_subtract 
                (xrange, yrange) = zoom(xrange, yrange, 1.05)
            end
            @show xrange
            @show yrange
            thread_mandelbrot!(pic, z, c, xrange, yrange )
            p[] = Array(pic)
        end
        # Let the event reach other listeners
        return false
    end
    fig
end

function interactive_julia_sim(f)
    (pic, z, c, xrange, yrange) = setup(1600, 800, xlow=-2, xhigh=2)
    thread_julia!(f, pic, z, c, xrange, yrange)
    p = Node(Array(pic))
    fig = Makie.heatmap(p, figure=(resolution=(1600, 800), ))
    on(events(fig).keyboardbutton) do event
        if event.action in (Keyboard.press, Keyboard.repeat)
            if event.key == Keyboard.left 
                (xrange, yrange) = shift(xrange, yrange, -0.05, 0)
            elseif event.key == Keyboard.up     
                (xrange, yrange) = shift(xrange, yrange, 0, 0.05 )
            elseif event.key == Keyboard.right 
                (xrange, yrange) = shift(xrange, yrange, 0.05, 0)
            elseif event.key == Keyboard.down  
                (xrange, yrange) = shift(xrange, yrange, 0, -0.05)
            elseif event.key == Keyboard.kp_add 
                (xrange, yrange) = zoom(xrange, yrange, 0.95)
            elseif event.key == Keyboard.kp_subtract 
                (xrange, yrange) = zoom(xrange, yrange, 1.05)
            end
            @show xrange
            @show yrange
            thread_julia!(f, pic, z, c, xrange, yrange )
            p[] = Array(pic)
        end
        # Let the event reach other listeners
        return false
    end
    fig
end

function combined_sim()
    f = Figure()
    max = Axis(f[1, 1])
    jax = Axis(f[2, 1])
    (mpic, mz, mc, mxrange, myrange) = setup(1600, 800)
    (jpic, jz, jc, jxrange, jyrange) = setup(800, 400, xlow=-2, xhigh=2)
    thread_julia!((z) -> z^2, jpic, jz, jc, jxrange, jyrange)
    thread_mandelbrot!(mpic, mz, mc, mxrange, myrange)
    mp = Node(Array(mpic))
    jp = Node(Array(jpic))
    fig = Makie.heatmap!(max, mxrange, myrange, mp)
    Makie.heatmap!(jax, jxrange, jyrange, jp)
    on(events(fig).keyboardbutton) do event
        if event.action in (Keyboard.press, Keyboard.repeat)
            if event.key == Keyboard.left 
                (mxrange, myrange) = shift(mxrange, myrange, -0.05, 0)
            elseif event.key == Keyboard.up     
                (mxrange, myrange) = shift(mxrange, myrange, 0, 0.05 )
            elseif event.key == Keyboard.right 
                (mxrange, myrange) = shift(mxrange, myrange, 0.05, 0)
            elseif event.key == Keyboard.down  
                (mxrange, myrange) = shift(mxrange, myrange, 0, -0.05)
            elseif event.key == Keyboard.kp_add 
                (mxrange, myrange) = zoom(mxrange, myrange, 0.95)
            elseif event.key == Keyboard.kp_subtract 
                (mxrange, myrange) = zoom(mxrange, myrange, 1.05)
            end
            @show mxrange
            @show myrange
            thread_mandelbrot!(mpic, mz, mc, mxrange, myrange )
            mp[] = Array(mpic)
        end
        # Let the event reach other listeners
        return false
    end
    on(events(fig).mousebutton, priority = 0) do event
        if event.button == Mouse.left
            if event.action == Mouse.press
                pos = mouseposition(f.scene)
                @show pos
                c = jxrange[round(Int, pos[1])-55]+ im* jyrange[round(Int, pos[2])-340]
                thread_julia!((z) -> z^2 + c, jpic, jz, jc, jxrange, jyrange)
                @show c
                #@show jpic
                jp[] = jpic
            end
        end
        # Do not consume the event
        return false
    end
    f
end


function shift(xrange, yrange, dx, dy)
    #println("shifting")
    dx = dx *(maximum(xrange) - minimum(xrange))
    dy = dy *(maximum(yrange) - minimum(yrange))
    return (xrange .+ dx , yrange .+ dy)
end

function zoom(xrange, yrange, factor)
    xmid = (minimum(xrange) + maximum(xrange))/2
    ymid = (minimum(yrange) + maximum(yrange))/2
    xr = (maximum(xrange) - minimum(xrange))/2*factor
    yr = (maximum(yrange) - minimum(yrange))/2*factor
    return (range(xmid-xr, xmid+xr, length=length(xrange)),
        range(ymid-yr, ymid+yr, length=length(yrange)))
end