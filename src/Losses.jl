module Losses

using Base: Fix2


using Flux
using Zygote

using Boilerplate
using Pythonish
using Arithmetics: mean
using ZygoteExtensions: gradient_pro
using Utils: Fix2_more




@inline se(ŷ, Y)         = @. (ŷ - Y) ^ 2 
@inline se(ŷ, Y, v)      = @. (ŷ - Y) ^ 2 * v
@inline mse(ŷ, Y)        = mean(se(ŷ, Y))
@inline ⎷mse(ŷ, Y)       = sqrt(mse(ŷ, Y))
@inline se_with_L2(ŷ, Y, vP)       = se(ŷ, Y) * (1f0 + L2(vP) * 1f0 * 0.0003f0)
@inline mse_with_L2(ŷ, Y, vP)      = mse(ŷ, Y) * (1f0 + L2(vP) * 1f0 * 0.0003f0)
@inline ⎷mse_with_L2(ŷ, Y, vP)     = ⎷mse(ŷ, Y) * (1f0 + L2(vP) * 1f0 * 0.0003f0)
@inline softmax(ŷ, Y)              = ⎷mse(Flux.softmax(ŷ; dims=3), Y)
@inline softmax_with_L2(ŷ, Y, vP)  = ⎷mse(Flux.softmax(ŷ; dims=3), Y) * (1f0 + L2(vP) * 1f0 * 0.0003f0)
@inline onehot(ŷ, y) = sum(argmax(ŷ, dims=3) .== argmax(y, dims=3)) / size(ŷ,1)
@inline onehot_custom(ŷ, y, meta, val) = sum(is_idx_eq_val(val), argmax(ŷ, dims=3) .&& map(is_idx_eq_val(val), argmax(y, dims=3))) ./ sum(map(is_idx_eq_val(val), argmax(y, dims=3)))
is_idx_eq_val(val) = (idx::CartesianIndex) -> idx[3] === val


L1(v) = mean(abs.(v))
L2(v) = mean(v .^ 2)


loss_grad(loss_fn, Ŷ, Y)     = gradient_pro(Fix2(loss_fn, Y), Ŷ)
loss_grad(loss_fn, Ŷ, Y, vP) = gradient_pro(Fix2_more(loss_fn, Y), Ŷ, vP)


# loss_mse(Ŷ, Y, vP) = begin
# 	tmp_fn=ŷ -> mse(ŷ, Y)
# 	gradient_pro(
# 		(ŷ, ps) -> tmp_fn(ŷ) * (1f0 + L2(ps) *1f0* 0.0003f0),  # paraméterek száma talán számít az L2-nél???
# 		# (ŷ, ps) -> mse(ŷ, Y) * (1f0 + L2(ps) * 0.0),  # paraméterek száma talán számít az L2-nél???
# 		# (ŷ, ps) -> mse(ŷ, Y) * (0.001f0 + L2(ps) * 0.00003f0),  # paraméterek száma talán számít az L2-nél???
# 		# (ŷ, ps) -> mse(ŷ, Y) * (1f0 + L2(ps) * 3f0),
# 		# (ŷ, ps) -> crossentropy(ŷ, Y) *(1f0 + L2(ps) * 0.00003f0),
# 		Ŷ, vP)..., nothing
# end

# @inline loss_mse(Ypred, Yref, N) = @views mse.(Ypred[:,:,N], Yref[:,:,1])  # TODO check speeeed?
# @inline loss_mnist(Ypred, Yref) = mse.(Ypred[:,:,15], Yref[:,:,1])  # TODO MNIST



function sign_mask(x)
	@. ifelse(x < 0.0f0, 0.0f0, 1f0)
end
Zygote.@adjoint function sign_mask(x)
  lbound = 0.01f0
	my_pullback_fn(dy) = begin
		grad_leak = 200f0
		g = @. ifelse(x <= 0.0f0, 
									0.0f0, 
									ifelse(x < lbound, 0.10f0, 0.0f0)) + 
						ifelse(x < -1.0f0, 
									grad_leak, 
									ifelse(x < 0f0, 
												2.000f0, 
												ifelse(x < 1.0f0, 
															0.10f0, 
															0)))
		return (g .* dy, nothing)
	end
	val = @. ifelse(x < 0.0f0, 0.0f0, 1f0)
  return (val, my_pullback_fn)
end
function switchmask(x, leaky_alpha)
  lbound = -0.01f0
  upval_bound = 1f0 - leaky_alpha
	# @show upval_bound
  reciproc = upval_bound/lbound
	# @show reciproc
  # @. ifelse(x >= 0f0, 0f0, ifelse(x > lbound, x * reciproc, upval_bound)) + ifelse(x > 0f0, 0f0, ifelse(x > -1f0, -x * leaky_alpha, leaky_alpha))
	@. ifelse(x > 0.0f0, 0.0f0, 1f0)
end
Zygote.@adjoint function switchmask(x, leaky_alpha)
  lbound = -0.01f0
  upval_bound = 1.0f0 - leaky_alpha
  reciproc = upval_bound / lbound
	my_pullback_fn(dy) = begin
		grad_leak = -0.1f0*1000
		g = @. ifelse(x >= 0.0f0, 
									0.0f0, 
									ifelse(x > lbound, -leaky_alpha*10f0, 0.0f0)) + 
						ifelse(x > 1.0f0, 
									grad_leak, 
									ifelse(x >= 0f0, 
												-leaky_alpha*100f0, 
												ifelse(x > -1.0f0, 
															-leaky_alpha*10f0, 
															0)))
		return (g .* dy, nothing)
	end
  # val = @. ifelse(x >= 0.0f0, 0.0f0, ifelse(x > lbound, x * reciproc, upval_bound)) + ifelse(x > 1.0f0, 0f0, ifelse(x > -1.0f0, x * leaky_alpha, leaky_alpha))
	val = @. ifelse(x > 0.0f0, 0.0f0, 1f0)
  return (val, my_pullback_fn)
end


stable_mask_mul(a, b) = a .* b
Zygote.@adjoint stable_mask_mul(a, mask) = begin
	a .* mask, dy -> begin
		dy .* 1f0, dy .* a
	end
end


Ŷ_sp_assign!(Ŷ_sp, Ŷ, rowcols) = begin
	Ŷ_sp .-= 30
	for (i, (row,col)) in enum(rowcols)
		Ŷ_sp[:,:,row,col] .+= Ŷ[:,:,i] .+ 30f0 .+ (row == col ? 0.3f0 : 0f0)
	end end
Ŷ_rev_assign!(gradŶ, grad, rowcols) = for (i, (row,col)) in enum(rowcols)
	gradŶ[:,:,i] .= grad[:,:,row,col]
end
mask_rev_assign!(gradŶ, grad, rowcols, meta) = nothing


include("Untracked.jl")


end # module Losses
