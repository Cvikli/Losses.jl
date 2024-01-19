module Print

using ..Losses: mse_with_L2, softmax, softmax_with_L2, onehot

state_postfix(acc) = size(acc,2) > 1 ? " (states are zeroed)" : ""

@inline print_accuracy(acc,       ::Any)                      = @info "Acc: $acc"                                                             * state_postfix(acc)
@inline print_accuracy(acc, accv, ::Any)                      = @info "Acc: $acc / $accv"                                                     * state_postfix(acc)
@inline print_accuracy(acc,       ::typeof(onehot))           = @info "Acc: $(round(acc*100f0, digits=2))%"                                   * state_postfix(acc)
@inline print_accuracy(acc, accv, ::typeof(onehot))           = @info "Acc: $(round(acc*100f0, digits=2))% / $(round(accv*100f0, digits=2))%" * state_postfix(acc)
@inline print_accuracy(acc,       ::typeof(softmax_with_L2))  = @info "Acc: $(round(acc*100f0, digits=2))%"                                   * state_postfix(acc)
@inline print_accuracy(acc, accv, ::typeof(softmax_with_L2))  = @info "Acc: $(round(acc*100f0, digits=2))% / $(round(accv*100f0, digits=2))%" * state_postfix(acc)

end