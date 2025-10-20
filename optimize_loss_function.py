from loss_function_and_gradient import loss_and_grad, Q
import numpy as np


def update_Q(Q_dict, grad_dict, step_size):
    if Q_dict.keys() == grad_dict.keys():
        for key in Q_dict:
            Q_dict[key] = Q_dict[key] - step_size * grad_dict[key]
    return Q_dict

def gradient_descent(initial_Q, step_size, max_iters, tol=1e-6):
    Q_current = {k: v.copy() for k, v in initial_Q.items()}
    loss, grad = loss_and_grad(Q_current)
    for _ in range(max_iters):
        if grad is None:
            break
        grad_norm = sum(np.linalg.norm(g) for g in grad.values())
        if grad_norm < tol:
            break
        Q_current = update_Q(Q_current, grad, step_size)
        loss, grad = loss_and_grad(Q_current)
    return Q_current, loss

if __name__ == "__main__":
    Q_opt, final_loss = gradient_descent(Q, 10, 1000)
    print(final_loss)