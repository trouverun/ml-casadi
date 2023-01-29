import time

import numpy as np
import torch
import casadi as cs
import ml_casadi.torch as mc
import matplotlib.pyplot as plt


def target_fun(x):
    return torch.stack([5*torch.sin(2*x[:, 0]), 5*torch.cos(2*x[:, 1])], dim=1)


def example():
    casadi_sym_inp = cs.MX.sym('inp', 2)

    inputs = torch.stack([torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100)], dim=1)
    targets = target_fun(inputs)

    wc = 1
    ws = 2
    ww = np.ones(2)

    model2 = mc.nn.GPModel(inputs, targets, input_size=2, output_size=2, device="cuda:0")
    casadi_lin_approx_sym_out = model2.approx(casadi_sym_inp, order=1, weakening_cutoff=wc, weakening_strength=ws, difference_weights=ww)
    casadi_lin_approx_func = cs.Function('model2_lin',
                                         [casadi_sym_inp,
                                          model2.sym_approx_params(flat=True, order=1)],
                                         [casadi_lin_approx_sym_out])
    lin_i = np.random.randint(0, len(inputs))
    lin_p = inputs[lin_i]
    t1 = time.time_ns()
    casadi_lin_approx_param = model2.approx_params(flat=True, order=1, a=lin_p.unsqueeze(0).numpy())
    t2 = time.time_ns()
    print("Took %d ms for 1st order taylor" % ((t2-t1)/1e6))

    casadi_quad_approx_sym_out = model2.approx(casadi_sym_inp, order=2, weakening_cutoff=wc, weakening_strength=ws, difference_weights=ww)
    casadi_quad_approx_func = cs.Function('model2_quad',
                                          [casadi_sym_inp,
                                           model2.sym_approx_params(flat=True, order=2)],
                                          [casadi_quad_approx_sym_out])
    t1 = time.time_ns()
    casadi_quad_approx_param = model2.approx_params(flat=True, order=2, a=lin_p.unsqueeze(0).numpy())
    t2 = time.time_ns()
    print("Took %d ms for 2nd order taylor" % ((t2-t1)/1e6))

    casadi_lin_out = []
    casadi_quad_out = []

    # Casadi can not handle batches
    for i in range(100):
        casadi_lin_out.append(casadi_lin_approx_func(inputs[i].numpy(), casadi_lin_approx_param))
        casadi_quad_out.append(casadi_quad_approx_func(inputs[i].numpy(), casadi_quad_approx_param))

    casadi_lin_out = np.array(casadi_lin_out)
    casadi_quad_out = np.array(casadi_quad_out)

    import matplotlib
    matplotlib.use('QtAgg')
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))
    fig.tight_layout()

    model_outs = model2(inputs.cuda()).detach().cpu()
    for i in range(2):
        ax[i].plot(targets[:, i], label='Target', linewidth=4)
        ax[i].plot(casadi_lin_out[:, 0, i], label='Casadi Linear')
        ax[i].plot(casadi_quad_out[:, 0, i], label='Casadi Quadratic')
        ax[i].plot(model_outs[:, 0, i], label="Full GP")
        ax[i].fill_between(np.arange(len(inputs)), casadi_lin_out[:, 0, i] - casadi_lin_out[:, 1, i], casadi_lin_out[:, 0, i] + casadi_lin_out[:, 1, i], label='linear std', alpha=0.5)
        ax[i].fill_between(np.arange(len(inputs)), casadi_quad_out[:, 0, i] - casadi_quad_out[:, 1, i], casadi_quad_out[:, 0, i] + casadi_quad_out[:, 1, i], label='quad std', alpha=0.5)
        ax[i].fill_between(np.arange(len(inputs)), model_outs[:, 0, i] - model_outs[:, 1, i], model_outs[:, 0, i] + model_outs[:, 1, i], label='real std', alpha=0.5)
        ax[i].plot(lin_i, targets[lin_i, i], 'k*', markersize=10)
        ax[i].set_ylim(-7.5, 7.5)
        ax[i].legend()

    plt.show()


if __name__ == '__main__':
    example()
