class Args:
    """parameter"""
    batch_size = 100
    epochs = 500
    input_shape = (10, 1)
    output_shape = (10, 1)
    lr = 0.001
    mu1 = 100.0
    mu2 = 0.01
    nu1 = 0.1
    nu2 = 0.1
    num_samples = [7, 5, 3]
    param_b = 1.0
    param_c = 1.0
    param_d = 0.5
    param_init = True
    param_b_init = 0.6
    param_c_init = 0.6
    param_d_init = 0.1
    rho = 0.15
    sigma_i = 0
    sigma_p = 1
    project = "MAS"
    name = "test"
    logs_dir = "logs"
    device = 'cpu'
