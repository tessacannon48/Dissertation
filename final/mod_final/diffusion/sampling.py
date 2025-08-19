import torch

# =============================================================================
# SAMPLING METHODS
# =============================================================================

def p_sample_loop_ddpm(model, scheduler, shape, cond, attrs, device):
    """DDPM sampling method."""
    # Start from pure random noise x_t
    x = torch.randn(shape).to(device)

    # Loop through all of the timesteps in the scheduler from T to 1
    for t in reversed(range(scheduler.timesteps)):

        # Create a batch-sized tensor with each element set to the timestep t
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        # Predict the clean input x_0 given the pure random noise and the timestep t 
        pred_x0 = model(x, cond, attrs, t_batch)

        # Get the alphas and betas for each timestep from the diffusion scheduler
        alpha_t = scheduler.alphas[t]
        alpha_cumprod_t = scheduler.alpha_cumprod[t]
        
        # Handle alpha_cumprod_prev
        if hasattr(scheduler, 'alpha_cumprod_prev'):
            alpha_cumprod_prev_t = scheduler.alpha_cumprod_prev[t]
        else:
            alpha_cumprod_prev_t = scheduler.alpha_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        
        beta_t = 1 - alpha_t

        # If we're not at the final time_step, add noise 
        if t > 0:
            noise = torch.randn_like(x)
        # At t=0, we stop adding noise and just return the predicted sample
        else:
            noise = torch.zeros_like(x)

        # Get coefficient for x_0
        coef1 = torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1 - alpha_cumprod_t)
        
        # Get coefficient for x_t
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)

        # Calculate posterior mean of x_t-1 given x_t
        mean = coef1 * pred_x0 + coef2 * x

        # Calculate the variance of x_t-1 given x_t
        var = beta_t * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)
        std = torch.sqrt(var)

        # Produce x_t-1 by sampling from the Gaussian 
        x = mean + std * noise

    # Return the final predicted sample
    return x


def p_sample_loop_ddim(model, scheduler, shape, cond, attrs, device, eta=0.0):
    """DDIM sampling method."""
    # Start from pure random noise x_t
    x = torch.randn(shape).to(device)

    # Loop through all of the timesteps in the scheduler from T to 1
    for i, t in enumerate(reversed(range(scheduler.timesteps))):

        # Create a batch-sized tensor with each element set to the timestep t
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        # Predict the clean input x_0 given the current sample and the timestep t 
        pred_x0 = model(x, cond, attrs, t_batch)

        # Get the alphas for each timestep from the diffusion scheduler
        alpha_cumprod_t = scheduler.alpha_cumprod[t]
        
        # Get the previous timestep (or 0 if this is the last step)
        if i < scheduler.timesteps - 1:
            t_prev = list(reversed(range(scheduler.timesteps)))[i + 1]
            alpha_cumprod_prev_t = scheduler.alpha_cumprod[t_prev]
        else:
            alpha_cumprod_prev_t = torch.tensor(1.0)

        # Calculate the predicted noise (epsilon)
        pred_epsilon = (x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(1 - alpha_cumprod_t)

        # Calculate the direction pointing towards x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev_t - eta**2 * (1 - alpha_cumprod_t / alpha_cumprod_prev_t)) * pred_epsilon

        # If we're not at the final timestep and eta > 0, add stochastic noise
        if i < scheduler.timesteps - 1 and eta > 0:
            noise = torch.randn_like(x)
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_prev_t)
        else:
            noise = torch.zeros_like(x)
            sigma_t = 0

        # Calculate the deterministic part: x_0 prediction scaled by alpha
        x0_part = torch.sqrt(alpha_cumprod_prev_t) * pred_x0

        # Produce x_t-1 using DDIM formula
        x = x0_part + dir_xt + sigma_t * noise

    # Return the final denoised sample
    return x


def p_sample_loop_plms(model, scheduler, shape, cond, attrs, device, order=4):
    """PLMS sampling method."""
    # Start from pure random noise x_t
    x = torch.randn(shape).to(device)
    
    # Store previous noise predictions for PLMS
    prev_eps = []

    # Loop through all of the timesteps in the scheduler from T to 1
    for i, t in enumerate(reversed(range(scheduler.timesteps))):

        # Create a batch-sized tensor with each element set to the timestep t
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        # Predict the clean input x_0 given the current sample and the timestep t 
        pred_x0 = model(x, cond, attrs, t_batch)

        # Get the alphas for each timestep from the diffusion scheduler
        alpha_cumprod_t = scheduler.alpha_cumprod[t]
        
        # Get the previous timestep (or 0 if this is the last step)
        if i < scheduler.timesteps - 1:
            t_prev = list(reversed(range(scheduler.timesteps)))[i + 1]
            alpha_cumprod_prev_t = scheduler.alpha_cumprod[t_prev]
        else:
            alpha_cumprod_prev_t = torch.tensor(1.0)

        # Calculate the predicted noise (epsilon)
        pred_epsilon = (x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(1 - alpha_cumprod_t)
        
        # Add current epsilon to the history
        prev_eps.append(pred_epsilon)
        
        # Keep only the required number of previous predictions
        if len(prev_eps) > order:
            prev_eps = prev_eps[-order:]

        # Calculate PLMS coefficients based on the number of stored predictions
        if len(prev_eps) == 1:
            # First step: use Euler method
            eps = prev_eps[-1]
        elif len(prev_eps) == 2:
            # Second step: use 2nd order Adams-Bashforth
            eps = (3/2) * prev_eps[-1] - (1/2) * prev_eps[-2]
        elif len(prev_eps) == 3:
            # Third step: use 3rd order Adams-Bashforth
            eps = (23/12) * prev_eps[-1] - (16/12) * prev_eps[-2] + (5/12) * prev_eps[-3]
        else:
            # Fourth step and beyond: use 4th order Adams-Bashforth
            eps = (55/24) * prev_eps[-1] - (59/24) * prev_eps[-2] + (37/24) * prev_eps[-3] - (9/24) * prev_eps[-4]

        # Calculate the deterministic part: x_0 prediction scaled by alpha
        x0_part = torch.sqrt(alpha_cumprod_prev_t) * pred_x0

        # Calculate the direction pointing towards x_t using PLMS-corrected epsilon
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev_t) * eps

        # At t=0, we stop adding noise and just return the predicted sample
        if i == scheduler.timesteps - 1:
            noise = torch.zeros_like(x)
        else:
            noise = torch.zeros_like(x)  # PLMS is typically deterministic

        # Produce x_t-1 using PLMS formula
        x = x0_part + dir_xt + noise

    # Return the final denoised sample
    return x