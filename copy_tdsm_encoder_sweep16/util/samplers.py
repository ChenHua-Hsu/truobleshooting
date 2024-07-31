import functools, torch, tqdm
import numpy as np

class pc_sampler:
    def __init__(self, sde, padding_value, snr=0.2, sampler_steps=100, steps2plot=(), device='cuda', eps=1e-3, jupyternotebook=False, serialized_model=False):
        ''' Generate samples from score based models with Predictor-Corrector method
            Args:
            score_model: A PyTorch model that represents the time-dependent score-based model.
            marginal_prob_std: A function that gives the std of the perturbation kernel
            diffusion_coeff: A function that gives the diffusion coefficient 
            of the SDE.
            batch_size: The number of samplers to generate by calling this function once.
            num_steps: The number of sampling steps. 
            Equivalent to the number of discretized time steps.    
            device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
            eps: The smallest time step for numerical stability.

            Returns:
                samples
        '''
        self.sde = sde
        self.diffusion_coeff_fn = functools.partial(self.sde.sde)
        self.snr = snr
        self.padding_value = padding_value
        self.sampler_steps = sampler_steps
        self.steps2plot = steps2plot
        self.device = device
        self.eps = eps
        self.jupyternotebook = jupyternotebook
        
        # Dictionary objects hold lists of variable values at various stages of diffusion process
        # Used for visualisation and diagnostic purposes only
        
        # Objects for one cloud through all diffusion steps
        self.step_scores = { x:[] for x in range(0,sampler_steps) }
        self.step_hite = { x:[] for x in range(0,sampler_steps) }
        self.step_hitx = { x:[] for x in range(0,sampler_steps) }
        self.step_hity = { x:[] for x in range(0,sampler_steps) }
        self.step_hitz = { x:[] for x in range(0,sampler_steps) }
        
        self.step_revdrift = { x:[] for x in range(0,sampler_steps) }
        
        # Objects for all clouds at selected diffusion steps
        self.hit_energy_stages = { x:[] for x in self.steps2plot }
        self.hit_x_stages = { x:[] for x in self.steps2plot }
        self.hit_y_stages = { x:[] for x in self.steps2plot }
        self.deposited_energy_stages = { x:[] for x in self.steps2plot }
        self.av_x_stages = { x:[] for x in self.steps2plot }
        self.av_y_stages = { x:[] for x in self.steps2plot }
        self.incident_e_stages = { x:[] for x in self.steps2plot }
        
        self.serialized_model = serialized_model
        self.hist_bins = [np.linspace(-5., 5., 101), np.linspace(-0.25, sampler_steps + 0.75, sampler_steps*2 + 2)] 
        self.hist = None
        self.mean_e = []
        self.std_e = []
        self.mean_x = []
        self.std_x = []
        self.mean_y = []
        self.std_y = []
        self.mean_z = []
        self.std_z = []

    def random_sampler(self, pdf,xbin):
        myCDF = np.zeros_like(xbin,dtype=float)
        myCDF[1:] = np.cumsum(pdf)
        a = np.random.uniform(0, 1)
        return xbin[np.argmax(myCDF>=a)-1]

    def get_prob_dist(self, x,y,nbins):
        '''
        2D histogram:
        x = incident energy per shower
        y = # valid hits per shower
        '''
        hist,xbin,ybin = np.histogram2d(x,y,bins=nbins,density=False)
        # Normalise histogram
        sum_ = hist.sum(axis=-1)
        sum_ = sum_[:,None]
        hist = hist/sum_
        # Remove NaN
        hist[np.isnan(hist)] = 0.0
        return hist, xbin, ybin

    def generate_hits(self, prob, xbin, ybin, x_vals, n_features, device='cpu'):
        '''
        prob = 2D PDF of nhits vs incident energy
        x/ybin = histogram bins
        x_vals = sample of incident energies (sampled from GEANT4)
        n_features = # of feature dimensions e.g. (E,X,Y,Z) = 4
        Returns:
        pred_nhits = array of nhit values, one for each shower
        y_pred = array of tensors (one for each shower) of initial noise values for features of each hit, sampled from normal distribution
        '''
        # bin index each incident energy falls into
        ind = np.digitize(x_vals, xbin) - 1
        ind[ind==len(xbin)-1] = len(xbin)-2
        ind[ind==-1] = 0
        # Construct list of nhits for given incident energies
        prob_ = prob[ind,:]
        
        y_pred = []
        pred_nhits = []
        for i in range(len(prob_)):
            nhits = int(self.random_sampler(prob_[i],ybin + 1))
            pred_nhits.append(nhits)
            # Generate random values for features in all hits
            #ytmp = torch.normal(0,1,size=(nhits, n_features), device=device)
            ytmp = self.sde.prior_sampling((nhits,4))
            y_pred.append( ytmp )
        return pred_nhits, y_pred
    
    def __call__(self, score_model, sampled_energies, init_x, batch_size=1, diffusion_on_mask=False, corrector_steps=100):
       
        # Padding masks defined by initial # hits / zero padding
        attn_padding_mask = (init_x[:,:,0] == self.padding_value).type(torch.bool)
        # Time array
        t = torch.ones(batch_size, device=self.device)
        # mask avoids perturbing padded values
        mask_tensor  = (~attn_padding_mask).float()[...,None]
        # Create array of time steps
        time_steps = np.linspace(1., self.eps, self.sampler_steps)
        step_size = time_steps[0]-time_steps[1]
        if self.jupyternotebook:
            time_steps = tqdm.notebook.tqdm(time_steps)
        
        x = init_x
        
        diffusion_step_ = 0
        with torch.no_grad():
            # Matrix multiplication in GaussianFourier projection doesnt like float64
            sampled_energies = sampled_energies.to(x.device, torch.float32)
            # Noise to add to input
            z = torch.normal(0,1,size=x.shape, device=x.device)
           
            # Diffusion flow plot
            x_to_hist = x[:,:,0].view(-1).cpu().numpy().copy()
            t_to_hist = np.ones(len(x_to_hist)) * 0
            hist_,_,_ = np.histogram2d(x_to_hist, t_to_hist, bins = self.hist_bins)
            if self.hist is None:
                self.hist = hist_

            # Iterate through time steps
            for time_idx, time_step in enumerate(time_steps):
                
                # Input shower = noise * std from SDE
                if not diffusion_on_mask:
                    x = x*mask_tensor
                    z = z*mask_tensor
                # if not self.jupyternotebook:
                #     print(f"Sampler step: {time_step:.4f}") 
                batch_time_step = torch.ones(batch_size, device=x.device) * time_step
                alpha = torch.ones_like(torch.tensor(time_step))
                # Calculate gradients
                if self.serialized_model:
                    grad = score_model([x, batch_time_step, sampled_energies, attn_padding_mask])
                else:
                    grad = score_model(x, batch_time_step, sampled_energies, mask=attn_padding_mask)
                
                nc_steps = corrector_steps
                self.step_scores[diffusion_step_].extend(grad[1,:,0].cpu().tolist() )
                self.step_hite[diffusion_step_].extend(x[1,:,0].cpu().tolist())
                self.step_hitx[diffusion_step_].extend(x[1,:,1].cpu().tolist())
                self.step_hity[diffusion_step_].extend(x[1,:,2].cpu().tolist())
                self.step_hitz[diffusion_step_].extend(x[1,:,3].cpu().tolist())
                
                # Ensure the first set of values for plots are the unperturbed inputs
                if diffusion_step_ == 0:
                    step_incident_e = []
                    step_hit_e = []
                    step_hit_x = []
                    step_hit_y = []
                    step_deposited_energy = []
                    step_av_x_pos = []
                    step_av_y_pos = []
                    for shower_idx in range(0,len(x)):
                        all_ine = sampled_energies[shower_idx].cpu().numpy().reshape(-1,1)
                        all_ine = all_ine.flatten().tolist()
                        step_incident_e.extend( all_ine )
                        
                        all_e = x[shower_idx,:,0].cpu().numpy().reshape(-1,1)
                        total_deposited_energy = np.sum( all_e )
                        all_e = all_e.flatten().tolist()
                        step_hit_e.extend( all_e )
                        step_deposited_energy.extend( [total_deposited_energy] )
                        
                        all_x = x[shower_idx,:,1].cpu().numpy().reshape(-1,1)
                        av_x_position = np.mean( all_x )
                        all_x = all_x.flatten().tolist()
                        step_hit_x.extend(all_x)
                        step_av_x_pos.extend( [av_x_position] )
                        
                        all_y = x[shower_idx,:,2].cpu().numpy().reshape(-1,1)
                        av_y_position = np.mean( all_y )
                        all_y = all_y.flatten().tolist()
                        step_hit_y.extend(all_y)
                        step_av_y_pos.extend( [av_y_position] )
                    self.incident_e_stages[diffusion_step_].extend(step_incident_e)
                    self.hit_energy_stages[diffusion_step_].extend(step_hit_e)
                    self.hit_x_stages[diffusion_step_].extend(step_hit_x)
                    self.hit_y_stages[diffusion_step_].extend(step_hit_y)
                    self.deposited_energy_stages[diffusion_step_].extend(step_deposited_energy)
                    self.av_x_stages[diffusion_step_].extend(step_av_x_pos)
                    self.av_y_stages[diffusion_step_].extend(step_av_y_pos)
                
                # Corrector step (Langevin MCMC)
                for n_ in range(nc_steps):
                    # Langevin corrector
                    noise = torch.normal(0,1,size=x.shape, device=x.device)
                    # Mask Langevin noise
                    if not diffusion_on_mask:
                        noise = noise*mask_tensor
                    # Step size calculation: snr * ratio of gradients in noise / prediction used to calculate
                    flattened_scores = grad.reshape(grad.shape[0], -1)
                    grad_norm = torch.linalg.norm( flattened_scores, dim=-1 ).mean()
                    flattened_noise = noise.reshape(noise.shape[0],-1)
                    noise_norm = torch.linalg.norm( flattened_noise, dim=-1 ).mean()
                    # Langevin step-size
                    langevin_step_size = 2 * alpha * (self.snr * noise_norm / grad_norm)**2
                    # Adjust inputs according to scores using Langevin iteration rule
                    x_mean = x + langevin_step_size * grad
                    x = x_mean + torch.sqrt(2 * langevin_step_size) * noise
                    if not diffusion_on_mask:
                        x = x*mask_tensor
              
                x_to_hist = x[:,:,0].view(-1).cpu().numpy().copy()
                t_to_hist = np.ones(len(x_to_hist)) * (time_idx + 0.5)
                hist_,_,_ = np.histogram2d(x_to_hist, t_to_hist, bins = self.hist_bins)
                if self.hist is None:
                  self.hist = hist_
                else:
                  self.hist = self.hist + hist_
                # Euler-Maruyama Predictor
                # Adjust inputs according to scores
                
                ### Euler-Maruyama Predictor ###
                
                # Forward SDE coefficients
                drift, diff = self.diffusion_coeff_fn(x,batch_time_step)
                
                # Drift term for reverse SDE 
                if self.serialized_model:
                    drift = drift - (diff**2)[:, None, None] * score_model([x, batch_time_step, sampled_energies, attn_padding_mask])
                else:
                    drift = drift - (diff**2)[:, None, None] * score_model(x, batch_time_step, sampled_energies, mask=attn_padding_mask)
                
                # Take step according to drift
                x_mean = x - drift*step_size
                
                if not diffusion_on_mask:
                    x_mean = x_mean*mask_tensor
                
                # Add the diffusion term of the reverse SDE
                x = x_mean + torch.sqrt(diff**2*step_size)[:, None, None] * z

                x_to_hist = x[:,:,0].view(-1).cpu().numpy().copy()
                t_to_hist = np.ones(len(x_to_hist)) * (time_idx + 1)
                hist_,_,_ = np.histogram2d(x_to_hist, t_to_hist, bins = self.hist_bins)
                if self.hist is None:
                  self.hist = hist_
                else:
                  self.hist = self.hist + hist_          

                
                ### For plots of reverse drift/diffusion ###
                self.step_revdrift[diffusion_step_].extend(drift[1,:,0].cpu().tolist() )
                
                # Store distributions at different stages of diffusion (for visualisation purposes only)
                if diffusion_step_ in self.steps2plot and diffusion_step_!=0:
                    step_incident_e = []
                    step_hit_e = []
                    step_hit_x = []
                    step_hit_y = []
                    step_deposited_energy = []
                    step_av_x_pos = []
                    step_av_y_pos = []
                    for shower_idx in range(0,len(x_mean)):
                        all_ine = sampled_energies[shower_idx].cpu().numpy().reshape(-1,1)
                        all_ine = all_ine.flatten().tolist()
                        step_incident_e.extend( all_ine )
                        
                        all_e = x_mean[shower_idx,:,0].cpu().numpy().reshape(-1,1)
                        total_deposited_energy = np.sum( all_e )
                        all_e = all_e.flatten().tolist()
                        step_hit_e.extend( all_e )
                        step_deposited_energy.extend( [total_deposited_energy] )
                        
                        all_x = x_mean[shower_idx,:,1].cpu().numpy().reshape(-1,1)
                        av_x_position = np.mean( all_x )
                        all_x = all_x.flatten().tolist()
                        step_hit_x.extend(all_x)
                        step_av_x_pos.extend( [av_x_position] )
                        
                        all_y = x_mean[shower_idx,:,2].cpu().numpy().reshape(-1,1)
                        av_y_position = np.mean( all_y )
                        all_y = all_y.flatten().tolist()
                        step_hit_y.extend(all_y)
                        step_av_y_pos.extend( [av_y_position] )
                    self.incident_e_stages[diffusion_step_].extend(step_incident_e)
                    self.hit_energy_stages[diffusion_step_].extend(step_hit_e)
                    self.hit_x_stages[diffusion_step_].extend(step_hit_x)
                    self.hit_y_stages[diffusion_step_].extend(step_hit_y)
                    self.deposited_energy_stages[diffusion_step_].extend(step_deposited_energy)
                    self.av_x_stages[diffusion_step_].extend(step_av_x_pos)
                    self.av_y_stages[diffusion_step_].extend(step_av_y_pos)
                non_zero_mask = x[:,:,0] != 0

                # Extract non-zero elements
                non_zero_elements = x[non_zero_mask]
                #print(non_zero_elements)
                self.mean_e.append(torch.mean(non_zero_elements[:,0]).item())
                self.mean_x.append(torch.mean(non_zero_elements[:,1]).item())
                #print(torch.mean(non_zero_elements[:,2]).item())
                self.mean_y.append(torch.mean(non_zero_elements[:,2]).item())
                self.mean_z.append(torch.mean(non_zero_elements[:,3]).item())
                self.std_e.append(torch.std(non_zero_elements[:,0]).item())
                self.std_x.append(torch.std(non_zero_elements[:,1]).item())
                self.std_y.append(torch.std(non_zero_elements[:,2]).item())
                self.std_z.append(torch.std(non_zero_elements[:,3]).item())
                
                
                #print if there is any nan in mean

                if torch.isnan(x_mean).any():
                   with open ("nan.txt","a") as file:
                      file.write("nan found in mean\n")
                else:
                   with open ("nan.txt","a") as file:
                      file.write("no nan found in mean\n")
                      
                   
                
                diffusion_step_+=1
                
        # Do not include noise in last step
        return x_mean

class new_pc_sampler:
    def __init__(self, sde, padding_value, snr=0.2, sampler_steps=100, steps2plot=(), device='cuda', eps=1e-3, jupyternotebook=False, serialized_model=False):
        ''' Generate samples from score based models with Predictor-Corrector method
            Args:
            score_model: A PyTorch model that represents the time-dependent score-based model.
            marginal_prob_std: A function that gives the std of the perturbation kernel
            diffusion_coeff: A function that gives the diffusion coefficient 
            of the SDE.
            batch_size: The number of samplers to generate by calling this function once.
            num_steps: The number of sampling steps. 
            Equivalent to the number of discretized time steps.    
            device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
            eps: The smallest time step for numerical stability.

            Returns:
                samples
        '''
        self.sde = sde
        self.diffusion_coeff_fn = functools.partial(self.sde.sde)
        self.snr = snr
        self.padding_value = padding_value
        self.sampler_steps = sampler_steps
        self.steps2plot = steps2plot
        self.device = device
        self.eps = eps
        self.jupyternotebook = jupyternotebook
        
        # Dictionary objects  hold lists of variable values at various stages of diffusion process
        # Used for visualisation and diagnostic purposes only
        self.hit_energy_stages = { x:[] for x in self.steps2plot}
        self.hit_x_stages = { x:[] for x in self.steps2plot}
        self.hit_y_stages = { x:[] for x in self.steps2plot}
        self.deposited_energy_stages = { x:[] for x in self.steps2plot}
        self.av_x_stages = { x:[] for x in self.steps2plot}
        self.av_y_stages = { x:[] for x in self.steps2plot}
        self.incident_e_stages = { x:[] for x in self.steps2plot}
        self.serialized_model = serialized_model

    def __call__(self, score_model, sampled_energies, init_x, batch_size=1, diffusion_on_mask=False):
        
        # Time array
        #t = torch.ones(batch_size, device=self.device)
        # Padding masks defined by initial # hits / zero padding
        padding_mask = (init_x[:,:,0]== self.padding_value).type(torch.bool)
        mask_tensor  = (~padding_mask).float()[...,None]
        # Create array of time steps
        #time_steps = np.linspace(1., self.eps, self.sampler_steps)
        #step_size = time_steps[0]-time_steps[1]
        t=1.
        h=0.01
        epsilon_rel=0.05
        theta=0.9
        r=0.9

        #r=0.5

        #if self.jupyternotebook:
            #time_steps = tqdm(range(time_steps))
            #tqdm(time_steps, desc="Processing")
        
        # Input shower is just some noise * std from SDE
        x = init_x
        x_pre=x
        
        diffusion_step_ = 0

        diffusion_step_plot=0
        with torch.no_grad():
            # Matrix multiplication in GaussianFourier projection doesnt like float64
            sampled_energies = sampled_energies.to(x.device, torch.float32)
            
            # Iterate through time steps
            while (t-h)>0. and h!=0:
                
                if not self.jupyternotebook:
                    print(f"Sampler step: {t:.4f}") 
                
                batch_time_step = torch.ones(batch_size, device=x.device) * t
                batch_time_step_pron = torch.ones(batch_size, device=x.device) * (t-h)


                #alpha = torch.ones_like(torch.tensor(t))

                # Corrector step (Langevin MCMC)
                # Noise to add to input
                z = torch.normal(0,1,size=x.shape, device=x.device)
                
                # Conditional score prediction gives estimate of noise to remove
                if not diffusion_on_mask:
                    x = x*mask_tensor
                    z = z*mask_tensor

                if self.serialized_model:
                  grad = score_model([x, batch_time_step, sampled_energies, padding_mask])
                else:
                  grad = score_model(x, batch_time_step, sampled_energies, mask=padding_mask)

                drift, diff = self.diffusion_coeff_fn(x,batch_time_step)

                #print("drift:",drift)
                #print("diff:",diff)

                drift=drift.to(self.device)
                diff=diff.to(self.device)
                #score_model=score_model.to('cuda:0') 

                x_pron=x-h*drift+h*(diff**2)[:,None,None]*grad+h**0.5*(diff)[:,None,None]*z

                #print("x':",x_pron)

                drift_pron, diff_pron = self.diffusion_coeff_fn(x_pron,batch_time_step_pron) 

                #print("drift_pron:",drift_pron)
                #print("diff_pron:",diff_pron)
                #print("score_model:",score_model(x_pron, batch_time_step_pron, sampled_energies, mask=padding_mask))

                drift_pron=drift_pron.to(self.device)
                diff_pron=diff_pron.to(self.device)
                #score_model=score_model.to('cuda:0')

                if not diffusion_on_mask:
                  x_pron = x_pron * mask_tensor

                if self.serialized_model:
                  grad_pron = score_model([x_pron, batch_time_step_pron, sampled_energies, padding_mask])
                else:
                  grad_pron = score_model(x_pron, batch_time_step_pron, sampled_energies, mask=padding_mask)


                x_tilta=x-h*drift_pron+h*(diff_pron**2)[:,None,None]*grad_pron+h**0.5*(diff_pron)[:,None,None]*z


                if not diffusion_on_mask:
                  x_tilta = x_tilta * mask_tensor
                #print("x_tilta:",x_tilta)

                x_pron_pron=0.5*(x_pron+x_tilta)

                epsilon_abs = torch.ones_like(x_pron, device=x.device) * 0.1

                delta=np.maximum(epsilon_abs.cpu(),epsilon_rel*np.maximum(torch.abs(x_pron.cpu()),torch.abs(x_pre.cpu())))

                #print("delta:",delta)

                x=x.to(self.device)
                x_pron=x_pron.to(self.device)
                x_pron_pron=x_pron_pron.to(self.device)
                delta=delta.to(self.device)

                #E=1/((x.shape[0])**(0.5))*torch.linalg.norm(((x_pron-x_pron_pron)/delta).reshape(x.shape[0],-1), dim=-1).mean()
                E=1./((x.shape[0])**(0.5))*torch.linalg.norm(((x_pron-x_pron_pron)/delta))

                #print("E:",E)

                #print("t:",t)
                #print("h:",h)

                if E <= 1 :
                    x=x_pron_pron
                    t=t-h
                    x_pre=x_pron

                    #print("hello")
                        

                       
                    diffusion_step_+=1

                    #print("diffusion step:", diffusion_step_)
                h=min(t,theta*h*E**(-r))
                
        # Do not include noise in last step
        #x_mean = x_mean
        if not diffusion_on_mask:
          x = x * mask_tensor
        return x  
