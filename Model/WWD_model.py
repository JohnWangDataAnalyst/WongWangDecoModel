## wong-wang Deco Model with BOLD

## author: Zheng Wang


import numpy as np
class WongWangDeco():
    param = {
    
    # Parameters for the integration
    "dt"         : 0.05,     # integration step size   
    "ROI_num" : 96,       # number of neural nodes
    "sigma"      : 0.02,    # standard deviation of the Gaussian noise
    
    # Parameters for the ODEs
    # Excitatory population
    "W_E" : 1.,              # scale of the external input
    "tau_E" : 100.,          # decay time
    "gamma_E" : 0.641/1000.,       # other dynamic parameter (?)
    
    # Inhibitory population
    "W_I" : 0.7,             # scale of the external input
    "tau_I" : 10.,           # decay time
    "gamma_I" : 1./1000.,          # other dynamic parameter (?)
    
    # External input
    "I_0" : 0.32,          # external input
    "I_external" : 0.,       # external stimulation
    
    # Coupling parameters
    "g" : 20.,               # global coupling (from all nodes E_j to single node E_i)
    "g_EE" : .1,            # local self excitatory feedback (from E_i to E_i)
    "g_IE" : .1,            # local inhibitory coupling (from I_i to E_i)
    "g_EI" : 0.1,            # local excitatory coupling (from E_i to I_i)
    "lamb" : 0.,             # scale of global coupling on I_i (compared to E_i)
       
    "aE":310,
    "bE" :125,
    "dE":0.16,
    "aI":615,
    "bI" :177, 
    "dI" :0.087, 
       
    # Output (BOLD signal)
   
    "alpha" : 0.32,
    "rho" : 0.34,
    "k1" : 2.38,
    "k2" : 2.0,
    "k3" : 0.48, # adjust this number from 0.48 for BOLD fluctruate around zero
    "V" : .2,
    "E0" : 0.34, 
    "tau_s" : 0.65,
    "tau_f" : 0.41,
    "tau_0" : 0.98
   
    } ### initial values of all model prameters
    
    kI= 0 ### label for using I as input to bolloon model
    num_param = 5 
    num_state = 6
    
    theta_name = ['g', 'g_EE', 'g_IE', 'g_EI', 'sigma']
    state_name = ['E', 'I', 'q', 'v', 'f', 'x']
    
    ##### ADAM hyperparaneters
    beta_1 = 0.9
    beta_2 = 0.999
    lamda = 0# decay term for convergence
    epsilon = 1e-5
    
    ### initials of model parameters
    theta = np.array([80., .15, .15, 0.5, 1.])  ## inital gues of model parameters
    
    ### inital for Adam parameters
    m=np.zeros((num_param,))
    v=np.zeros((num_param,))
    t=1
    
    #### initals for states and gradient states
    x= 0.45*np.random.uniform(0.,1.0,(num_state, param['ROI_num']))
    x[1] = 0.2*x[1]
    xg = np.zeros((num_state*num_param,param['ROI_num']))
    
    def __init__(self, sc, ts, Tr):
        
        self.Tr = Tr              # time interval between two data points
        
        if sc.shape[0] == sc.shape[1]: ### SC is a square matrix 
            self.param['ROI_num'] = sc.shape[0] ## define the number of nodes
            self.sc = sc   #### define the structural connectivity 
            self.l_s = -(np.diag(np.sum(sc, axis=1)) -sc) ### define negative Laplacian
            
        if ts.shape[1]  == self.param['ROI_num']: ## check ts shape: shape 1 should the same as the number of nodes
            self.ts = ts   #### define fMRI time series which is a matrix with num_datapoints X num of nodes
            
            
            total_t = Tr*len(ts)  # total time length of the time series
            dt = self.param['dt']

            self.n_integration_steps = int(total_t/dt)
            self.n_steps_keep_results = int(Tr/dt)
        else:
            print("TS should be a datanum *" + str(self.param['ROI_num']) + 'matrix')
    ##### Wong-wang deco + Balloon BOLD
    def derivative_orig(self): ### var is a vectior with 6 which is arragned by: E, I from Neural Model 
                                                                               ###    q, v, f, x from Balloon Model
        def h_tf(a, b, d, x):
            return (a*x-b)/(1.0000 -np.exp(-d*(a*x-b)))

        def smooth_normalize(x):
            x[x< 0.000001] =0.000001
            return x

        # Equations for the neural masses
        E = self.x[0]
        I = self.x[1]
        kI= self.kI #### input for bold with or without I

        W_E = self.param["W_E"]
        tau_E = self.param["tau_E"]
        gamma_E = self.param["gamma_E"]

        W_I = self.param["W_I"]
        tau_I = self.param["tau_I"]
        gamma_I = self.param["gamma_I"]

        I_0 = self.param["I_0"]
        I_external = self.param["I_external"]
        lamb = self.param["lamb"]
        g = self.param["g"]
        g_EE = self.param["g_EE"]
        g_IE = self.param["g_IE"]
        g_EI = self.param["g_EI"]
        aE = self.param["aE"]
        bE = self.param["bE"] 
        dE = self.param["dE"]
        aI = self.param["aI"]
        bI = self.param["bI"] 
        dI = self.param["dI"]

        IE = np.tanh( smooth_normalize(W_E*I_0 +g_EE*E +g*np.dot(self.l_s, E) -g_IE*I +I_external)) #  input currents for E
        II = np.tanh( smooth_normalize(W_I*I_0 +g_EI*E -I +lamb*g*np.dot(self.l_s, E))) #  input currents for I 
        rE = h_tf(aE, bE, dE, IE) # firing rate for E
        rI = h_tf(aI, bI, dI, II) # firing rate for I 
        ddE = -E/tau_E +gamma_E*(1.-E)*rE  ### equlibrim point at E=(tau_E*gamma_E*rE)/(1+tau_E*gamma_E*rE)
        ddI = -I/tau_I +gamma_I*rI         ### equlibrim point at I=tau_I*gamma_I*rI


        # Equations for the ouput (BOLD signal)
        q = self.x[2] ### Balloon components
        v = self.x[3] ### Balloon components
        f = self.x[4]  ## rCBF component
        x = self.x[5]  # Linear component


        alpha = self.param["alpha"]
        rho = self.param["rho"]

        tau_s = self.param["tau_s"]
        tau_f = self.param["tau_f"]
        tau_0 = self.param["tau_0"]


        dq = (f*(1.-(1.-rho)**(1./f))/rho -q*(v)**(1./alpha)/v)/tau_0 
        dv = (f -v**(1./alpha))/tau_0
        df = x
        dx = E-kI*I -x/tau_s -(f-1.)/tau_f


        return np.array([ddE, ddI, dq, dv, df, dx])

        #### sensitivity model (gradients of wong-wang + balloon model on g gEE gIE gEI sigma and V 5+1 model parameteres)

    def derivative_gradient(self, w): ### var a vector with 6(states)*5(model parameters), var_e a vection with 6 states
                                            ### w is noise (std= 1.0) vector with num of nodes

        def smooth_normalize(x):
            x[x< 0.000001] =0.000001
            return x
        
        def h_tf(a, b, d, x):
            return (a*x-b)/(1.0000 -np.exp(-d*(a*x-b)))
        
        def dh_tf(a,b,d, x):
    
            tmp_e=np.exp(-d*(a*x-b))
            tmp_d=1.-np.exp(-d*(a*x-b))
            slope_E=(a*tmp_d-(a*x-b)*d*a*tmp_e)/tmp_d**2
            return slope_E

        # Equations for the neural masses
        E = self.x[0]
        I = self.x[1]
        q = self.x[2]
        v = self.x[3]
        f = self.x[4]
        x = self.x[5]
        var = self.xg
        Edg = var[0]
        Idg = var[1]
        qdg = var[2]
        vdg = var[3]
        fdg = var[4]
        xdg = var[5]
        EdgEE = var[6]
        IdgEE = var[7]
        qdgEE = var[8]
        vdgEE = var[9]
        fdgEE = var[10]
        xdgEE = var[11]
        EdgIE = var[12]
        IdgIE = var[13]
        qdgIE = var[14]
        vdgIE = var[15]
        fdgIE = var[16]
        xdgIE = var[17]
        EdgEI = var[18]
        IdgEI = var[19]
        qdgEI = var[20]
        vdgEI = var[21]
        fdgEI = var[22]
        xdgEI = var[23]
        Edsig = var[24]
        Idsig = var[25]
        qdsig = var[26]
        vdsig = var[27]
        fdsig = var[28]
        xdsig = var[29]

        W_E = self.param["W_E"]
        tau_E = self.param["tau_E"]
        gamma_E = self.param["gamma_E"]

        W_I = self.param["W_I"]
        tau_I = self.param["tau_I"]
        gamma_I = self.param["gamma_I"]

        I_0 = self.param["I_0"]
        I_external = self.param["I_external"]
        lamb = self.param["lamb"]
        g = self.param["g"]
        g_EE = self.param["g_EE"]
        g_IE = self.param["g_IE"]
        g_EI = self.param["g_EI"]
        aE = self.param["aE"]
        bE = self.param["bE"] 
        dE = self.param["dE"]
        aI = self.param["aI"]
        bI = self.param["bI"] 
        dI = self.param["dI"]

        kg = 0.1 #### self feedback gain for sensitivity stability
        kon = 1.0 #### kon is label for using inplicivie derrivative 
        kI= self.kI #### the label for I giving Balloon Model
        
        ROI_num = self.param['ROI_num']

        IE_nocons =  W_E*I_0 +g_EE*E +g*np.dot(self.l_s, E) -g_IE*I +I_external
        IE = np.tanh(smooth_normalize(IE_nocons))
        II_nocons =  W_I*I_0 +g_EI*E -I +lamb*g*np.dot(self.l_s, E)
        II = np.tanh(smooth_normalize(II_nocons)) #0.4*np.tanh(2.5*(W_I*I_0 +g_EI*E -I +lamb*g*np.dot(l_s, E)))# 
        #II[II>0.4] =.4
        rE = h_tf(aE, bE, dE, IE)
        rI = h_tf(aI, bI, dI, II)
        c_E = (1. -E)*gamma_E*dh_tf(aE, bE, dE, IE)/np.cosh(IE_nocons)**2*(IE_nocons>0)
        c_I = gamma_I*dh_tf(aI, bI, dI, II)/np.cosh(II_nocons)**2*(II_nocons>0)
        dEdE = -1.0/tau_E -gamma_E*rE + c_E*g_EE*1.0
        dEdI = -c_E*g_IE
        dIdE = c_I*g_EI
        dIdI = -1./tau_I -c_I
        #ddE = -E/tau_E +gamma_E*(1.-E)*rE 
        #ddI = -I/tau_I +gamma_I*rI 
        dEdg =  c_E*np.dot(self.l_s, E)+kon*(dEdI*Idg +dEdE*Edg) +c_E*g*np.dot(self.l_s, Edg) - kg*Edg
        dIdg = np.zeros((ROI_num)) +kon*(dIdE*Edg + dIdI*Idg) -kg*Idg

        dEdgEE =  c_E*E +kon*(dEdI*IdgEE +dEdE*EdgEE) +c_E*g*np.dot(self.l_s, EdgEE) - kg*EdgEE
        dIdgEE = np.zeros((ROI_num))+kon*(dIdE*EdgEE + dIdI*IdgEE)-kg*IdgEE

        dEdgIE =  - c_E*I +kon*(dEdI*IdgIE +dEdE*EdgIE)+c_E*g*np.dot(self.l_s, EdgIE)  - kg*EdgIE
        dIdgIE = np.zeros((ROI_num)) +kon*(dIdE*EdgIE + dIdI*IdgIE)-kg*IdgIE

        dEdgEI =  np.zeros((ROI_num)) +kon*(dEdI*IdgEI +dEdE*EdgEI) +c_E*g*np.dot(self.l_s, EdgEI) - kg*EdgEI 
        dIdgEI =  c_I*E + dIdE*EdgEI + kon*(dIdI*IdgEI -kg*IdgEI)

        dEdsig = dEdI*Idsig +dEdE*Edsig  +c_E*g*np.dot(self.l_s, Edsig)+w*0.01 - kg*Edsig 
        dIdsig = dIdE*EdgEI + dIdI*IdgEI-kg*Idsig

        #print(g, g_EE, g_IE, g_EI)

        alpha = self.param["alpha"]
        rho = self.param["rho"]

        tau_s = self.param["tau_s"]
        tau_f = self.param["tau_f"]
        tau_0 = self.param["tau_0"]




        dxdg = Edg -kI*Idg-xdg/tau_s -fdg/tau_f #-Idg

        dfdg = xdg

        dvdg = fdg/tau_0 - (1./alpha/tau_0)*v**(1./alpha-1.0)*vdg
        dqdg = (fdg*(1.-(1.-rho)**(1./f))/rho +fdg/f*(1.-rho)**(1./f)/rho-qdg*(v)**(1./alpha)/v\
                -vdg*(1/alpha-1)*q*(v)**(-2.0+1./alpha))/tau_0
        dxdgEE = EdgEE -kI*IdgEE-xdgEE/tau_s -fdgEE/tau_f#-IdgEE

        dfdgEE = xdgEE

        dvdgEE = fdgEE/tau_0 - (1./alpha/tau_0)*v**(1./alpha-1.0)*vdgEE
        dqdgEE = (fdgEE*(1.-(1.-rho)**(1./f))/rho +fdgEE/f*(1.-rho)**(1./f)/rho-qdgEE*(v)**(1./alpha)/v\
                -vdgEE*(1/alpha-1)*q*(v)**(-2.0+1./alpha))/tau_0

        dxdgIE = EdgIE -kI*IdgIE-xdgIE/tau_s -fdgIE/tau_f#-IdgIE

        dfdgIE = xdgIE

        dvdgIE = fdgIE/tau_0 - (1./alpha/tau_0)*v**(1./alpha-1.0)*vdgIE
        dqdgIE = (fdgIE*(1.-(1.-rho)**(1./f))/rho +fdgIE/f*(1.-rho)**(1./f)/rho-qdgIE*(v)**(1./alpha)/v\
                -vdgIE*(1/alpha-1)*q*(v)**(-2.0+1./alpha))/tau_0

        dxdgEI = EdgEI -kI*IdgEI-xdgEI/tau_s -fdgEI/tau_f #-IdgEI 

        dfdgEI = xdgEI

        dvdgEI = fdgEI/tau_0 - (1./alpha/tau_0)*v**(1./alpha-1.0)*vdgEI
        dqdgEI = (fdgEI*(1.-(1.-rho)**(1./f))/rho +fdgEI/f*(1.-rho)**(1./f)/rho-qdgEI*(v)**(1./alpha)/v\
                -vdgEI*(1/alpha-1)*q*(v)**(-2.0+1./alpha))/tau_0

        dxdsig = Edsig-kI*Idsig-xdsig/tau_s -fdsig/tau_f #-IdgEI 

        dfdsig = xdsig

        dvdsig = fdsig/tau_0 - (1./alpha/tau_0)*v**(1./alpha-1.0)*vdsig
        dqdsig = (fdsig*(1.-(1.-rho)**(1./f))/rho +fdsig/f*(1.-rho)**(1./f)/rho-qdsig*(v)**(1./alpha)/v\
                -vdsig*(1/alpha-1)*q*(v)**(-2.0+1./alpha))/tau_0

        return np.array([dEdg, dIdg, dqdg, dvdg, dfdg, dxdg,\
                         dEdgEE, dIdgEE, dqdgEE, dvdgEE, dfdgEE, dxdgEE,\
                         dEdgIE, dIdgIE, dqdgIE, dvdgIE, dfdgIE, dxdgIE,\
                         dEdgEI, dIdgEI, dqdgEI, dvdgEI, dfdgEI, dxdgEI,
                         dEdsig, dIdsig, dqdsig, dvdsig, dfdsig, dxdsig]) 
    
    ### gradients caculation based on r correlation 
    def gradient_compute_r(self, results_x, results_g, empBOLD, simBOLD):
        
        w_c1 = 0.1 # weight on constraint gEE and gIE
        w_c2 =0.5  # weight on constraint gIE
        k1 = self.param["k1"]
        k2 = self.param["k2"]
        k3 = self.param["k3"]
        V = self.param["V"]
        g_EE = self.param["g_EE"]
        g_IE = self.param["g_IE"]
        g_EI = self.param["g_EI"]
        w = self.param["sigma"]
        E0 = self.param["E0"]
        
        grad = np.zeros((5))

        err=[]

        fc= np.corrcoef(empBOLD.T)
        edq_sum=0
        edv_sum=0
        q_ls =[]
        v_ls =[]
        q1dg_ls=[]
        q1dgEE_ls=[]
        q1dgIE_ls=[]
        q1dgEI_ls=[]
        q1dsig_ls=[]
        v1dg_ls=[]
        v1dgEE_ls=[]
        v1dgIE_ls=[]
        v1dgEI_ls=[]
        v1dsig_ls=[]

        for i in range(self.batch_size):
            xg1 = results_g[i]
            v = results_x[i][3]
            q = results_x[i][2]
            v_ls.append(v)
            q_ls.append(q)

            edq = V/E0*(-k1-k2/v)
            edv = V/E0*(-k3+k2*q/v**2)
            edq_sum += edq
            edv_sum += edv

            q1dg_ls.append(xg1[2])
            q1dgEE_ls.append(xg1[8])
            q1dgIE_ls.append(xg1[14])
            q1dgEI_ls.append(xg1[20])
            q1dsig_ls.append(xg1[26])
            v1dg_ls.append(xg1[2+1])
            v1dgEE_ls.append(xg1[8+1])
            v1dgIE_ls.append(xg1[14+1])
            v1dgEI_ls.append(xg1[20+1])
            v1dsig_ls.append(xg1[26+1])


        for i in range(1, self.param['ROI_num']):
            for j in range(i):
                rii =np.sum((simBOLD[:,i]-simBOLD[:,i].mean())*(simBOLD[:,i]-simBOLD[:,i].mean()))
                rjj =np.sum((simBOLD[:,j]-simBOLD[:,j].mean())*(simBOLD[:,j]-simBOLD[:,j].mean()))
                rij =np.sum((simBOLD[:,i]-simBOLD[:,i].mean())*(simBOLD[:,j]-simBOLD[:,j].mean()))
                e = rij/np.sqrt(rii)/np.sqrt(rjj)-fc[i,j]
                err.append(e**2)
                v1= np.array(v_ls)[:,i]
                q1= np.array(q_ls)[:,i]


                v2= np.array(v_ls)[:,j]
                q2= np.array(q_ls)[:,j]

                riidq1 = 2.0*(V/E0*(-k1-k2/v1)-edq_sum[i]/self.batch_size)*(simBOLD[:,i]-simBOLD[:,i].mean())
                rjjdq2 = 2.0* (V/E0*(-k1-k2/v2)-edq_sum[j]/self.batch_size)*(simBOLD[:,j]-simBOLD[:,j].mean())
                rijdq1 = (V/E0*(-k1-k2/v1)-edq_sum[i]/self.batch_size)*(simBOLD[:,j]-simBOLD[:,j].mean())
                rijdq2 = (V/E0*(-k1-k2/v2)-edq_sum[j]/self.batch_size)*(simBOLD[:,i]-simBOLD[:,i].mean())

                riidv1 = 2.0*(V/E0*(-k3+k2*q1/v1**2) -edv_sum[i]/self.batch_size)*(simBOLD[:,i]-simBOLD[:,i].mean())
                rjjdv2 = 2.0*(V/E0*(-k3+k2*q2/v2**2) -edv_sum[j]/self.batch_size)*(simBOLD[:,j]-simBOLD[:,j].mean())
                rijdv1 = (V/E0*(-k3+k2*q1/v1**2) -edv_sum[i]/self.batch_size)*(simBOLD[:,j]-simBOLD[:,j].mean())
                rijdv2 = (V/E0*(-k3+k2*q2/v2**2) -edv_sum[j]/self.batch_size)*(simBOLD[:,i]-simBOLD[:,i].mean())

                edq1 = rijdq1/np.sqrt(rjj)/np.sqrt(rii)-0.5*np.power(rii, -1.5)*riidq1/np.sqrt(rjj)*rij
                edq2 = rijdq2/np.sqrt(rjj)/np.sqrt(rii)-0.5*np.power(rjj, -1.5)*rjjdq2/np.sqrt(rii)*rij
                edv1 = rijdv1/np.sqrt(rjj)/np.sqrt(rii)-0.5*np.power(rii, -1.5)*riidv1/np.sqrt(rjj)*rij
                edv2 = rijdv2/np.sqrt(rjj)/np.sqrt(rii)-0.5*np.power(rjj, -1.5)*rjjdv2/np.sqrt(rii)*rij

                q2dg = np.array(q1dg_ls)[:,j]
                q2dgEE = np.array(q1dgEE_ls)[:,j]
                q2dgIE = np.array(q1dgIE_ls)[:,j]
                q2dgEI = np.array(q1dgEI_ls)[:,j]
                q2dsig = np.array(q1dsig_ls)[:,j]
                v2dg = np.array(v1dg_ls)[:,j]
                v2dgEE = np.array(v1dgEE_ls)[:,j]
                v2dgIE = np.array(v1dgIE_ls)[:,j]
                v2dgEI = np.array(v1dgEI_ls)[:,j]
                v2dsig = np.array(v1dsig_ls)[:,j]

                q1dg = np.array(q1dg_ls)[:,i]
                q1dgEE = np.array(q1dgEE_ls)[:,i]
                q1dgIE = np.array(q1dgIE_ls)[:,i]
                q1dgEI = np.array(q1dgEI_ls)[:,i]
                q1dsig = np.array(q1dsig_ls)[:,i]
                v1dg = np.array(v1dg_ls)[:,i]
                v1dgEE = np.array(v1dgEE_ls)[:,i]
                v1dgIE = np.array(v1dgIE_ls)[:,i]
                v1dgEI = np.array(v1dgEI_ls)[:,i]
                v1dsig = np.array(v1dsig_ls)[:,i]


                grad += e*np.array([np.sum(edq1*q1dg+edv1*v1dg +edq2*q2dg+edv2*v2dg), \
                                         np.sum(edq1*q1dgEE+edv1*v1dgEE + edq2*q2dgEE+edv2*v2dgEE)+w_c1*(g_EE+g_IE-0.3),\
                                         np.sum(edq1*q1dgIE+edv1*v1dgIE +edq2*q2dgIE+edv2*v2dgIE)+w_c1*(g_EE+g_IE-0.3), \
                                         np.sum(edq1*q1dgEI+edv1*v1dgEI+edq2*q2dgEI+edv2*v2dgEI)+w_c2*(g_EI-1.0), \
                                         np.sum(edq1*q1dsig+edv1*v1dsig+edq2*q2dsig+edv2*v2dsig)] )
        return grad, np.array(err).mean() ### output gradient and cost 
    
    ##### BOLD outputs
    def output_bold(self):

        q = self.x[2]#smooth_normalize_center(var[2], 1.)   # that's done already in the derivative function...
        v = self.x[3]#smooth_normalize_center(var[3], 1.)   # probably doesn't do much, all values close to the linear regime


        k1 = self.param["k1"]
        k2 = self.param["k2"]
        k3 = self.param["k3"]
        V = self.param["V"]
        E0 = self.param["E0"]


        y = k1*(1.-q) +k2*(1.-q/v) +k3*(1.-v)

        return V*y/E0
    
    
    def compute_noise(self):
    
        return self.param["sigma"]*np.random.randn(self.param["ROI_num"])
    
    ### get states of wong wang models and sensitivity models in batch
    def simBOLD_batch(self):
        
        ROI_num = self.param['ROI_num']
        dt = self.param['dt']
        
        
        results_sim=[]
        results_g=[]
        results_x=[]
        
        for i in range(self.batch_size*self.n_steps_keep_results):
            w=np.random.randn(ROI_num)
            self.xg = self.xg + self.derivative_gradient(w)*dt
            self.x = self.x + self.derivative_orig()*dt
            self.x[0] = self.x[0] +np.sqrt(dt)*self.param['sigma']*w
            #x[1] = x[1] #+np.sqrt(dt)*compute_noise(param)

            if (i+1) % self.n_steps_keep_results == 0:
                results_sim.append(self.output_bold())
                results_g.append(self.xg)
                results_x.append(self.x)
        return results_x, results_g, np.array(results_sim)
    
    ### ADAM algorithm for model parameters updates        
    def update_adam(self, grad):
        
        
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(grad, 2)
        m_hat = self.m / (1 - np.power(self.beta_1, self.t))
        v_hat = self.v / (1 - np.power(self.beta_2, self.t))
        self.theta = self.theta - self.alpha * (m_hat / (np.sqrt(v_hat) + self.epsilon)+self.lamda*self.theta)
        #print(m,v,mh, vh)

    
    def param_norm(self):
        self.param["g"] =  0.001+self.theta[0]*(self.theta[0]> 0)#50. +.2 *np.tanh(theta_opt[0]/.2)##5# #100. +10.1 *np.tanh(theta[0]/10.1)#
        self.param["g_EE"] =0.001+self.theta[1]*(self.theta[1]> 0)# 0.075+0.0749*np.tanh(theta_opt[1]/0.0749)#theta[1]*(theta[1]> 0)#0.1 +0.095 *np.tanh(theta[1]/0.095)#
        self.param["g_IE"] = 0.001+self.theta[2]*(self.theta[2]> 0)#0.075+0.0749*np.tanh(theta_opt[2]/0.0749)#0.03+0.029 *np.tanh(theta[2]/0.029)##0.1 +0.095 *np.tanh(theta[2]/0.095)#
        self.param["g_EI"] = 0.001+self.theta[3]*(self.theta[3]> 0)#0.5 +0.499 *np.tanh(theta_opt[3]/0.499)#0.1 +0.095 *np.tanh(theta[3]/0.095)#
        self.param["sigma"] = 0.01+0.01*self.theta[4]*(self.theta[4]> 0)#0.01+0.01*np.tanh(theta_opt[4])# 0.005 +0.0049 *np.tanh(theta_opt[4]/0.0049)#
        
    

    
    ###### date training
    def deco_train(self, batch_size, epoch_num, alpha):
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.alpha = alpha
        
        
        cost_ls = []
        theta_ls = []
        
        
        
        # add initials into parameter list
        theta_ls.append(self.theta)

       


        ROI_num = self.param['ROI_num']
        
        #### normalized parameters (> 0)
        self.param_norm()

        err_epoch=[]
        
        err_y=[]
        err_y.append(2)
        for i_ep in range(epoch_num):
            #print(i_ep)
            #print(theta_ls)
            #theta = np.array([100., 0.1, 0.1, 0.1])
            #x= 0.45*np.random.uniform(0.,1.0,(6, ROI_num))
            #x[1] = 0.15*x[1]


            y=[]
            e_bat=[]
            states=[]
            """m=0
            v=0
            t= 1"""
            #xg = np.zeros((30,ROI_num))  #### initals of states of sensitivity model

            for i_batch in range(len(self.ts)// batch_size):
                #print(i_batch)
                #xg = np.zeros((24,ROI_num))
                empBOLD = self.ts[i_batch*batch_size:(i_batch+1)*batch_size]
                result_x, result_g, simBOLD = self.simBOLD_batch()
                y.append(simBOLD)
                states.append(np.array(result_x))
                #print(simBOLD)
                #cost_r = cost(simBOLD, empBOLD)
                #cost_ls.append(cost_r)
                #print('theta', theta_ls)
                #print('cost' , cost_ls)

                #grad = gradient_est(xg, theta, theta_name, cost_r, l_s, empBOLD, param)
                grad, ey = self.gradient_compute_r(result_x, result_g, empBOLD, simBOLD)
                #print(grad)
                self.update_adam(grad)
                #print('new Theta', self.theta)
                self.t += 1
                theta_ls.append(self.theta)
                err_y.append(ey)
                e_bat.append(ey)
                self.param_norm()
            err_epoch.append(np.array(e_bat).mean())   
            #theta_sum += np.array([param["g"], param["g_EE"], param["g_IE"], param["g_EI"]])
            ts_sim = np.concatenate(y, axis = 0)
            err_bold=np.array(err_y)
            states_sim = np.concatenate(states, axis = 0)
            FC_sim = np.corrcoef(ts_sim.T)
            FC_emp = np.corrcoef(self.ts.T)
            cost_ls.append(np.corrcoef(FC_sim[np.tril_indices(ROI_num, -1)], FC_emp[np.tril_indices(ROI_num, -1)])[0,1])

        return np.array(cost_ls), np.array(theta_ls), states_sim, err_bold, np.array(err_epoch)###### test your model with the optimal model parameters

    def deco_test(self, n_tr_steps):
        results_sim = []
        Is_sim = []
        Es_sim = []
        # Random initial conditions
        ROI_num = self.param['ROI_num']
        dt = self.param['dt']
        #self.x= 0.45*np.random.uniform(0,1,(6, ROI_num))
        #self.x[1] = 0.15*x[1]
        """param["g"] = 20
        param["g_EE"] =0.1
        param["g_IE"] =0.1
        param["g_EI"] =0.1
        param['sigma'] = 0.005 #xp[0]"""
        """x[0] = np.random.uniform(low=0., high=1., size=(ROI_num))
        x[1] = np.random.uniform(low=0., high=.5, size=(ROI_num))"""


        for k in range(n_tr_steps*self.n_steps_keep_results):

            self.x = self.x +self.derivative_orig()*dt
            self.x[0] = self.x[0] +self.compute_noise()*np.sqrt(dt) #+.0333*epsilon_E[k]#
            #x[1] = x[1] #+compute_noise(param_trained)*np.sqrt(dt) #.0333*epsilon_I[k] #



            if ((k+1) %self.n_steps_keep_results == 0):
                results_sim.append(self.output_bold())#+0.0333*epsilon_y[k //n_steps_keep_results])#output_bold(x, param))
                Is_sim.append(self.x[1])
                Es_sim.append(self.x[0])

        results_sim = np.array(results_sim)
        Is_sim = np.array(Is_sim)
        Es_sim = np.array(Es_sim)
        return results_sim, Es_sim, Is_sim
