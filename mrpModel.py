from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd



def encodeInteraction(a1,a2):
    n1 = len(np.unique(a1))
    return a2*n1+a1

def hierarchical_normal(name, shape, mu=0.,cs=1.,sigma=None):
    delta = pm.Normal('delta_{}'.format(name), 0., 1., shape=shape)
    if sigma is None:
        sigma = pm.HalfNormal('sigma_{}'.format(name), sd=cs)
    
    return pm.Deterministic(name, mu + delta * sigma)

class linearEffect:
    def __init__(self,labels,df,scale=5.):
        self.labels = labels
        self.x     = df[labels]
        self.scale = scale

class categoricalEffect:
    def __init__(self,label,idx,ncat,order,priorCat=None,priorLin=None,scale=1.):
        self.label = label
        self.idx   = idx  
        self.ncat  = ncat 
        self.order = order
        self.priorCat = priorCat
        self.priorLin = priorLin
        self.scale = scale
        self.mean = 0.

    def getPrior(self,suffix):

        if self.priorLin is not None:
            betas  = tt.stack([pm.Normal(label+suffix,mu=0.,sd=self.priorLin.scale) for label in self.priorLin.labels],axis=0)
            x      = tt.as_tensor_variable(self.priorLin.x.as_matrix())
            self.mean    += x.dot(betas)

        if self.priorCat is not None:
            for effect in self.priorCat:
                coeff = hierarchical_normal(effect.label+suffix,effect.ncat,cs=effect.scale)
                self.mean += coeff[effect.idx]
                
            
        
def interactionFromDict(interactions,eDict):
    names = []
    for interaction in interactions:
        name = " x ".join([n for n in interaction])
        order = len(interaction)
        if order > 2 and "state_initnum" in interaction: continue
        if "state_initnum" in interaction: continue
        idx  = encodeInteraction(eDict[interaction[0]].idx,eDict[interaction[1]].idx)
        ncat = eDict[interaction[0]].ncat*eDict[interaction[1]].ncat
        if order > 2:
            for i in range(order-2):
                idx  = encodeInteraction(idx,eDict[interaction[i+2]].idx)
                ncat = ncat*eDict[interaction[i+2]].ncat
        eDict[name] = categoricalEffect(name,idx,ncat,order)
        names.append(name)
    return  eDict,names

def getIndexes(udf,mainEffectsLabels,intOrder):
    effects = {}

    if "GenderCat" in mainEffectsLabels:
        effects["GenderCat"] = categoricalEffect("GenderCat",udf.GenderCat.values,2,1)

    if "RaceCat" in mainEffectsLabels:
        effects["RaceCat"]   = categoricalEffect("RaceCat"  ,udf.RaceCat.values  ,len(np.unique(udf.RaceCat.values)),1)

    if "EduCat" in mainEffectsLabels:
        effects["EduCat"]    = categoricalEffect("EduCat"   ,udf.EduCat.values   ,len(np.unique(udf.EduCat.values)) ,1)

    if "AgeCat" in mainEffectsLabels:
        effects["AgeCat"]    = categoricalEffect("AgeCat"   ,udf.AgeCat.values   ,len(np.unique(udf.AgeCat.values)) ,1)

    if "MarCat" in mainEffectsLabels:
        effects["MarCat"]    = categoricalEffect("MarCat", udf.MarCat.values ,len(np.unique(udf.MarCat.values)) ,1)

    if "IncCat" in mainEffectsLabels:
        effects["IncCat"]    = categoricalEffect("IncCat", udf.IncCat.values ,len(np.unique(udf.IncCat.values)) ,1)

    if "USRCat" in mainEffectsLabels:
        effects["USRCat"]    = categoricalEffect("USRCat", udf.USRCat.values ,3,1)

    if "state_initnum" in mainEffectsLabels:
        effects["state_initnum"]     = categoricalEffect("state_initnum",udf.state_initnum.values,51,1)

    interactionsLoL = [[x for x in combinations(mainEffectsLabels,io+2)] for io in range(intOrder-1)]
    interactions = [item for sublist in interactionsLoL for item in sublist]
    effects,higherOrderEffectsLabels = interactionFromDict(interactions,effects)
    
    effectsLabels = mainEffectsLabels+higherOrderEffectsLabels
    
    indexes = np.zeros((len(udf),len(effectsLabels)),dtype=np.int64)
    for i,eff in enumerate(effectsLabels):
        indexes[:,i] = effects[eff].idx
    
    
    return indexes,effects,higherOrderEffectsLabels

def getLinearPropensity(indexes_,effects,mainEffectsLabels,higherOrderEffectsLabels,suffix=""):
    intOrder = np.max([effects[eff].order for eff in effects.keys()])
    #Baseline intercept
    a0   = pm.Normal('baseline'+suffix,mu=0.,sd=100.)
    
    #Individual predictors
    #Global scale
    sigma_glob = pm.HalfCauchy("globalScale"+suffix,1.)
    #Local scale for main effects
    for eff in mainEffectsLabels:
        effects[eff].localScale = pm.HalfNormal(eff+"_localScale"+suffix,sd=1.)
        if effects[eff].priorLin is not None or effects[eff].priorCat is not None:
            effects[eff].getPrior(suffix)

    #Local scale for higher order interactions
    if intOrder > 1:
        deltaScale = pm.HalfNormal("delta_higherOrder"+suffix,shape=intOrder-1 ,sd=1.)
        for eff in higherOrderEffectsLabels:
            mainEffs = eff.split(" x ")
            order = len(mainEffs)
            effects[eff].localScale = deltaScale[order-2]*np.prod([effects[meff].localScale for meff in mainEffs])
    
    eta = a0 
    for i,eff in enumerate(mainEffectsLabels+higherOrderEffectsLabels):
        print(i,eff)
        coeff = hierarchical_normal(eff+suffix,shape=effects[eff].ncat,sigma=effects[eff].localScale*sigma_glob,mu=effects[eff].mean)
        eta  += coeff[indexes_[:,i]]
    return eta


class mrpNestedModel:
    def __init__(self,poll,subPolls,intOrder):
        mainEffectsLabels = poll.mainEffectsLabels
        uniq_survey_df = poll.uniq_survey_df
        df = poll.df

        data = {}
        dataBlock = ["data {"]

        ps_df = pd.read_csv("ps_wUSR.csv")
        #!!!!!!this post stratification set uses 0 based indexing, convert to 1 based first!!!!
        ps_df = (ps_df.groupby(mainEffectsLabels).n.apply(sum)).reset_index()
        ps_df[mainEffectsLabels] = ps_df[mainEffectsLabels]+1
        indexes,effects,higherOrderEffectsLabels = getIndexes(ps_df,mainEffectsLabels,intOrder)
        print(ps_df)

        allEff = mainEffectsLabels+higherOrderEffectsLabels
        data["nEffects"] = len(allEff)
        dataBlock.append("  int<lower = 1> nEffects;")

        data["nCellPopulation"] = len(ps_df)
        dataBlock.append("  int<lower = 1> nCellPopulation;")
        data["indexes_Pop"] = indexes
        dataBlock.append("  int indexes_Pop[nCellPopulation,nEffects];")
        data["N_Pop"] = ps_df.n.values
        dataBlock.append("  int N_Pop[nCellPopulation];")

        N       = uniq_survey_df['n'].values
        outcome = uniq_survey_df[poll.outcome].values

        data["nCellSample_z"] = len(uniq_survey_df)
        dataBlock.append("  int<lower = 1> nCellSample_z;")
        
        data["nResponse_z"] = len(subPolls)
        dataBlock.append("  int<lower = 2> nResponse_z;")

        data["response_z"] = outcome
        dataBlock.append("  int response_z[nCellSample_z,nResponse_z];")

        print("Outcome: ",np.sum(outcome,axis=0))
        print("N      : ",np.sum(N))
        
        indexes,effects,higherOrderEffectsLabels = getIndexes(uniq_survey_df,mainEffectsLabels,intOrder)
 

        for eff in allEff:
            print("n"+effects[eff].label)
            data["n"+effects[eff].label] = effects[eff].ncat
            dataBlock.append("  int<lower = 1> "+"n"+effects[eff].label+";")

        data["indexes_z"] = indexes
        dataBlock.append("  int indexes_z[nCellSample_z,nEffects];")
        


        self.outcome  = outcome
        self.poll     = poll
        self.effects  = effects
        self.intOrder = intOrder
        self.mainEffectsLabels = mainEffectsLabels
        self.higherOrderEffectsLabels = higherOrderEffectsLabels

        for i,p in enumerate(subPolls):
            idx = str(i+1)
            N       = p['n'].values
            outcome = p["Success"].values

            data["nCellSample_"+idx] = len(p)
            dataBlock.append("  int<lower = 1> nCellSample_"+idx+";")
            
            data["N_"+idx] = N
            dataBlock.append("  int N_"+idx+"[nCellSample_"+idx+"];")

            data["response_"+idx] = outcome
            dataBlock.append("  int response_"+idx+"[nCellSample_"+idx+"];")

            print("Outcome: ",np.sum(outcome,axis=0))
            print("N      : ",np.sum(N))
            
            indexes,effects,higherOrderEffectsLabels = getIndexes(p,mainEffectsLabels,intOrder)

            data["indexes_"+idx] = indexes
            dataBlock.append("  int indexes_"+idx+"[nCellSample_"+idx+",nEffects];")
            
        dataBlock.append("}")
        

        paramBlock = ["parameters{"]
        tParamBlock = ["transformed parameters{"]
        #Declare variables
        #Coefficients for the multinomial model
        for i in self.poll.outcome[1:]:
            name = "intercept_z_"+str(i)
            paramBlock.append("  real "+name+";")
            for eff in allEff:
                name = effects[eff].label+"_z_"+str(i)
                n = "n"+effects[eff].label
                paramBlock.append("  vector"+"["+n+"]"+"delta_"+name+";")
                paramBlock.append("  real <lower=0> "+"stdv_"+name+";")
                tParamBlock.append("  vector"+"["+n+"]"+"a_"+name+";")
        #Coefficients for the binomial models
        for i in self.poll.outcome:
            name = "intercept_"+str(i)
            paramBlock.append("  real "+name+";")
            name = "eta_"+str(i)
            tParamBlock.append("  vector[nCellSample_"+str(i)+"]"+name+";")
            for eff in allEff:
                name = effects[eff].label+"_"+str(i)
                n = "n"+effects[eff].label
                paramBlock.append("  vector"+"["+n+"]"+"delta_"+name+";")
                paramBlock.append("  real <lower=0> "+"stdv_"+name+";")
                tParamBlock.append("  vector"+"["+n+"]"+"a_"+name+";")

        tParamBlock.append("  matrix[nCellSample_z,nResponse_z] eta_z;")
        tParamBlock.append("  vector[nCellSample_z] zeros;")
        tParamBlock.append("  zeros = rep_vector(0,nCellSample_z);")

        #Specify non centered parametrizations
        #Coefficients for the multinomial model
        tParamBlock.append("  eta_z[:,1] = zeros;")
        for i in self.poll.outcome[1:]:
            etaStr = "  eta_z[:,"+str(i)+"] = "
            for j,eff in enumerate(allEff):
                name = effects[eff].label+"_z_"+str(i)
                tParamBlock.append("  a_"+name+"= stdv_"+name+" * delta_"+name+";")
                etaStr += "a_"+name+"[indexes_z[:,"+str(j+1)+"]]"
                if j + 1 < len(allEff):
                    etaStr += " + "
            tParamBlock.append(etaStr+";")
        #Coefficients for the binomial models
        for i in self.poll.outcome:
            etaStr = "  eta_"+str(i)+" = "
            for j,eff in enumerate(allEff):
                name = effects[eff].label+"_"+str(i)
                tParamBlock.append("  a_"+name+"= stdv_"+name+" * delta_"+name+";")
                etaStr += "a_"+name+"[indexes_"+str(i)+"[:,"+str(j+1)+"]]"
                if j + 1 < len(allEff):
                    etaStr += " + "
            tParamBlock.append(etaStr+";")

        paramBlock.append("}")
        tParamBlock.append("}")

        modelBlock = ["model {"]
        #specify the model
        #Coefficients for the multinomial model
        for i in self.poll.outcome[1:]:
            name = "intercept_z_"+str(i)
            modelBlock.append("  "+name+" ~ normal(0,100);")
            for eff in allEff:
                name = effects[eff].label+"_z_"+str(i)
                modelBlock.append("  delta_"+name+" ~ normal(0,1);")
                modelBlock.append("  stdv_"+name+" ~ normal(0,1);")
        #Coefficients for the binomial models
        for i in self.poll.outcome:
            name = "intercept_"+str(i)
            modelBlock.append("  "+name+" ~ normal(0,100);")
            for eff in allEff:
                name = effects[eff].label+"_"+str(i)
                modelBlock.append("  delta_"+name+" ~ normal(0,1);")
                modelBlock.append("  stdv_"+name+" ~ normal(0,1);")

        modelBlock.append("  for(n in 1:nCellSample_z)")
        modelBlock.append("    response_z[n,:]   ~ multinomial(softmax(to_vector(eta_z[n,:])));")
        for i in self.poll.outcome:
            istr = str(i)
            modelBlock.append("  for(n in 1:nCellSample_"+istr+")")
            modelBlock.append("    response_"+istr+"[n]   ~ binomial(N_"+istr+"[n],inv_logit(eta_"+istr+"[n]));")

        modelBlock.append("}")

        gqBlock = ["generated quantities {"]
        gqBlock.append("  simplex[nResponse_z] probs;")
        gqBlock.append("  vector[nResponse_z] etaTemp_z;")
        gqBlock.append("  vector[nResponse_z] etaTemp;")
        gqBlock.append("  int countsTemp[nResponse_z];")
        gqBlock.append("  int totalYes;")
        gqBlock.append("  int totalN;")
        gqBlock.append("  real totalPct;")
        gqBlock.append("  totalYes=0;")
        gqBlock.append("  totalN=0;")
        gqBlock.append("  etaTemp_z[1]=0;")
        gqBlock.append("  for(i in 1:nCellPopulation){")
        for i in self.poll.outcome[1:]:
            etaStr = "    etaTemp_z["+str(i)+"] = "
            for j,eff in enumerate(allEff):
                name = effects[eff].label+"_z_"+str(i)
                etaStr += "a_"+name+"[indexes_Pop[i,"+str(j+1)+"]]"
                if j + 1 < len(allEff):
                    etaStr += " + "
            gqBlock.append(etaStr+";")
        gqBlock.append("    probs = softmax(etaTemp_z);")
        gqBlock.append("    countsTemp = multinomial_rng(probs,N_Pop[i]);")
        gqBlock.append("    totalN += N_Pop[i];")
        for i in self.poll.outcome:
            etaStr = "    etaTemp["+str(i)+"] = "
            for j,eff in enumerate(allEff):
                name = effects[eff].label+"_"+str(i)
                etaStr += "a_"+name+"[indexes_Pop[i,"+str(j+1)+"]]"
                if j + 1 < len(allEff):
                    etaStr += " + "
            gqBlock.append(etaStr+";")
        gqBlock.append("    for(j in 1:nResponse_z){")
        gqBlock.append("      totalYes += binomial_rng(countsTemp[j],inv_logit(etaTemp[j]));")
        gqBlock.append("    }")
        gqBlock.append("  }")
        gqBlock.append("  totalPct = 100.*totalYes/totalN;")

        gqBlock.append("}")
        with open("MRP.stan","w") as f:
            for line in dataBlock:
                f.write(line+"\n")
            for line in paramBlock:
                f.write(line+"\n")
            for line in tParamBlock:
                f.write(line+"\n")
            for line in modelBlock:
                f.write(line+"\n")
            for line in gqBlock:
                f.write(line+"\n")
        self.data = data
            
        return
        
    def setPrior(self,effectName,df,linTerms=None,catTerms=None):
        if linTerms is not None:
            self.effects[effectName].priorLin = linearEffect(linTerms,df)
            for eff in self.subEffects:
                eff[effectName].priorLin = linearEffect(linTerms,df)
        if catTerms is not None:
            self.effects[effectName].priorCat = [categoricalEffect(c,df[c],len(np.unique(df[c])),1) for c in catTerms]
            for eff in self.subEffects:
                eff[effectName].priorCat = [categoricalEffect(c,df[c],len(np.unique(df[c])),1) for c in catTerms]
        return
        
