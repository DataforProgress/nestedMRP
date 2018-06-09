from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import us

fips2abbr = us.states.mapping('fips', 'abbr')
abbr2fips = us.states.mapping('abbr', 'fips')

class MRPpoll:
    def __init__(self,name,df,predictors,weightCol):
        self.name = name
        self.df   = df
        self.predictors = predictors
        self.weightCol  = weightCol
        self.mainEffectsLabels = [value.newName for key,value in predictors.items()]
            
        return

    def encodePredictors(self):
        for key,p in self.predictors.items():
            #Initialize
            self.df[p.newName] = 1
            for i,label in enumerate(p.catNames):
                self.df.ix[self.df[p.colName].isin(p.catMapping[label]),p.newName] = i+1
    
        return

    def encodeOutcomes(self,question,dichot,drop):
        self.df = self.df[~self.df[question].isnull()]
        try:
            self.df = self.df[self.df[question]!='__NA__']
        except:
            pass
        self.df[question] = self.df[question].apply(lambda x: int(x))
        if len(drop) > 0:
            self.df = self.df[~self.df[question].isin(drop)]

        #Dichotomize
        if dichot is not None:
            self.df["Success"] = 0
            self.df.ix[self.df[question].isin(dichot), 'Success'] = 1
            self.outcome = "Success"
            self.isDichotomous = True
        else:
            #self.df.ix[self.df[question]==7,question] = 4
            unq = np.unique(self.df[question])
            print(unq)
            self.outcome = list(unq)
            self.isDichotomous = False
        return

    def condense(self,question,adjust):
        mainEffectsLabels = self.mainEffectsLabels
        #Condense survey responses by category
        if self.isDichotomous:
            uniq_survey_df = (self.df.groupby(mainEffectsLabels).Success.agg(['sum','size']).reset_index())
            uniq_survey_df["n"]           = uniq_survey_df['size']
            uniq_survey_df["Success"]     = uniq_survey_df['sum']
        else:
            grouped = self.df.groupby(mainEffectsLabels)
            uniq_survey_df = grouped[question].value_counts().unstack(level=len(mainEffectsLabels)).reset_index().fillna(0.)
            uniq_survey_df[self.outcome] = uniq_survey_df[self.outcome].astype(np.int64)
            uniq_survey_df["n"]  = uniq_survey_df[self.outcome].sum(axis=1)
        
        
        if adjust and self.isDichotomous:
            #Compute adjusted counts to correct for the design effect
            #Thinking of removing this
            de_df          = (self.df.groupby(mainEffectsLabels).apply(lambda x: 1.+(x[self.weightCol].std(axis=0)/x[self.weightCol].mean(axis=0))**2)).reset_index().fillna(1.)
            wm_df          = (self.df.groupby(mainEffectsLabels).apply(lambda x: (x[self.weightCol]*x["Success"]).sum(axis=0)/x[self.weightCol].sum(axis=0))).reset_index().fillna(1.)
            de_df = de_df.rename(columns={0:"DesignEffect"})
            wm_df = wm_df.rename(columns={0:"WeightedMeanResponse"})
            uniq_survey_df["DesignEffect"]         = de_df["DesignEffect"]
            uniq_survey_df["WeightedMeanResponse"] = wm_df["WeightedMeanResponse"]
            uniq_survey_df["n"]           = (np.ceil(uniq_survey_df['size']/uniq_survey_df['DesignEffect'].mean())).astype(np.int64)
            uniq_survey_df["Success"]     = (np.around(uniq_survey_df['n']*uniq_survey_df['WeightedMeanResponse'])).astype(np.int64)

        self.uniq_survey_df = uniq_survey_df
        return

class pollPredictor:
    def __init__(self,colName,newName,catNames,catMapping):
        self.colName    = colName
        self.newName    = newName
        self.catNames   = catNames
        self.catMapping = catMapping
        return


def recodeAge(df,ageNumericCol,ageCodeDNR,ageCodeCol,ageDict):
    df["AgeRecode"] = 0
    ageCats = [(18.,29.),
               (30.,44.),
               (45.,54.),
               (55.,64.),
               (65.,150.)]
    for i,bounds in enumerate(ageCats):
        df.ix[(df[ageNumericCol] >= bounds[0])&(df[ageNumericCol] <= bounds[1]), 'AgeRecode'] = i
    
    if ageCodeDNR is not None:
        df.ix[df[ageNumericCol] == ageCodeDNR, 'AgeRecode'] = 5
    
    #If some ages are not given numerically
    if ageCodeCol:
        df.ix[(df[ageNumericCol]==ageCodeDNR)&(df[ageCodeCol]==ageDict["18-29"]),"AgeRecode"] = 0
        df.ix[(df[ageNumericCol]==ageCodeDNR)&(df[ageCodeCol]==ageDict["30-44"]),"AgeRecode"] = 1
        df.ix[(df[ageNumericCol]==ageCodeDNR)&(df[ageCodeCol]==ageDict["45-54"]),"AgeRecode"] = 2
        df.ix[(df[ageNumericCol]==ageCodeDNR)&(df[ageCodeCol]==ageDict["55-64"]),"AgeRecode"] = 3
        df.ix[(df[ageNumericCol]==ageCodeDNR)&(df[ageCodeCol]==ageDict["65+"])  ,"AgeRecode"] = 4
        df.ix[(df[ageNumericCol]==ageCodeDNR)&(df[ageCodeCol]==ageDict["DNR"])  ,"AgeRecode"] = 5

    ageDict    = {"18-29":[0],"30-44":[1],"45-54":[2],"55-64":[3],"65+":[4],"DNR":[5]}
    return df,"AgeRecode",ageDict

ageLabels    = ["18-29","30-44","45-54","55-64","65+","DNR"]
educLabels   = ["No HS","HS Grad","Some College","College Grad","DNR"]
marLabels    = ["Married","Widowed","Divorced","Separated","Never Married","DNR"]
incLabels    = ["Under 20k","20k-50k",'50k-75k','75k-100k','100k +',"DNR"]
raceLabels   = ["Other Race","White","Black","Hispanic","Asian","DNR"]
genderLabels = ["Male","Female"]
usrLabels    = ["Urban","Suburban","Rural","DNR"]

stateLabels = [s.abbr for s in us.states.STATES]
stateLabels = list(np.sort(stateLabels))
print(len(stateLabels))
print(stateLabels)
#States have an identity mapping
stateDict = {abbr:[abbr] for abbr in stateLabels}


def getMRPpoll(survey,question,dichot,drop,condense=True):
    validSurveys = ["VSG16"]
    if survey not in validSurveys:
        print(survey," is not a valid survey, try again")
        sys.exit()
    usrCol     = None
    incCol     = None
    marCol     = None
    ageCodeDNR = None
    ageCodeCol = None
    ageDict    = None
    if survey == "VSG16":
        surveyFile = "VOTER_Survey_December16_Release1.csv"
        df = pd.read_csv(surveyFile,dtype={"izip_baseline":object})
    
        weightCol = "weight"
    
        educCol    = "educ_2016"
        educDict   = {"No HS":[1],"HS Grad":[2],"Some College":[3,4],"College Grad":[5,6],'DNR':[None]}
    
        raceCol    = "race_2016"
        raceDict   = {"Other Race":[5,6,7,8],"White":[1],"Black":[2],'Hispanic':[3],'Asian':[4],'DNR':[None]}
    
        genderCol  = "gender_baseline"
        genderDict = {"Male":[1],"Female":[2]}
    
        ageCodeCol = None
        df["age"] = 2016 - df["birthyr_baseline"]
    
        zip2usr = pd.read_csv("zip2usr.csv",dtype={"ZIP_CODE":object})
        zip2usr = zip2usr[["ZIP_CODE","USR"]]
        zip2usr = zip2usr.rename(columns={"ZIP_CODE":"izip_baseline"})
        df = pd.merge(df,zip2usr,on="izip_baseline")
    
        usrCol     = "USR"
        usrDict    = {"Urban":["U"],"Suburban":["S"],"Rural":["R"],"DNR":[None]}
    
        marCol     = "marstat_2016"
        marDict    = {"Married":[1],"Widowed":[4],'Divorced':[3],'Separated':[2],'Never Married':[5,6],"DNR":[8]}
    
        incCol     = "faminc_2016"
        incDict    = {"Under 20k":[1,2],"20k-50k":[3,4,5],'50k-75k':[6,7,8],'75k-100k':[9],'100k +':[10,11,12,13,14,15,16,31],"DNR":[97]}
    
        stateCol       = "STATEABBR"
        df[stateCol]   = df["post_inputstate_2012"].apply(lambda x: fips2abbr[str(x).zfill(2)])
    

    df,ageCol,ageDict = recodeAge(df,"age",ageCodeDNR,ageCodeCol,ageDict)
    predictors = {}
    predictors["age"]       = pollPredictor(ageCol ,"AgeCat",ageLabels,ageDict)
    predictors["education"] = pollPredictor(educCol,"EduCat",educLabels,educDict)
    predictors["race"]      = pollPredictor(raceCol,"RaceCat",raceLabels,raceDict)
    predictors["gender"]    = pollPredictor(genderCol,"GenderCat",genderLabels,genderDict)
    if marCol is not None:
        predictors["marstat"]   = pollPredictor(marCol,"MarCat",marLabels,marDict)
    if usrCol is not None:
        predictors["usr"]       = pollPredictor(usrCol,"USRCat",usrLabels,usrDict)
    if incCol is not None:
        predictors["famincome"] = pollPredictor(incCol,"IncCat",incLabels,incDict)
    #predictors["state"]     = pollPredictor(stateCol,"state_initnum",stateLabels,stateDict)
    
    poll = MRPpoll(survey,df,predictors,weightCol)
    poll.encodePredictors()
    poll.encodeOutcomes(question,dichot,drop)
    if condense:
        poll.condense(question,True)


    return poll
