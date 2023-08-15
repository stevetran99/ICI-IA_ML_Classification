

import datetime as dt
import os

### TODAY'S DATE
Date = dt.date.today().strftime('%y%m%d')

### New directory
mainpath = 'R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/FullBootstrapTest{}/'.format(Date)

if not os.path.exists(mainpath):
    os.makedirs(mainpath)

with open('CA_Phe_AllCode_FullBootstrap.py','r') as thisFile:
    logData = thisFile.read()
    with open(mainpath+'ScriptSave_{}.py'.format(Date),'w') as logFile:
        logFile.write(logData)
thisFile.close()
logFile.close()


#====================================================
import pandas as pd
import numpy as np
import datetime as dt
import glob

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample

from sklearn import metrics
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

import pickle
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import RandomOverSampler


#====================================================


#%%====================================================

def diag_icd9to10(df):
    mp = pd.read_csv('R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/ICD_9to10_Diag_Mapping.csv',dtype=str)
    mp = mp.loc[(mp['ICD10']!='NoDx')]
    
    ddf10 = df.loc[(df['type']=='Diag') &
                    (df['vocab']=='ICD-10-CM')].copy()
    print('ICD10:',ddf10.shape)
    ddf9 = df.loc[(df['type']=='Diag') &
                   (df['vocab']=='ICD-9-CM')].copy()
    print('ICD9:', ddf9.shape)
    nondf = df.loc[~(df['type']=='Diag')].copy()
    
    ddf10['code'] = ddf10['code'].str.replace('.','')
    ddf9['code'] = ddf9['code'].str.replace('.','')
    
    ddf9j = ddf9.merge(mp,how='inner',
                       left_on='code',right_on='ICD9')
    ddf9j['code'] = ddf9j['ICD10']
    ddf9j.drop(['ICD9','ICD10'],axis=1,inplace=True)
    ddf9j['vocab'] = 'ICD-10-CM'
    print('ICD9 mapped:',ddf9j.shape)
    
    return pd.concat([ddf10,ddf9j,nondf]).reset_index(drop=True)


def keepCPTonly(df):
    dfin = df.copy()
    
    dfin.loc[(dfin['type']=='Proc') &
             (dfin['vocab']=='cpt'),'vocab'] = 'CPT'
    
    dfout = pd.concat([dfin.loc[(dfin['type']=='Proc') &
                                (dfin['vocab']=='CPT')].copy(),
                       dfin.loc[~(dfin['type']=='Proc')].copy()]).reset_index(drop=True)
    
    return dfout







#%%====================================================

### OPTION FUNCTION
# Function to create the Predictor variable
# befaft=True specifies that we want to distinguish between before and after ICI initiation
def varPredictor(df,befaft=1):
    dfin = df.copy()
    
    # Convert ICI start and date to datetime
    dfin['ICI_start_datetime'] = pd.to_datetime(dfin.ICI_start_datetime)
    dfin['date'] = pd.to_datetime(dfin.date)
    
    # Create after_ICI column and set to 'Bef'
    dfin['afterICI'] = 'Bef'
    # Set after_ICI to 'Aft' for all rows with date after ICI_start_datetime
    dfin.loc[(dfin.date>dfin.ICI_start_datetime),'afterICI'] = 'Aft'
    
    dfin['Predictor'] = dfin['type'] + '_' + dfin['code']
    
    # Before/After
    if befaft==1:
        dfin['Predictor'] = dfin['Predictor'] + '_' + dfin['afterICI']
    else:
        dfin['Predictor'] = dfin['Predictor'] + '_noTemporal'
    
    return dfin


def agg_presence(df):
    df_out = df.groupby(by=['MRN','Predictor']).size().reset_index(name='counts')
    
    # Presence = 1 where count >= 1
    df_out['values'] = 0
    df_out.loc[(df_out.counts >= 1),'values'] = 1
    
    df_out.drop(['counts'],axis=1,inplace=True)
    
    df_out['Predictor'] = df_out.Predictor + '_exists'
    
    return df_out


def fil_topFeat(df,featuredf,top,model):
    df_in = df.copy()
    
    if model=='RF':
        featdf = featuredf.sort_values(by=['Importance'],ascending=False).head(n=top).copy()
    elif model=='LReg':
        featdf = featuredf.copy()
        featdf['magnitude'] = featdf['coef'].abs()
        featdf = featdf.sort_values(by=['magnitude'],ascending=False).head(top).copy()
    
    featdf['Predictor'] = featdf['type']+'_'+featdf['code']+'_'+featdf['modifier']
    
    df_out = df_in.merge(featdf[['Predictor']],how='inner',
                         on=['Predictor'])
    
    return df_out


# pivot function
def pivot_df(df):
    # pivot with MRN/cohort being rows and predictors as columns
    dfP = df.pivot(index=['MRN'],
                   columns=['Predictor'],
                   values=['values'])
    # change column names to the AttFeatICI names
    dfP.columns = dfP.columns.get_level_values(1)
    # reset index so MRN and cohort are columns not indices
    dfP = dfP.reset_index()
    # remove the name above the index
    dfP.columns.name = None
    
    ### Left join to ICICohort 
    ICICohort = pd.read_csv('R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_ChartReview/CA_fullCohortMRNs_230323.csv',dtype=str)
    ICICohort = ICICohort.rename(columns={'mrn':'MRN'})
    # Left join full ICI cohort to include subjects without any codes
    dfPF = ICICohort[['MRN']].merge(dfP,how='inner',
                           on=['MRN'])

    # replace all NaN with 0, since these are counts
    dfPF = dfPF.fillna(0)
    
    return dfPF



#%%====================================================

def splitTestTrain():    
    ICICohort = pd.read_csv('R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_ChartReview/CA_fullCohortMRNs_230323.csv',dtype=str)
    ICICohort = ICICohort.rename(columns={'mrn':'MRN'})

    ICICohort['cohort'] = [1 if c=='Case' else 0 for c in ICICohort.cohort]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    for train_index, test_index in split.split(ICICohort, ICICohort['cohort']):
        train_set = ICICohort.loc[train_index].reset_index(drop=True)
        test_set = ICICohort.loc[test_index].reset_index(drop=True)
        
    return train_set, test_set


def createTestTrain(df,testID,trainID,func='Classifier'):
    # inner join the data to test IDs
    test_set = testID.merge(df,how='inner',on=['MRN'])
    # inner join the data to train IDs
    train_set = trainID.merge(df,how='inner',on=['MRN'])
    
    if func == 'Classifier':
        ### Training: split X,y
        y_train = train_set.cohort
        X_train = train_set.drop(['MRN','cohort'],axis=1)

        ### Testing: split X,y
        y_test = test_set.cohort
        X_test = test_set.drop(['MRN','cohort'],axis=1)

        return X_train,y_train,X_test,y_test,test_set.MRN
    
    elif func == 'Anomaly':
        # drop the CA cases from the train set
        train_anom = train_set.loc[train_set.cohort==0].copy()
        # Training: split X,y
        X_trainA = train_anom.drop(['MRN','cohort'],axis=1)
        y_trainA = train_anom.cohort
        # Testing: split X,y
        X_testA = test_set.drop(['MRN','cohort'],axis=1)
        y_testA = test_set.cohort
        
        return X_trainA,y_trainA,X_testA,y_testA,test_set.MRN


#%%====================================================

def plot_ROC(model,y,X,func='df',anom=False):
    if func=='df':
        y_scores = model.decision_function(X)
    elif func=='pp':
        y_scores = model.predict_proba(X)[:,1]
        
    if anom:
        y_scores = y_scores*(-1)
    
    fpr, tpr, thresholds = metrics.roc_curve(y, y_scores)
    
    auc = metrics.roc_auc_score(y, y_scores)
    print('ROC:',auc)
    
    # Compute Youden's J, return optimal threshold
    J_thresh = thresholds[np.argmax(tpr-fpr)]
    
    plt.plot(fpr,tpr)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.text(0.95,0.01,'AUC: '+str(auc),
            verticalalignment='bottom',horizontalalignment='right')
    
    return auc,fpr,tpr,thresholds,J_thresh


def plot_precision_recall(model,y,X,func='df',anom=False):
    if func=='df':
        y_scores = model.decision_function(X)
    elif func=='pp':
        y_scores = model.predict_proba(X)[:,1]
    
    if anom:
        y_scores = y_scores*(-1)
        
    precisions, recalls, thresholds = metrics.precision_recall_curve(y, y_scores)
    
    auc = metrics.auc(recalls, precisions)
    print('PRC:',auc)
    
    F_thresh = thresholds[np.argmax((2*precisions*recalls)/(precisions+recalls))]
    
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.text(0.05,0.01,'AUC: '+str(auc),
            verticalalignment='bottom',horizontalalignment='left')
    
    return auc, precisions, recalls, thresholds, F_thresh


def plot_precision_recall_vs_threshold(log, y, X):
    y_scores = log.decision_function(X)
    
    precisions, recalls, thresholds = metrics.precision_recall_curve(y, y_scores)
    
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='center left')
    plt.ylim([0,1])
    
    plt.show()
    
    
def plot_sens_spec_vs_threshold(log, y, X):
    y_scores = log.decision_function(X)
    
    fpr, tpr, thresholds = metrics.roc_curve(y, y_scores)
    
    plt.plot(thresholds, tpr, 'b--', label='Sensitivity')
    plt.plot(thresholds, 1-fpr, 'g--', label='Specificity')
    plt.xlabel('Threshold')
    plt.legend(loc='center left')
    plt.ylim([0,1])
    
    plt.show()
    

def calculateCI(bootstrap_scores):
    #calculate 95% CI
    mean = np.mean(bootstrap_scores)
    sd = np.std(bootstrap_scores)
    
    CI_low = mean - (2*sd)
    CI_upp = mean + (2*sd)
    
    return CI_low, CI_upp


def bootstrap_TestSet(model,y,X,thresh,func='df',n_bootstrap=200,anom=False):
    if func=='df':
        y_scores = model.decision_function(X)
    elif func=='pp':
        y_scores = model.predict_proba(X)[:,1]
    
    if anom:
        y_scores = y_scores*(-1)
        
    y_bool = y_scores >= thresh
    y_class = y_bool.astype(int)
        
    seed=42
    bootstrap_auc=[]
    bootstrap_prc=[]
    bootstrap_sens=[]
    bootstrap_spec=[]
    bootstrap_ppv=[]
    bootstrap_npv=[]
    
    rng=np.random.RandomState(seed)
    
    for i in range(n_bootstrap):
        indices = rng.randint(0,len(y_scores),len(y_scores))
        if len(np.unique(y[indices])) < 2:
            continue
        
        bootstrap_auc.append(metrics.roc_auc_score(y[indices],y_scores[indices]))
        
        precisions, recalls, thresholds = metrics.precision_recall_curve(y[indices],y_scores[indices])
        bootstrap_prc.append(metrics.auc(recalls,precisions))
        
        tn, fp, fn, tp = metrics.confusion_matrix(y[indices],y_class[indices]).ravel()
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        ppv = tp/(tp+fp)
        npv = tn/(tn+fn)
        
        bootstrap_sens.append(sens)
        bootstrap_spec.append(spec)
        bootstrap_ppv.append(ppv)
        bootstrap_npv.append(npv)

    aucL,aucU = calculateCI(bootstrap_auc)
    prcL,prcU = calculateCI(bootstrap_prc)
    sensL,sensU = calculateCI(bootstrap_sens)
    specL,specU = calculateCI(bootstrap_spec)
    ppvL,ppvU = calculateCI(bootstrap_ppv)
    npvL,npvU = calculateCI(bootstrap_npv)
    
    return aucL,aucU,prcL,prcU,sensL,sensU,specL,specU,ppvL,ppvU,npvL,npvU


def basic_metrics(model,y,X,thresh,func='df',anom=False):
    if func=='df':
        y_scores = model.decision_function(X)
    elif func=='pp':
        y_scores = model.predict_proba(X)[:,1]
        
    # convert to classes based on J_thresh
    if anom:
        y_scores = y_scores*(-1)
    
    y_bool = y_scores >= thresh
    y_class = y_bool.astype(int)
        
    tn, fp, fn, tp = metrics.confusion_matrix(y,y_class).ravel()
    
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    
    print('Sens:',sens,'Spec:',spec,'PPV:',ppv,'NPV:',npv)
    
    return sens, spec, ppv, npv


def plot_AUCs(df,tot_n=32198,mode='Percent',metric='ROCAUC'):
    dfin = df.loc[(df['metric']==metric)].copy()
    ['BefAft','Agg','TopFeat','model','metric','threshType','value','lowCI','uppCI']
    
    dfin = dfin.drop(['BefAft','Agg','metric','threshType','lowCI','uppCI'],axis=1)
    
    dfin['TopPerc'] = dfin['TopFeat']*100/tot_n
    
    f,ax = plt.subplots(figsize=(8,8))
    
    if mode=='Percent':
        X = 'TopPerc'
    elif mode=='Number':
        X = 'TopFeat'
    
    sns.lineplot(ax=ax,data=dfin,x=X,y='value',hue='model')
    ax.set(xscale="log")
    ax.set(xlabel='{} of top features'.format(mode),ylabel='AUC')
    ax.set(ylim=(0.45,1))
    ax.set(ylabel=metric)

    if mode=='Percent':
        ax.set_xticks([.1,1,10,50])
        ax.set_xticklabels([.1,1,10,50])
    elif mode=='Number':
        ax.set_xticks([30,300,3000,15000])
        ax.set_xticklabels([30,300,3000,15000])
    
    plt.title('{} vs {} of top features trained on'.format(metric,mode))
    
    
def addNames(feat):
    dfin = feat.copy()
    
    CPT = pd.read_csv('CodeNames/CPT.csv',dtype=str)
    CPT['type'] = 'Proc'
    CPT['code'] = CPT['cpt_code']
    CPT['name'] = CPT['cpt_name']
    
    ICD10 = pd.read_csv('CodeNames/ICD10.csv',dtype=str)
    ICD10['type'] = 'Diag'
    ICD10['code'] = ICD10['diagnosis_code'].str.replace('.','')
    ICD10['name'] = ICD10['diagnosis_name']
    
    RxNorm = pd.read_csv('CodeNames/RxNorm.csv',dtype=str)
    RxNorm['type'] = 'Meds'
    RxNorm['code'] = RxNorm['rxnorm_code']
    RxNorm['name'] = RxNorm['rxnorm_name']
    
    LOINC = pd.read_csv('CodeNames/Loinc.csv',dtype=str)
    LOINC['type'] = 'Labs'
    LOINC['code'] = LOINC['LOINC_NUM']
    LOINC['name'] = LOINC['LONG_COMMON_NAME']
    
    names = pd.concat([CPT[['type','code','name']],
                       ICD10[['type','code','name']],
                       RxNorm[['type','code','name']],
                       LOINC[['type','code','name']]])
    
    dfmerge = dfin.merge(names, how='left',
                         left_on=['type','code'],
                         right_on=['type','code'])
    
    dfmerge.loc[(dfmerge['type']=='Diag'),'vocab'] = 'ICD10'
    dfmerge.loc[(dfmerge['type']=='Labs'),'vocab'] = 'LOINC'
    dfmerge.loc[(dfmerge['type']=='Meds'),'vocab'] = 'RxNorm'
    dfmerge.loc[(dfmerge['type']=='Proc'),'vocab'] = 'CPT'
    
    dfmerge['LongName'] = dfmerge['name']+' '+'('+dfmerge['vocab']+': '+dfmerge['code']+')'+' '+dfmerge['modifier'].str.split('_').str[0] 
    
    return dfmerge


def processFeatImp(featdf,model):
    dfin = featdf.copy()
    
    if model=='rf':
        dfin[['type','code','modifier']] = dfin['Feature'].str.split('_',n=2,expand=True)
        dfin.drop(['Feature'],axis=1,inplace=True)
    elif model=='lreg':
        dfin['magnitude'] = pd.to_numeric(dfin['coef']).abs()
    
    dfout = addNames(dfin)
    
    return dfout






#%%====================================================

def ML_saveScores(dfagg,MRNs,model,func):
    dfagg_cp = dfagg.copy()
    
    if func == 'df':
        dfagg_cp.insert(1,'Scores',model.decision_function(dfagg_cp))
    elif func == 'pp':
        dfagg_cp.insert(1,'Scores',model.predict_proba(dfagg_cp)[:,1])
    dfagg_cp.insert(1,'MRN',MRNs)
        
    return dfagg_cp.sort_values('Scores',ascending=False)


def ML_performances(model,y_test,X_test,scoring,Date,folder,label,aucDF,metricsDF,ba,agg,topF,ver,MRNs,anom=False):
    ### calculate ROC
    rocauc,fpr,tpr,rocthresholds,J_thresh = plot_ROC(model,y_test,X_test,func=scoring,anom=anom)
    aucDF = aucDF.append(pd.DataFrame({'FPR_Prec':fpr,'TPR_Recl':tpr,'type':'ROC','Model':folder,
                                       'BefAft':ba,'Agg':agg.__name__,'TopFeat':topF,'Ver':ver}),ignore_index=True)
    # save ROC fig
    plt.title(label)
    plt.savefig('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/{}/{}_ROC.png'.format(Date,folder,label))
    plt.close()
    ### calculate Precision vs recall
    prcauc,precisions,recalls,prcthresholds,F_thresh = plot_precision_recall(model,y_test,X_test,func=scoring,anom=anom)
    aucDF = aucDF.append(pd.DataFrame({'FPR_Prec':precisions,'TPR_Recl':recalls,'type':'PRC','Model':folder,
                                       'BefAft':ba,'Agg':agg.__name__,'TopFeat':topF,'Ver':ver}),ignore_index=True)
    # save PRC fig
    plt.title(label)
    plt.savefig('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/{}/{}_PrecRecCurve.png'.format(Date,folder,label))
    plt.close()
    
    ### calculate metrics
    # Youden's J
    sens,spec,ppv,npv = basic_metrics(model,y_test,X_test,func=scoring,thresh=J_thresh,anom=anom)
    aucL,aucU,prcL,prcU,sensL,sensU,specL,specU,ppvL,ppvU,npvL,npvU = bootstrap_TestSet(model,y_test,X_test,thresh=J_thresh,func=scoring,anom=anom)
    # save ROC AUC
    metricsDF = metricsDF.append(pd.DataFrame({'BefAft':ba,'Agg':agg.__name__,'TopFeat':topF,'Ver':ver,'model':folder,'metric':'ROCAUC',
                                               'threshType':'all','value':[rocauc],'lowCI':[aucL],'uppCI':[aucU]}),ignore_index=True)
    # save PRC AUC
    metricsDF = metricsDF.append(pd.DataFrame({'BefAft':ba,'Agg':agg.__name__,'TopFeat':topF,'Ver':ver,'model':folder,'metric':'PRCAUC',
                                               'threshType':'all','value':[prcauc],'lowCI':[prcL],'uppCI':[prcU]}),ignore_index=True)
    # save sens,spec,ppv,npv
    metricsDF = metricsDF.append(pd.DataFrame({'BefAft':ba,'Agg':agg.__name__,'TopFeat':topF,'Ver':ver,'model':folder,'metric':['Sensitivity','Specificity','PPV','NPV'],
                                               'threshType':'J','value':[sens,spec,ppv,npv],'lowCI':[sensL,specL,ppvL,npvL],
                                               'uppCI':[sensU,specU,ppvU,npvU]}),ignore_index=True)
    # F stat
    sens,spec,ppv,npv = basic_metrics(model,y_test,X_test,func=scoring,thresh=F_thresh,anom=anom)
    aucL,aucU,prcL,prcU,sensL,sensU,specL,specU,ppvL,ppvU,npvL,npvU = bootstrap_TestSet(model,y_test,X_test,thresh=F_thresh,func=scoring,anom=anom)
    # save sens,spec,ppv,npv
    metricsDF = metricsDF.append(pd.DataFrame({'BefAft':ba,'Agg':agg.__name__,'TopFeat':topF,'Ver':ver,'model':folder,'metric':['Sensitivity','Specificity','PPV','NPV'],
                                               'threshType':'F','value':[sens,spec,ppv,npv],'lowCI':[sensL,specL,ppvL,npvL],
                                               'uppCI':[sensU,specU,ppvU,npvU]}),ignore_index=True)

    
    # save scores on full cohort
    if anom:
        ML_saveScores(X_test,MRNs,model,scoring).sort_values('Scores',ascending=True).to_csv('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/{}/{}_Pred.csv'.format(Date,folder,label),index=False)
    else:
        ML_saveScores(X_test,MRNs,model,scoring).to_csv('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/{}/{}_Pred.csv'.format(Date,folder,label),index=False)
        
    ### Output metrics dataframes
    aucDF.to_csv('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/AUCtest_{}.csv'.format(Date,Date),index=False)
    metricsDF.to_csv('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/metrics_{}.csv'.format(Date,Date),index=False)
    
    return metricsDF,aucDF




#%%====================================================

def FullGridSearch(df,iterations=25,featsel='original',versions=1):
    dfin = df.copy()
    
    ### Options:
    Date = dt.date.today().strftime('%y%m%d')
    pred_befaft = [1]
    #aggregation = [agg_count,agg_logCount,agg_countDays,agg_logCountDays,agg_presence]
    aggregation = [agg_presence]
    filterFeat = {0:14,1:15}   #15 for bef aft, 14 for not
    
    ### Make directories
    mainpath = 'C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/'.format(Date)
    MLs = ['LReg','RF','GB','NN','KNN','SVM','SVMAnom']
    for d in MLs:
        if not os.path.exists(mainpath+d+'/'):
            os.makedirs(mainpath+d+'/')

    
    ### AUC dataframe
    aucDF = pd.DataFrame(columns=['FPR_Prec','TPR_Recl','type','Model','BefAft','Agg','TopFeat','Ver'])
    metricsDF = pd.DataFrame(columns=['BefAft','Agg','TopFeat','Ver','model','metric','threshType','value','lowCI','uppCI'])
    
    ros = RandomOverSampler()
    
        
    for ver in range(versions):
        ### Split train/test sets
        train_set, test_set = splitTestTrain()
        
        # create predictor column
        for ba in pred_befaft:
            dfpred = varPredictor(dfin,befaft=ba)
    
            # aggregation
            for agg in aggregation:
                dfagg = agg(dfpred)
                print(dfagg['Predictor'].nunique())
                
                # filter for top feature importance
                for fil in range(filterFeat[ba]):
                    topF = int(dfagg['Predictor'].nunique()/np.power(2,fil))  # num features / 2^fil
                    if fil==0:    # first round with all features is not filtered
                        LRdffil = pivot_df(dfagg)
                        #RFddfil = LRdffil.copy()
                    elif featsel != 'None':
                        # Read in logistic regression coefficients from previous round, filter top features
                        if featsel=='original':    # uses the feature list from the total feature run every time
                            LRfeatFile = 'C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/LReg/LReg_{}_ba{}_{}_Top{}_v{}_Coef.csv'.format(Date,Date,str(ba),agg.__name__,dfagg['Predictor'].nunique(),ver)
                            #RFfeatFile = 'C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/RF/RF_{}_ba{}_{}_Top{}_v{}_Feat.csv'.format(Date,Date,str(ba),agg.__name__,dfagg['Predictor'].nunique(),ver)
                        elif featsel=='sequential':    # uses the feature list from the previous feature fraction
                            LRfeatFile = 'C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/LReg/LReg_{}_ba{}_{}_Top{}_v{}_Coef.csv'.format(Date,Date,str(ba),agg.__name__,int(dfagg['Predictor'].nunique()/np.power(2,fil-1)),ver)
                            #RFfeatFile = 'C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/RF/RF_{}_ba{}_{}_Top{}_v{}_Feat.csv'.format(Date,Date,str(ba),agg.__name__,int(dfagg['Predictor'].nunique()/np.power(2,fil-1)),ver)
                        LRfeatImpDF = pd.read_csv(LRfeatFile)
                        LRdffil = pivot_df(fil_topFeat(df=dfagg,featuredf=LRfeatImpDF,top=topF,model='LReg'))
                        
                        #RFfeatImpDF = pd.read_csv(RFfeatFile)
                        #RFdffil = pivot_df(fil_topFeat(df=dfagg,featuredf=RFfeatImpDF,top=topF,model='RF'))
                        
                    else:
                        continue
                    
                    
                    
    
    
    
                    ### LOGISTIC REGRESSION
                    
                    print('Filter LR DF:',LRdffil.shape)
                    
                    ### create test/train sets 
                    # classifier
                    X_train,y_train,X_test,y_test,MRNs = createTestTrain(df=LRdffil,
                                                                    testID=test_set,trainID=train_set,
                                                                    func='Classifier')
                    
                    savedf = pd.DataFrame(columns=X_test.columns)
                    savedf.to_csv('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/LReg/AlignDF_LReg_{}_ba{}_{}_Top{}_v{}.csv'.format(Date,Date,str(ba),agg.__name__,topF,ver),index=False)
    
                    
                    # model creation
                    log = LogisticRegression(penalty='l2',max_iter=100000).fit(X_train,y_train)
                    model = log
                    scoring = 'df'
                    folder = 'LReg'
    
                    label = '{}_{}_ba{}_{}_Top{}_v{}'.format(folder,Date,str(ba),agg.__name__,topF,ver)
                    print(label)
    
                    # output coeficients
                    coef = pd.DataFrame(zip(X_train.columns, model.coef_[0]))
                    coef[['type','code','modifier']] = coef[0].str.split("_",n=2,expand=True)
                    coef['coef'] = coef[1]
                    coef = coef.drop([0,1],axis=1)
                    coef = coef.sort_values(by=['coef'], ascending=False)
                    coef = processFeatImp(coef,model='lreg')
                    coef.to_csv('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/{}/{}_Coef.csv'.format(Date,folder,label),
                                index=False)
    
                    # save model
                    picklefile = open('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/{}/{}_model.pickle'.format(Date,folder,label),'wb')
                    pickle.dump(model,picklefile)
                    picklefile.close()
    
                    #Evaluate model performance
                    metricsDF,aucDF = ML_performances(model,y_test,X_test,scoring,Date,folder,label,aucDF,metricsDF,ba,agg,topF,ver,MRNs,anom=False)
    
    
    
    
                    '''
                    ### RANDOM FOREST
                    
                    print('Filter RF DF:',RFdffil.shape)
                    
                    ### create test/train sets 
                    # classifier
                    X_train,y_train,X_test,y_test,MRNs = createTestTrain(df=RFdffil,
                                                                    testID=test_set,trainID=train_set,
                                                                    func='Classifier')
                    
                    savedf = pd.DataFrame(columns=X_test.columns)
                    savedf.to_csv('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/RF/AlignDF_RF_{}_ba{}_{}_Top{}_v{}.csv'.format(Date,Date,str(ba),agg.__name__,topF,ver),index=False)
    
                    # Model
                    rf = RandomForestClassifier()
                    param_grid = {'class__n_estimators':[80,160,320,640],'class__max_leaf_nodes':[10,20,40,80,160],
                                  'class__max_depth':[1,2,4,8,16,32,64,128]}
                    pipeline = Pipeline([('sampling',ros),('class',rf)])
                    rf_cv = RandomizedSearchCV(pipeline,param_grid, scoring='roc_auc', cv=5, n_iter=iterations)
                    rf_cv.fit(X_train,y_train)
                    model= rf_cv
                    scoring = 'pp'
                    folder = 'RF'
    
                    label = '{}_{}_ba{}_{}_Top{}_v{}'.format(folder,Date,str(ba),agg.__name__,topF,ver)
                    print(label)
                    print(model.best_params_)
    
                    # output feature importance
                    feat = pd.DataFrame(zip(X_train.columns,model.best_estimator_['class'].feature_importances_),
                                          columns=['Feature','Importance']).sort_values(by='Importance',ascending=False)
                    feat = processFeatImp(feat,model='rf')
                    feat.to_csv('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/{}/{}_Feat.csv'.format(Date,folder,label),
                                  index=False)
                    # write parameters
                    param = open('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/{}/{}_Param.txt'.format(Date,folder,label),'w')
                    param.write(str(model.best_params_)+'\n')
                    results = model.cv_results_
                    for mean_score, params in zip(results['mean_test_score'], results['params']):
                        param.write(str(mean_score)+';'+str(params)+'\n')
                    param.close()
    
                    # save model
                    picklefile = open('C:/Users/sdt0071/Desktop/TempProjectData/IMMUNE/CA_PheAllEvent/{}_TopFeat_OrigRF_Boot/{}/{}_model.pickle'.format(Date,folder,label),'wb')
                    pickle.dump(model,picklefile)
                    picklefile.close()
    
                    #Evaluate model performance
                    metricsDF,aucDF = ML_performances(model,y_test,X_test,scoring,Date,folder,label,aucDF,metricsDF,ba,agg,topF,ver,MRNs,anom=False)
                    '''
    



#%%====================================================

# Load data pull
rawdf = pd.concat([pd.read_csv(rawfilepath, dtype=str,encoding='iso-8859-1') for rawfilepath in glob.glob('R:/IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/BatchSQL_ICIonly_220113/*.csv')],ignore_index=True)

# filter for date
rawdf = rawdf.loc[(rawdf['date'] <= '2021-09-28 00:00:00')].copy()

rawdf = rawdf.drop(rawdf[(rawdf['MRN'].isna()) |
                          (rawdf['code'].isna()) |
                          (rawdf['date'].isna())].index)

rawdf = diag_icd9to10(rawdf)

rawdf = keepCPTonly(rawdf)

FullGridSearch(rawdf,iterations=10,featsel='original',versions=50)





