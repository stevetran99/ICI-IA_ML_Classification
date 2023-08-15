
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import shap

from scipy.stats import fisher_exact

import glob

import datetime as dt
import os

#%%
### TODAY'S DATE
Date = dt.date.today().strftime('%y%m%d')

### New directory
mainpath = 'R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/BootstrapAnalysis/{}/'.format(Date)

if not os.path.exists(mainpath):
    os.makedirs(mainpath)

with open('CA_Phe_AllCode_Bootstrap_PostAnalysis.py','r') as thisFile:
    logData = thisFile.read()
    with open(mainpath+'ScriptSave_{}.py'.format(Date),'w') as logFile:
        logFile.write(logData)
thisFile.close()
logFile.close()


#%% FUNCTIONS====================================================

def calculate_M_CI(vallist):
    
    mean = np.mean(vallist)
    ci = 2*np.std(vallist)
    
    return mean, mean-ci, mean+ci


def calculate_YoudenJ_Acc(metricsdf):
    
    met_sens = metricsdf.loc[(metricsdf['metric']=='Sensitivity') & (metricsdf['threshType']=='J')].copy()
    met_spec = metricsdf.loc[(metricsdf['metric']=='Specificity') & (metricsdf['threshType']=='J')].copy()
    
    met_J = met_sens.merge(met_spec,how='inner',
                           on=['BefAft','Agg','TopFeat','Ver','model'])
    print(met_J.head())
    
    met_Acc = met_J.copy()
    
    met_J['value'] = met_J['value_x'] + met_J['value_y'] - 1
    met_J['metric'] = 'YoudenJ'
    met_J['threshType'] = 'J'
    
    met_Acc['value'] = met_Acc['value_x'] * (89/2451) + met_Acc['value_y'] * (1 - 89/2451)
    met_Acc['metric'] = 'Accuracy'
    met_Acc['threshType'] = 'J'
    
    return pd.concat([metricsdf,
                      met_J[['BefAft','Agg','TopFeat','Ver','model','metric','threshType','value']],
                      met_Acc[['BefAft','Agg','TopFeat','Ver','model','metric','threshType','value']]],
                     ignore_index=True)


def calculate_bootMetrics(metricsdf):
                            
                            
    met_mean = metricsdf.groupby(['BefAft', 'Agg', 'TopFeat', 'model', 'metric', 'threshType'])[['value']].mean()
    met_std = metricsdf.groupby(['BefAft', 'Agg', 'TopFeat', 'model', 'metric', 'threshType'])[['value']].std() \
                            .rename(columns={'value':'std'})
                            
    met_1 = met_mean.merge(met_std,how='inner',on=['BefAft', 'Agg', 'TopFeat', 'model', 'metric', 'threshType'])
    met_1['lowCI'] = met_1['value'] - 2*met_1['std']
    met_1['uppCI'] = met_1['value'] + 2*met_1['std']
    
    return met_1.reset_index()


def plot_metricVfeat(metdf,metric_name):
    
    dfin = metdf.loc[(metdf['metric']==metric_name)].copy()
    dfin.loc[(dfin['model']=='LReg'),'model'] = 'Logistic Regression'
    dfin.loc[(dfin['model']=='RF'),'model'] = 'Random Forest'
    dfin = dfin.sort_values(['model'],ascending=False)
    maxtf = dfin['TopFeat'].max()
    dfin['Perc_TopFeat'] = dfin['TopFeat'] / maxtf * 100
    
    f,ax = plt.subplots(figsize=(10,6), dpi=600)

    sns.lineplot(ax=ax,data=dfin,
                    x='Perc_TopFeat',y='value',
                    hue='model',style='model')
    
    ax.set(xscale='log')
    ax.set_xticks([.1,1,10,50,100])
    ax.set_xticklabels([.1,1,10,50,100])
    
    plt.axvline(x=0.1,linestyle='--',color='k')
    
    handles,labels = plt.gca().get_legend_handles_labels()
    order = [0,1]
    #plt.legend(bbox_to_anchor=(.99,.01),loc='lower right',borderaxespad=0)
    plt.legend(handles=[handles[idx] for idx in order],labels=[labels[idx] for idx in order],bbox_to_anchor=(.99,.7),loc='upper right',borderaxespad=0)
    plt.ylabel(metric_name)
    plt.xlabel('Percent of Top Features')
    
    plt.show()



def calculate_FeatImp(featdf):
    
    feat_mean = featdf.groupby(['type','code','modifier','name','vocab','LongName'])[['Importance']].mean()
    feat_std = featdf.groupby(['type','code','modifier','name','vocab','LongName'])[['Importance']].std() \
                        .rename(columns={'Importance':'std'})
    
    feat_1 = feat_mean.merge(feat_std,how='inner',on=['type','code','modifier','name','vocab','LongName'])
    feat_1['lowCI'] = feat_1['Importance'] - 2*feat_1['std']
    feat_1['uppCI'] = feat_1['Importance'] + 2*feat_1['std']
    
    
    return feat_1.sort_values(['Importance'],ascending=False).reset_index()


def calculate_FeatCoef(coefdf):
    
    coef_mean = coefdf.groupby(['type','code','modifier','name','vocab','LongName'])[['coef']].mean()
    coef_std = coefdf.groupby(['type','code','modifier','name','vocab','LongName'])[['coef']].std() \
                        .rename(columns={'coef':'std'})
    
    coef_1 = coef_mean.merge(coef_std,how='inner',on=['type','code','modifier','name','vocab','LongName'])
    coef_1['lowCI'] = coef_1['coef'] - 2*coef_1['std']
    coef_1['uppCI'] = coef_1['coef'] + 2*coef_1['std']
    
    coef_1['magnitude'] = np.abs(coef_1['coef'])
    
    
    return coef_1.sort_values(['magnitude'],ascending=False).reset_index()



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



def zeroError(a,b):
    try:
        return a/b
    except ZeroDivisionError as e:
        print('zero division')
        return 0


def fishTest(df,itemlist=None,beforeCA=False):
    
    cadfin = df.loc[(df['cohort']=='Case')].copy().reset_index(drop=True)
    ncadfin = df.loc[(df['cohort']=='Control')].copy().reset_index(drop=True)

    fish_results = pd.DataFrame(columns=['Feature','p_value','odds','odds_val','odds_low','odds_high','CA_frac','CA_ct','nCA_frac','nCA_ct'])
    
    if itemlist is None:
        itemlist = list(df['Predictor'].unique())
    
    
    # iterate through irAEs
    for value in itemlist:
        print(value)
        
        obs = [cadfin.loc[(cadfin['Predictor']==value)]['mrn'].nunique(),
               cadfin['mrn'].nunique() - cadfin.loc[(cadfin['Predictor']==value)]['mrn'].nunique()]
        obs1 = [x/float(sum(obs)) for x in obs]
        exp = [ncadfin.loc[(ncadfin['Predictor']==value)]['mrn'].nunique(),
               ncadfin['mrn'].nunique() - ncadfin.loc[(ncadfin['Predictor']==value)]['mrn'].nunique()]
        exp1 = [x/float(sum(exp)) for x in exp]
        
        print('CA:',obs,obs1)
        print('nCA:',exp,exp1)
        #print(np.array([obs,exp]))
        
        fp='None'
        odds='None'
        oddsstr='None'
        if obs[0] != 0 and exp[0] != 0:
            odds,fp = fisher_exact(np.array([obs,exp]))
            try: cil = np.exp(np.log(odds)-1.96*np.sqrt(1/obs[0]+1/obs[1]+1/exp[0]+1/exp[1]))
            except: cil='nan'
            try: ciu = np.exp(np.log(odds)+1.96*np.sqrt(1/obs[0]+1/obs[1]+1/exp[0]+1/exp[1])) 
            except: ciu='nan'
            oddsstr = '{:.3f} ({:.3f}-{:.3f})'.format(odds,float(cil),float(ciu))
            
            print('fp =',fp,'| odds=',oddsstr)
        
        else:
            cil='nan'
            ciu='nan'

        fish_results = pd.concat([fish_results,
                                pd.DataFrame({'Feature':[value],
                                              'p_value':[fp],
                                              'odds':[oddsstr],
                                              'odds_val':[odds],
                                              'odds_low':[cil],
                                              'odds_high':[ciu],
                                              'CA_frac':[obs1[0]],
                                              'CA_ct':[obs[0]],
                                              'nCA_frac':[exp1[0]],
                                              'nCA_ct':[exp[0]]})])
            
            
    fish_results['CA_string'] = fish_results['CA_ct'].astype(str) +' ('+(fish_results['CA_frac']*100.0).astype(float).round(2).astype(str)+')'
    fish_results['nCA_string'] = fish_results['nCA_ct'].astype(str) +' ('+(fish_results['nCA_frac']*100.0).astype(float).round(2).astype(str)+')'

        
    return fish_results



#%%====================================================


#%% METRICS DATA
met_r = pd.read_csv('R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/230307_TopFeat_OrigRF_Boot_2557/metrics_230307.csv')

met_1 = calculate_YoudenJ_Acc(met_r)

met_1 = calculate_bootMetrics(met_1)

met_1['valueCI_str'] = met_1['value'].round(3).astype(str) + ' (' + met_1['lowCI'].round(3).astype(str) + '-' + met_1['uppCI'].round(3).astype(str) + ')'

met_1.to_csv('R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/230307_TopFeat_OrigRF_Boot_2557/metrics_bootagg_230307_v2.csv',index=False)

#plot_metricVfeat(met_r,'ROCAUC')

#%% RF FEATURE DATA

#### 
# batch read in all feature data and merge into DF
feat_r = pd.DataFrame()
directory = 'R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/230307_TopFeat_OrigRF_Boot_2557/RF/'
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.startswith('RF_230307_ba1_agg_presence_Top32682_') and filename.endswith('Feat.csv'):
        filepath = '/'.join([os.fsdecode(directory),filename])
        #print('Processing raw batch csv file:',filename)
        adddf = pd.read_csv(filepath,encoding='ISO-8859-1')
            
        if feat_r.empty:
            feat_r = adddf.copy()
        else:
            feat_r = pd.concat([feat_r,adddf],ignore_index=True)
    else:
        continue
                        
#%%

feat_1 = calculate_FeatImp(feat_r)
feat_1['feature'] = feat_1['type'] + '_' + feat_1['code'] + '_' + feat_1['modifier']

#feat_1.to_csv('R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/230307_TopFeat_OrigRF_Boot_2557/RF_FeatImp_230307.csv',index=False)



#%% RF FEATURE DATA

# batch read in all feature data and merge into DF
coef_r = pd.DataFrame()
directory = 'R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/230307_TopFeat_OrigRF_Boot_2557/LReg/'
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.startswith('LReg_230307_ba1_agg_presence_Top32682_') and filename.endswith('Coef.csv'):
        filepath = '/'.join([os.fsdecode(directory),filename])
        #print('Processing raw batch csv file:',filename)
        adddf = pd.read_csv(filepath,encoding='ISO-8859-1')
            
        if coef_r.empty:
            coef_r = adddf.copy()
        else:
            coef_r = pd.concat([coef_r,adddf],ignore_index=True)
    else:
        continue



#%%

coef_1 = calculate_FeatCoef(coef_r)
coef_1['feature'] = coef_1['type'] + '_' + coef_1['code'] + '_' + coef_1['modifier']

#coef_1.to_csv('R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/230307_TopFeat_OrigRF_Boot_2557/LReg_Coef_230307.csv',index=False)


#%% CODE ASSOCIATIONS

# Load data pull
rawdf = pd.concat([pd.read_csv(rawfilepath, dtype=str,encoding='iso-8859-1') for rawfilepath in glob.glob('R:/IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/BatchSQL_ICIonly_220113/*.csv')],ignore_index=True)

rawdf = rawdf.drop(rawdf[(rawdf['MRN'].isna()) |
                          (rawdf['code'].isna()) |
                          (rawdf['date'].isna())].index)

rawdf = diag_icd9to10(rawdf)

rawdf = keepCPTonly(rawdf)

rawdf = agg_presence((varPredictor(rawdf)))


#%%

ICICohort = pd.read_csv('R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_ChartReview/CA_fullCohortMRNs_230323.csv',dtype=str)
raw_1 = ICICohort.merge(rawdf,how='inner',left_on='mrn',right_on='MRN').reset_index(drop=True)


#%%

coef_fish = fishTest(raw_1,itemlist=list(coef_1['feature'].unique()))

#coef_fish.to_csv('R:IPHAM/CHIP/Projects/IMMUNE/Data/CA_PheAllEvent/230307_TopFeat_OrigRF_Boot_2557/Feature_Associations_230307.csv',index=False)










