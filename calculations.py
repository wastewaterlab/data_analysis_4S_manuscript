import pandas as pd
import numpy as np
from scipy import stats as sci
from scipy.stats.mstats import gmean
from scipy.stats import gstd


def calculate_gc_per_l(qpcr_data ):
    '''
    calculates and returns gene copies / L

    Params
    qpcr_data-- dataframe with qpcr technical triplicates averaged. Requires the columns
            gc_per_ul_input
            Quantity_mean
            template_volume
            elution_vol_ul
            weight_vol_extracted_ml
    Returns
    qpcr_data: same data, with additional column
    gc_per_L
    '''
    qpcr_data['gc_per_ul_input'] = qpcr_data['Quantity_mean'].astype(float) / qpcr_data['template_volume'].astype(float)
    qpcr_data['gc_per_L']= np.nan
    qpcr_data.loc[qpcr_data.blod_master_curve== True, 'gc_per_L'] = 1000 * qpcr_data['gc_per_ul_input'].astype(float) * qpcr_data['elution_vol_ul'].astype(float) / 50
    qpcr_data.loc[qpcr_data.blod_master_curve!= True, 'gc_per_L'] = 1000 * qpcr_data['gc_per_ul_input'].astype(float) * qpcr_data['elution_vol_ul'].astype(float) / qpcr_data['weight_vol_extracted_ml'].astype(float)
    return qpcr_data['gc_per_L']


def normalize_to_pmmov(qpcr_data, replace_bloq= False):

    '''
    calculates a normalized mean to pmmov when applicable and returns dataframe with new columns

      Params
        qpcr_data-- dataframe with qpcr technical triplicates averaged. Requires the columns
                Target
                Quantity_mean
                Sample
                Task
      Returns
      qpcr_m: same data, with additional columns
            mean_normalized_to_pmmov: takes every column and divides by PMMoV that is associated with that sample name (so where target == PMMoV it will be 1)
            log10mean_normalized_to_log10pmmov: takes the log10 of N1 and the log 10 of PMMoV then normalizes
            log10_mean_normalized_to_pmmov: takes the log10 of mean_normalized_to_pmmov
    '''
    pmmov=qpcr_data[qpcr_data.Target=='PMMoV']
    pmmov=pmmov[['Quantity_mean','Sample','Task']]
    pmmov.columns=['pmmov_mean',  "Sample", "Task"]
    qpcr_m=qpcr_data.merge(pmmov, how='left', on=["Sample", "Task"])
    qpcr_m["mean_normalized_to_pmmov"] = qpcr_m['Quantity_mean']/qpcr_m['pmmov_mean']
    qpcr_m["log10mean_normalized_to_log10pmmov"] = np.log10(qpcr_m['Quantity_mean'])/np.log10(qpcr_m['pmmov_mean'])
    qpcr_m['log10_mean_normalized_to_pmmov']=np.log10(qpcr_m['mean_normalized_to_pmmov'])

    return qpcr_m

def normalize_to_18S(qpcr_data, replace_bloq= False):

    '''
    calculates a normalized mean to 18S when applicable and returns dataframe with new columns

      Params
        qpcr_data-- dataframe with qpcr technical triplicates averaged. Requires the columns
                Target
                Quantity_mean
                Sample
                Task
      Returns
      qpcr_m: same data, with additional columns
            mean_normalized_to_18S: takes every column and divides by 18Sthat is associated with that sample name (so where target == 18S it will be 1)
            log10mean_normalized_to_log1018S: takes the log10 of N1 and the log 10 of 18S then normalizes
            log10_mean_normalized_to_18S: takes the log10 of mean_normalized_to_18S
    '''
    n_18S=qpcr_data[qpcr_data.Target=='18S']
    n_18S=n_18S[['Quantity_mean','Sample','Task']]
    n_18S.columns=['18S_mean',  "Sample", "Task"]
    qpcr_m=qpcr_data.merge(n_18S, how='left', on=["Sample", "Task"])
    qpcr_m["mean_normalized_to_18S"] = qpcr_m['Quantity_mean']/qpcr_m['18S_mean']
    qpcr_m["log10mean_normalized_to_log1018S"] = np.log10(qpcr_m['Quantity_mean'])/np.log10(qpcr_m['18S_mean'])
    qpcr_m['log10_mean_normalized_to_18S']=np.log10(qpcr_m['mean_normalized_to_18S'])
    return qpcr_m



def analyze_qpcr_inhibition(qpcr_raw):
  ''' Starting from raw qPCR data, calculate average,
  filter for data generated inhibition testing
  make column for sample name and dilution factor,
  then calculate dCt and expected dCt
  '''
  qpcr_inhibition=qpcr_raw[qpcr_raw.inhibition_testing== "Y"].copy()
  qpcr_inhibition = qpcr_inhibition.groupby(['plate_id',"Sample",'Sample_plate', "Target",'Task','inhibition_testing']).agg(
                                                                    template_volume=('template_volume','max'),
                                                                    Cq_mean=('Cq', 'mean'),
                                                                    Cq_std=('Cq', 'std'),
                                                                    Cq_count=('Cq','count')
                                                                    ).reset_index()

  qpcr_inhibition=qpcr_inhibition[~np.isnan(qpcr_inhibition.Cq_mean)]
  qpcr_inhibition=qpcr_inhibition[qpcr_inhibition.Task=="Unknown"]
  qpcr_inhibition["dilution"]= pd.to_numeric(qpcr_inhibition["Sample"].apply(lambda x: x.split('_')[0].replace('x','')))
  qpcr_inhibition["Sample_full"]=qpcr_inhibition["Sample"]
  qpcr_inhibition["Sample"]=qpcr_inhibition["Sample"].apply(lambda x: x.split('_',1)[1])

  #calculate dCtrelative to 1x
  onex=qpcr_inhibition[qpcr_inhibition.dilution==1].copy()
  onex=onex[["Sample","Target","Cq_mean"]]
  onex=onex.rename(columns={'Cq_mean': 'onex_avg_ct' })
  qpcr_inhibition=qpcr_inhibition.merge(onex, how='left')
  qpcr_inhibition["dCt"] = qpcr_inhibition['Cq_mean']-qpcr_inhibition['onex_avg_ct']

  #merge and calculate efficiency
  # qpcr_inhibition["exp_dCt"]= np.log(qpcr_inhibition.dilution) / np.log((1+(qpcr_inhibition.efficiency)))
  qpcr_inhibition["exp_dCt_perfect"]= np.log(qpcr_inhibition.dilution) / np.log((2)) #assumes perfect efficiency
  return(qpcr_inhibition)


def fix_std_issues(qpcr_data,plates_base,plates_fix, target):
    '''Plate 35 and 38 had standard curve issues so replacing with standard curves from plates 29 and 32'''

    std_cv=qpcr_data[qpcr_data.plate_id.isin(plates_base)]
    std_cv=std_cv[std_cv.Task=='Standard'].copy()
    std_cv=std_cv[std_cv.Target==target]
    for plate in plates_fix:
      std_cv['plate_id']=plate
      qpcr_data=qpcr_data.append(std_cv)

    qpcr_data.plate_id=pd.to_numeric(qpcr_data.plate_id)
    return qpcr_data
