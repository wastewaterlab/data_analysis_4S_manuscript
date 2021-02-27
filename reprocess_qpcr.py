import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import math
from pandas.api.types import CategoricalDtype
from scipy import stats as sci
from scipy.stats import linregress
from scipy.stats.mstats import gmean
from scipy.stats import gstd
from sklearn.utils import resample
import pdb
from sklearn.metrics import r2_score
from statistics import median
import scikit_posthocs as sp
import warnings
# np.seterr(all='raise')


def get_pass_grubbs_test(plate_df, groupby_list, col="Cq"):
  # make list that will become new df
  plate_df_with_grubbs_test = pd.DataFrame()

  # iterate thru the dataframe, grouped by Sample
  # this gives us a mini-df with just one sample in each iteration
  for groupby_list, df in plate_df.groupby(groupby_list,  as_index=False):
    d = df.copy() # avoid set with copy warning

    # make new column 'grubbs_test' that includes the results of the test
    if (len(d[col].dropna())<3): #cannot evaluate for fewer than 3 values
        if (len(d[col].dropna())==2): #no outlier removal if just 2 pts
            d.loc[:, 'grubbs_test'] = True
            plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)
        else:
            d.loc[:, 'grubbs_test'] = False
            plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)

    else:

        b=list(d[col]) #needs to be given unindexed list
        if all([element == b[0] for element in b]): #grubbs doesn't like when all are the same
            d.loc[:, 'grubbs_test'] = True
            plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)
        else:
            nonoutliers= sp.outliers_grubbs(b)
            outlier_len=len(b)-len(nonoutliers)
            if outlier_len > 0:
                d.loc[:, 'grubbs_test'] = False
                d.loc[d[col].isin(nonoutliers), 'grubbs_test'] = True
                plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)
            else:
                d.loc[:, 'grubbs_test'] = True
                plate_df_with_grubbs_test=plate_df_with_grubbs_test.append(d)
  return(plate_df_with_grubbs_test)


def compute_linear_info(plate_data):
    '''compute the information for linear regression

    Params
        plate_data: pandas dataframe with columns
            Cq_mean (already processed to remove outliers)
            log_Quantity
    Returns
        slope, intercept, r2, efficiency
    '''
    y = plate_data['Cq_mean']
    x = plate_data['log_Quantity']
    model = np.polyfit(x, y, 1)
    predict = np.poly1d(model)
    r2 = r2_score(y, predict(x))
    slope, intercept = model
    efficiency = (10**(-1/slope)) - 1

    return(slope, intercept, r2, efficiency )

def combine_triplicates(plate_df_in, checks_include, master, use_master_curve):
    '''
    Flag outliers via grubbs test
    Calculate the Cq means, Cq stds, counts before & after removing outliers

    Params
    plate_df_in:
        qpcr data in pandas df, must be 1 plate with 1 target
        should be in the format from QuantStudio3 with
        columns 'Target', 'Sample', 'Cq'
    checks_include: must be set to 'grubbs_only'

    Returns
    plate_df: same data, with additional columns depending on checks_include
        grubbs_test (True or False) -- did it pass
        Cq_mean (calculated mean of Cq after excluding outliers)

    Note: Cq_raw preserves the raw values, Cq_fin is after subbing and outlier removal, and plain Cq_subbed is after subbing (so that it goes through grubbs)

    '''

    if (checks_include not in ['grubbs_only']):
        raise ValueError('''invalid input, must be grubbs_only''')

    if len(plate_df_in.Target.unique()) > 1:
        raise ValueError('''More than one target in this dataframe''')

    target= plate_df_in.Target.unique()
    plate_df = plate_df_in.copy() # fixes pandas warnings

    groupby_list = ['plate_id', 'Sample', 'sample_full','Sample_plate',
                    'Target','Task', 'inhibition_testing','is_dilution',"dilution"]

    # make copy of Cq column and later turn this to np.nan for outliers
    plate_df['Cq_raw'] = plate_df['Cq'].copy()
    plate_df["master_curve_bloq_qpcr_reps"]=False
    if ((use_master_curve) & (target[0] != "Xeno")):
        plate_df.loc[ (np.isnan(plate_df.Cq))| (plate_df.Cq>40), "master_curve_bloq_qpcr_reps"]= True
        plate_df.loc[ (np.isnan(plate_df.Cq))| (plate_df.Cq>40), "Cq"]= master.loc[master.Target==target[0], "LoD_Cq"].item()

    plate_df['Cq_subbed'] = plate_df['Cq'].copy()
    plate_df['Cq_fin'] = plate_df['Cq'].copy()

    # grubbs with scikit
    if checks_include == 'grubbs_only':
       plate_df = get_pass_grubbs_test(plate_df, ['Sample'])
       plate_df.loc[plate_df.grubbs_test== False, 'Cq_fin'] = np.nan

    # summarize to get mean, std, counts with and without outliers removed
    plate_df_avg = plate_df.groupby(groupby_list).agg(
                                               raw_Cq_values=('Cq_raw',list),
                                               sub_Cq_values=('Cq_subbed',list),
                                               outlier_Cq_values=('Cq_fin',list),
                                               template_volume=('template_volume','max'),
                                               Q_init_mean=('Quantity','mean'), #only needed to preserve quantity information for standards later
                                               Q_init_std=('Quantity','std'),
                                               Q_init_gstd=('Quantity', lambda x: np.nan if ( (len(x.dropna()) <2 )| all(np.isnan(x)) ) else (sci.gstd(x.dropna(),axis=0))),
                                               Cq_init_mean=('Cq_raw', 'mean'),
                                               Cq_init_std=('Cq_raw', 'std'),
                                               Cq_init_min=('Cq_raw', 'min'),
                                               replicate_init_count=('Cq','count'),
                                               Cq_mean=('Cq_fin', 'mean'),
                                               Cq_std=('Cq_fin', 'std'),
                                               replicate_count=('Cq_fin', 'count'),
                                               is_undetermined_count=('is_undetermined', 'sum'),
                                               is_bloq_count=('master_curve_bloq_qpcr_reps', 'sum')
                                               )
    plate_df_avg = plate_df_avg.reset_index()
    return(plate_df, plate_df_avg)

def process_standard(plate_df):
    '''
    from single plate with single target, calculate standard curve

    Params:
        plate_df: output from combine_triplicates(); df containing Cq_mean
        must be single plate with single target
    Returns
        num_points: number of points used in new std curve
        Cq_of_lowest_std_quantity: the Cq value of the lowest pt used in the new std curve
        lowest_std_quantity: the Quantity value of the lowest pt used in the new std curve
        Cq_of_lowest_std_quantity_gsd: geometric standard dviation of the Cq of the lowest standard quantity
        slope:
        intercept:
        r2:
        efficiency:
    '''
    if len(plate_df.Target.unique()) > 1:
        raise ValueError('''More than one target in this dataframe''')


    #what is the lowest sample Cq and quantity on this plate
    standard_df = plate_df[plate_df.Task == 'Standard'].copy()

    # require at least 2 triplicates or else convert to nan
    standard_df = standard_df[standard_df.replicate_count > 1]

    standard_df['log_Quantity'] = np.log10(standard_df['Q_init_mean'])
    std_curve_df = standard_df[['Cq_mean', 'log_Quantity', "Cq_std"]].drop_duplicates().dropna()
    num_points = std_curve_df.shape[0]

    if (all(standard_df.Cq_mean == "") | len(standard_df.Cq_mean) <2):
        slope, intercept, r2, efficiency,Cq_of_lowest_std_quantity,Cq_of_lowest_std_quantity_gsd,lowest_std_quantity =  np.nan, np.nan, np.nan,np.nan, np.nan,np.nan, np.nan
    else:
        #find the Cq of the lowest standard quantity
        Cq_of_lowest_std_quantity = max(standard_df.Cq_mean)
        sort_a=standard_df.sort_values(by='Cq_mean',ascending=True).copy().reset_index()
        sort_a=standard_df.sort_values(by='Cq_mean',ascending=True).copy().reset_index()
        Cq_of_lowest_std_quantity_gsd = sort_a.Cq_std[0]

        # the  lowest standard quantity
        lowest_std_quantity = np.nan
        sort_b=standard_df.sort_values(by='Q_init_mean',ascending=True).copy().reset_index()
        slope, intercept, r2, efficiency = (np.nan, np.nan, np.nan, np.nan)

        if num_points > 2:
            lowest_std_quantity = sort_b.Q_init_mean.values[0]
            slope, intercept, r2, efficiency = compute_linear_info(std_curve_df)


    return(num_points, Cq_of_lowest_std_quantity,  lowest_std_quantity, Cq_of_lowest_std_quantity_gsd,  slope, intercept, r2, efficiency)

def process_unknown(plate_df, std_curve_info, use_master_curve, master):
    '''
    Calculates quantity based on Cq_mean and standard curve
    Params
        plate_df: output from combine_triplicates(); df containing Cq_mean
        must be single plate with single target
        std_curve_info: output from process_standard() as a list
    Returns
        unknown_df: the unknown subset of plate_df, with new columns
        Quantity_mean
        q_diff
        Cq_of_lowest_sample_quantity: the Cq value of the lowest pt used on the plate
        these columns represent the recalculated quantity using Cq mean and the
        slope and intercept from the std curve
        qpcr_coefficient_var the coefficient of variation for qpcr technical triplicates
        intraassay_var intraassay variation (arithmetic mean of the coefficient of variation for all triplicates on a plate)
    '''

    [num_points, Cq_of_lowest_std_quantity,  lowest_std_quantity, Cq_of_lowest_std_quantity_gsd, slope, intercept, r2, efficiency] = std_curve_info
    unknown_df = plate_df[plate_df.Task != 'Standard'].copy()
    unknown_df['Cq_of_lowest_sample_quantity'] = np.nan
    unknown_df['percent_CV']=(unknown_df['Q_init_gstd']-1)*100#the geometric std - 1 is the coefficient of variation using quant studio quantities to capture all the variation in the plate
    if all(np.isnan(unknown_df['percent_CV'])):
        unknown_df['intraassay_var'] = np.nan #avoid error
    else:
        unknown_df['intraassay_var']= np.nanmean(unknown_df['percent_CV'])

    # Set the Cq of the lowest std quantity for different situations
    if len(unknown_df.Task) == 0: #only standard curve plate
        unknown_df['Cq_of_lowest_sample_quantity'] = np.nan
    else:
        if all(np.isnan(unknown_df.Cq_mean)): #plate with all undetermined samples
            unknown_df['Cq_of_lowest_sample_quantity']= np.nan #avoid error
        else:
            targs=unknown_df.Target.unique() #other  plates (most  cases)
            for target in targs:
                unknown_df.loc[(unknown_df.Target==target),'Cq_of_lowest_sample_quantity']=np.nanmax(unknown_df.loc[(unknown_df.Target==target),'Cq_mean']) #because of xeno

    unknown_df['Quantity_mean'] = np.nan
    unknown_df['q_diff'] = np.nan

    if ~use_master_curve:
        unknown_df["blod_master_curve"]=False
        unknown_df['Quantity_mean'] = 10**((unknown_df['Cq_mean'] - intercept)/slope)

        #initialize columns
        unknown_df['Quantity_std_combined_after']=np.nan
        unknown_df['Quantity_mean_combined_after']=np.nan
        for row in unknown_df.itertuples():
            ix=row.Index
            filtered_1= [element for element in row.raw_Cq_values if ~np.isnan(element) ] #initial nas
            filtered= [10**((element - intercept)/slope) for element in filtered_1]
            if(len(filtered)>1):
                    filtered= [element for element in filtered if ~np.isnan(element) ] #nas introduced when slope and interceptna
                    if(len(filtered)>1):
                        if (row.Target != "Xeno"):
                            unknown_df.loc[ix,"Quantity_mean_combined_after"]=sci.gmean(filtered)
                            if(all(x >0 for x in filtered)):
                                unknown_df.loc[ix,"Quantity_std_combined_after"]=sci.gstd(filtered)
    if use_master_curve:
        targs=unknown_df.Target.unique()
        for targ in targs:
            if targ != "Xeno":
                unknown_df["blod_master_curve"]=False
                m_b=master.loc[master.Target==targ, "b"].item()
                m_m=master.loc[master.Target==targ, "m"].item()
                lowest=master.loc[master.Target==targ, "lowest_quantity"].item()
                lod=master.loc[master.Target==targ, "LoD_quantity"].item()
                unknown_df.loc[unknown_df.Target==targ, 'Quantity_mean'] = 10**((unknown_df.loc[unknown_df.Target==targ, 'Cq_mean'] - m_b)/m_m)
                unknown_df.loc[unknown_df.Quantity_mean< lowest, "blod_master_curve"] = True
                unknown_df.loc[unknown_df.Quantity_mean< lowest, 'Quantity_mean'] = lod

    unknown_df.loc[unknown_df[unknown_df.Cq_mean == 0].index, 'Quantity_mean'] = np.nan
    unknown_df['q_diff'] = unknown_df['Q_init_mean'] - unknown_df['Quantity_mean']

    return(unknown_df)

def process_ntc(plate_df):
    ntc = plate_df[plate_df.Task == 'Negative Control']
    ntc_result = np.nan
    if ntc.is_undetermined.all():
        ntc_result = 'negative'
    else:
        if all(np.isnan(ntc.Cq)):
            ntc_result = np.nan #avoid error
        else:
            ntc_result = np.nanmin(ntc.Cq)
    return(ntc_result)



def determine_samples_BLoD(raw_outliers_flagged_df, checks_include):
        '''
        For each target in raw qpcr data, this function defines the limit of detection as the fraction of qpcr replicates at a quantity that are detectable
        It works depending on which test was selected, so if grubbs was selected, it only evaluates for replicates that pass grubbs

        Params:
            Task
            Quantity
            Target
            Cq
            Sample
        prints a dataframe with Target and the limit of detection
        '''
        dfm= raw_outliers_flagged_df
        dfm=dfm[dfm.Task=='Standard'] #only standards
        dfm=dfm[dfm.Quantity!=0] #no NTCs
        assay_assessment_df=pd.DataFrame(columns=["Target","LoD_Cq","LoD_Quantity"]) #empty dataframe with desired columns

        #iterate through targets, groupby quantity, and determine the fraction of the replicates that were detectable
        targs=dfm.Target.unique()
        for target in targs:
            print(target)
            df_t=dfm[dfm.Target==target].copy()
            out=df_t.groupby(["Quantity"]).agg(
                                    Cq_mean=('Cq', 'mean'),
                                    negatives=('is_undetermined','sum'),
                                    total=('Sample', 'count')).reset_index()
            out['fr_pos']=(out.total-out.negatives)/out.total
            print(out)
        return (assay_assessment_df)


def determine_samples_BLoQ(qpcr_p):
    '''
    from processed unknown qpcr data this will return qpcr_processed with a boolean column indicating samples bloq
    (as defined as Cq < 40 and within the standard curve)
    samples that have Cq_mean that is nan are classified as bloq (including true negatives and  samples removed during processing)
    Params:
        Cq_mean the combined triplicates of the sample
        Cq_of_lowest_sample_quantity the max cq of the samples on the plate

    Returns
        same data with column bloq a boolean column indicating if the sample is below the limit of quantification
    '''
    qpcr_p['bloq']=np.nan
    qpcr_p.loc[(np.isnan(qpcr_p.Cq_mean)),'bloq']= True
    qpcr_p.loc[(qpcr_p.Cq_mean >= 40),'bloq']= True
    qpcr_p.loc[(qpcr_p.Cq_mean > qpcr_p.Cq_of_lowest_std_quantity),'bloq']= True
    qpcr_p.loc[(qpcr_p.Cq_mean <= qpcr_p.Cq_of_lowest_std_quantity)&(qpcr_p.Cq_mean < 40),'bloq']= False
    return(qpcr_p)


def process_qpcr_raw(qpcr_raw, master=np.nan, use_master_curve=False, checks_include='grubbs_only'):
    '''wrapper to process whole sheet at once by plate_id and Target
    params
    qpcr_raw: df from read_qpcr_data()
    checks_include must be grubbs_only
    '''
    if (checks_include not in ['grubbs_only']):
        raise ValueError('''invalid input, must be grubbs_only''')
    std_curve_df = []
    qpcr_processed = []
    raw_outliers_flagged_df = []
    for [plate_id, target], df in qpcr_raw.groupby(["plate_id", "Target"]):

        ntc_result = process_ntc(df)
        outliers_flagged, no_outliers_df = combine_triplicates(df, checks_include, master, use_master_curve)

        # define outputs and fill with default values
        num_points,  Cq_of_lowest_std_quantity,lowest_std_quantity, Cq_of_lowest_std_quantity_gsd, slope, intercept, r2, efficiency = np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # if there are >3 pts in std curve, calculate stats and recalculate quants
        num_points = no_outliers_df[no_outliers_df.Task == 'Standard'].drop_duplicates('Sample').shape[0]
        num_points,  Cq_of_lowest_std_quantity, lowest_std_quantity, Cq_of_lowest_std_quantity_gsd, slope, intercept, r2, efficiency = process_standard(no_outliers_df)
        std_curve_info = [num_points,  Cq_of_lowest_std_quantity, lowest_std_quantity, Cq_of_lowest_std_quantity_gsd, slope, intercept, r2, efficiency]
        unknown_df = process_unknown(no_outliers_df, std_curve_info, use_master_curve, master)
        std_curve_df.append([plate_id, target, num_points,  Cq_of_lowest_std_quantity, lowest_std_quantity, Cq_of_lowest_std_quantity_gsd, slope, intercept, r2, efficiency, ntc_result])
        qpcr_processed.append(unknown_df)
        raw_outliers_flagged_df.append(outliers_flagged)

    # compile into dataframes
    raw_outliers_flagged_df = pd.concat(raw_outliers_flagged_df)
    assay_assessment_df=determine_samples_BLoD(raw_outliers_flagged_df, checks_include) #only prints a dataframe
    std_curve_df = pd.DataFrame.from_records(std_curve_df,
                                             columns = ['plate_id',
                                                        'Target',
                                                        'num_points',
                                                        'Cq_of_lowest_std_quantity',
                                                        'lowest_std_quantity',
                                                        'Cq_of_lowest_std_quantity_gsd',
                                                        'slope',
                                                        'intercept',
                                                        'r2',
                                                        'efficiency',
                                                        'ntc_result'])
    qpcr_processed = pd.concat(qpcr_processed)
    qpcr_processed = qpcr_processed.merge(std_curve_df, how='left', on=['plate_id', 'Target'])
    control_df=qpcr_processed[(qpcr_processed.Sample.str.contains("control"))|(qpcr_processed.Task!="Unknown")].copy()
    qpcr_processed=qpcr_processed[qpcr_processed.Task=="Unknown"].copy()

    #make  columns calculated in other functions to go in the standard curve info
    qpcr_m=qpcr_processed[["plate_id","Target","Cq_of_lowest_sample_quantity",'intraassay_var']].copy().drop_duplicates(keep='first')
    std_curve_df=std_curve_df.merge(qpcr_m, how='left') # add Cq_of_lowest_sample_quantity and intraassay variation

    if ~use_master_curve:
        qpcr_processed= determine_samples_BLoQ(qpcr_processed)
    std_curve_df=std_curve_df[std_curve_df.Target != "Xeno"].copy()

    #check for duplicates
    a=qpcr_processed[(qpcr_processed.Sample!="__")&(qpcr_processed.Sample!="")]
    a=a[a.duplicated(["Sample","Target"],keep=False)].copy()
    if len(a) > 0:
        plates=a.plate_id.unique()
        l=len(plates)
        warnings.warn("\n\n\n {0} plates have samples that are double listed in qPCR_Cts spreadsheet. Check the following plates and make sure one is_primary_value is set to N:\n\n\n{1}\n\n\n".format(l,plates))

    return(qpcr_processed, std_curve_df, raw_outliers_flagged_df, control_df)
