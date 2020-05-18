'''
Code included in a thesis presented for the degree of
Master in Innovation and Research in Informatics - Data Science
Facultat d’Informàtica de Barcelona (FIB)
Universitat Politècnica De Catalunya (UPC)

Indicators & Plots to support the identification of tasks and/or process properties 
that are eligible to be automated using RPA.

Parameters
-----------
log
    Trace log
parameters
    Parameters of the log representation algorithm:
        str_ev_attr -> String event attributes to consider in feature representation (single necessary)
        str_tr_attr -> String trace attributes to consider in feature representation (single additional optional)
Returns
-----------
indicator plots
indicator
    A list containing, per activity all indicators in each sublog 


 folder structure 
 -home
	--code: where this code is found
	--data: XES files are found
	--plots: resulting plots are saved
The code can be run from command line having navigated to code folder by executing: 
`python pm4RPA.py financial_log.xes concept:name AMOUNT_REQ`
'''


import os, sys, random, re 
import pandas as pd
import numpy as np
# used in generatePlots()
import matplotlib.pyplot as plt 
# Importing IEEE XES files used in main()
from pm4py.objects.log.importer.xes import factory as xes_import_factory
# Apply PCA + DBSCAN clustering after creating a representation of the log; used in clusterlog()
from pm4pyclustering.algo.other.clustering import factory as clusterer 
# Retrieve the number of occurrences of the activities; used in execFreq()
from pm4py.algo.filtering.log.attributes import attributes_filter
# Retrieve variants and the number of occurence; used in execFreqCase()
from pm4py.statistics.traces.log import case_statistics
# Converts a log to interval format (e.g. an event has two timestamps) 
# from lifecycle format (an event has only a timestamp, and a transition lifecycle); used in execTime()
from pm4py.objects.log.util import interval_lifecycle
# Get the paths (pair activities) of the log along with their count; used in priorFollowVar()
from pm4py.algo.filtering.log.paths import paths_filter 

def clusterlog(log,clusterParams):
	print("Apply PCA + DBSCAN clustering from log representation obtained using ", clusterParams)
	random.seed(42)
	clusters = clusterer.apply(log, parameters = clusterParams)
	for i in range(len(clusters)):
		print("sublog",i," has ",len(clusters[i]), " traces")
	return(clusters)

def execFreq(clusters, activityKey):
	EF = []
	for i in range(len(clusters)):
		activities_count = attributes_filter.get_attribute_values(clusters[i],attribute_key =activityKey)
		EF.append(activities_count)
	EF_df = pd.DataFrame.from_dict(EF, orient='columns', dtype=None).T
	EF_df = EF_df.reset_index().melt( id_vars='index', var_name='cluster', value_name='activityCount')
	EF_df = EF_df.fillna(0)
	EF_df= EF_df.rename(columns={'index':'activity'})
	##############  Execution Frequency: case ############
	EF_EFc = execFreqCase(clusters,EF_df)
	return(EF_EFc)

def execFreqCase(clusters, EF_df):
	activityL = EF_df['activity'].unique().tolist()
	variant_EF_A = []
	for clusteri in range(len(clusters)):
	    # per cluster get the variants along with their count
	    variants_count = case_statistics.get_variant_statistics(clusters[clusteri])
	    for variant in range(len(variants_count)):
	        # per variant count the number of occurence of each activity
	        for key, value in variants_count[variant].items():
	            if key=="variant":
	                activityVariant = []
	                for i in range(len(activityL)):
	                    EF = len(re.findall(activityL[i], value))
	                    if EF>0:
	                        activityVariant.append({'cluster':clusteri,'variant':variant,
	                                                'activity': activityL[i], 'EF': EF})   
	            else:
	                #also include the count of this variant
	                for item in activityVariant:
	                    item.update({"count":value})
	        variant_EF_A.extend(activityVariant)
	variant_EF_A_df = pd.DataFrame.from_dict(variant_EF_A, orient='columns', dtype=None)
	variant_EF_A_df['EFsum'] = variant_EF_A_df.apply(lambda x:  x['EF']*x['count'], axis=1)
	EFc_df = variant_EF_A_df.groupby(by=['cluster','activity']).agg({'EFsum': "sum",'count':"sum"}).reset_index()
	EFc_df['EFc'] = EFc_df.apply(lambda x:  x['EFsum']/x['count'], axis=1)
	EF_EFc_df =  pd.merge(left = EF_df,
							right=EFc_df.drop(['EFsum','count'], axis=1),
							right_on=['cluster','activity'], 
							left_on =['cluster','activity'], how='left')
	EF_EFc_df = EF_EFc_df.rename(columns={'activityCount' : 'EF' })
	EF_EFc_df = EF_EFc_df.fillna(0)
	return(EF_EFc_df)


def execTime(clusters, EF_EFc):
	ET = []
	for clusteri in range(len(clusters)):
	    enriched_log = interval_lifecycle.assign_lead_cycle_time(clusters[clusteri]) 
	    for i in range(len(enriched_log)):
	        for j in range(len(enriched_log[i])):
	            activity = enriched_log[i][j]["concept:name"]
	            duration = enriched_log[i][j]['@@duration']
	            ET.append({'cluster':clusteri,'activity': activity, 'duration': duration})
	ET_df = pd.DataFrame.from_dict(ET, orient='columns', dtype=None)
	ET_df_m = ET_df.groupby(['cluster', 'activity'])['duration'].mean().reset_index()
	ET_df_m = ET_df_m.rename(columns={'duration':'ET'})
	# per cluster mean activity duration (ET) & activity event duration  
	ET_df = pd.merge(left = ET_df , right=ET_df_m, right_on=['cluster','activity'], 
                     left_on =['cluster','activity'], how='inner')
	############## Inverse Stability ############
	EF_EFc_ET_ST = invStability(EF_EFc, ET_df)
	return(EF_EFc_ET_ST)


def invStability(EF_EFc, ET_df):
	# calculate mean squared differences between @@duration and ET
	ET_ssd = ET_df.groupby(['cluster','activity']).apply(lambda x: (x['duration']-x['ET'])**2).sum(level=[0,1]).reset_index()
	ET_ssd = ET_ssd.rename(columns={0:'ssd'})
	# include ssd in EF_EFc that contains all activities as well as EF measure
	EF_EFc_ET_ssd = pd.merge(left = EF_EFc , right=ET_ssd, right_on=['cluster','activity'], 
	                  		left_on =['cluster','activity'], how='left')
	EF_EFc_ET_ssd = EF_EFc_ET_ssd.rename(columns={0:'ssd'})
	ET_df = ET_df.drop(['duration'], axis=1).groupby(['cluster','activity','ET']).size().reset_index()
	EF_EFc_ET_ssd = pd.merge(left = EF_EFc_ET_ssd , right=ET_df.drop([0], axis=1), right_on=['cluster','activity'], 
	                  		left_on =['cluster','activity'], how='left')
	# activities that didn't have @@duration similarly we weren't able to calculate the ssd 
	EF_EFc_ET_ssd = EF_EFc_ET_ssd.fillna(0)
	# calculate inverse stability measure; the higher the number the worse
	EF_EFc_ET_ssd['ST'] = EF_EFc_ET_ssd.apply(lambda x: 0 if x['ssd']==0 else x['ssd']/(x['EF']*x['ET']), axis=1)
	EF_EFc_ET_ST = EF_EFc_ET_ssd.drop(['ssd'], axis=1)
	return(EF_EFc_ET_ST)
	


def priorFollowVar(clusters):
	# Creating an empty Dataframe with column names only
	PFv = pd.DataFrame(columns=['cluster', 'activity', 'PFv'])
	
	for clusteri in range(len(clusters)):
	    # Get the paths of the log along with their count 
	    # returns pairs activity_a, activity_b with count 
	    paths4act =  paths_filter.get_paths_from_log(clusters[clusteri])
	    paths4act_df = pd.Series(paths4act).to_frame('count').reset_index()
	    paths4act_df[['activityStart', 'activityEnd']] = paths4act_df['index'].str.split(',', n=1, expand=True)
	    paths4act_df = paths4act_df.drop(['index'], axis=1)
	    paths4act_df.sort_values(by=['activityStart','activityEnd'], inplace=True)
	    # for activity_a get all possible activity_b's
	    # calculate sum of all counts eg. {x_1, y_1,10} {x_1, y_2,30} for x_1 sum is 40
	    activityStartSum = paths4act_df[(paths4act_df['activityEnd']!=paths4act_df['activityStart'])]
	    activityStartSum = activityStartSum.groupby(by=['activityStart']).sum().groupby(level=[0]).cumsum().reset_index()
	    activityStartSum = activityStartSum.rename(columns={'count':'sum'})
	    # get maximum of those counts eg. {x_1, y_1,10} {x_1, y_2,30} for x_1 max is 30
	    activityStartMax = paths4act_df[(paths4act_df['activityEnd']!=paths4act_df['activityStart'])]
	    activityStartMax =activityStartMax.groupby(by=['activityStart']).max().reset_index().drop(['activityEnd'], axis=1)
	    activityStartMax= activityStartMax.rename(columns={'count':'max'})
	    # get number of pairs eg. {x_1, y_1,10} {x_1, y_2,30} for x_1 ndist is 2
	    activityStartNdist = paths4act_df[(paths4act_df['activityEnd']!=paths4act_df['activityStart'])]
	    activityStartNdist = activityStartNdist.groupby(by=['activityStart']).size().reset_index(name='counts')
	    activityStartNdist= activityStartNdist.rename(columns={'counts':'ndist'})
	    activityStartdf = pd.merge(left = activityStartNdist , right=activityStartSum, on=['activityStart'], how='inner')
	    activityStartdf = pd.merge(left = activityStartdf , right=activityStartMax, on=['activityStart'], how='inner')
	    activityStartdf['PFvstart'] = activityStartdf.apply(lambda x: (1.0/x['ndist']) * (x['max']/x['sum']) , axis=1)
	    # for activity_b get all possible activity_a's
	    activityEndSum = paths4act_df[(paths4act_df['activityEnd']!=paths4act_df['activityStart'])]
	    activityEndSum =activityEndSum.groupby(by=['activityEnd']).sum().groupby(level=[0]).cumsum().reset_index()
	    activityEndSum = activityEndSum.rename(columns={'count':'sum'})
	    activityEndMax = paths4act_df[(paths4act_df['activityEnd']!=paths4act_df['activityStart'])]
	    activityEndMax =activityEndMax.groupby(by=['activityEnd']).max().reset_index().drop(['activityStart'], axis=1)
	    activityEndMax= activityEndMax.rename(columns={'count':'max'})
	    activityEndNdist = paths4act_df[(paths4act_df['activityEnd']!=paths4act_df['activityStart'])]
	    activityEndNdist =activityEndNdist.groupby(by=['activityEnd']).size().reset_index(name='counts')
	    activityEndNdist= activityEndNdist.rename(columns={'counts':'ndist'})
	    activityEnddf = pd.merge(left = activityEndNdist , right=activityEndSum, on=['activityEnd'], how='inner')
	    activityEnddf = pd.merge(left = activityEnddf , right=activityEndMax, on=['activityEnd'], how='inner')
	    activityEnddf['PFvend'] = activityEnddf.apply(lambda x: (1.0/x['ndist']) * (x['max']/x['sum']) , axis=1)
	    # combine activityStartdf and activityEnddf
	    # outer join since some activities may only be starting ones or ending ones 
	    PFv_df = pd.merge(left = activityStartdf.drop(['ndist','sum','max'], axis=1) , 
	                        right=activityEnddf.drop(['ndist','sum','max'], axis=1), 
	                        right_on=['activityEnd'], left_on =['activityStart'], how='outer')
	    PFv_df = PFv_df.fillna(0)
	    PFv_df['PFv'] = PFv_df.apply(lambda x: x['PFvstart']+x['PFvend'] , axis=1)
	    PFv_df = PFv_df.drop(['activityEnd','PFvstart','PFvend'], axis=1)
	    PFv_df= PFv_df.rename(columns={'activityStart':'activity'})
	    PFv_df['cluster'] = clusteri
	    # append to result df
	    PFv = pd.concat([PFv, PFv_df], sort=False)
	return(PFv) 

def getIndicators(clusters, activityKey):
	##############  Execution Frequency & case ############
	print("Calculating Execution Frequency & Execution Frequency:case")
	EF_EFc = execFreq(clusters, activityKey)
	##############  Execution Time & Inverse Stability ############
	print("Calculating Execution Time & Inverse Stability")
	indicators = execTime(clusters, EF_EFc)
	##############  Prior/follow variability ############
	print("Calculating Prior/follow variability")
	PFv = priorFollowVar(clusters)
	##############  all indicators ############
	indicators =  pd.merge(left = indicators , right=PFv, right_on=['cluster','activity'], 
                      		left_on =['cluster','activity'], how='left')
	indicators = indicators.fillna(0)
	return(indicators)

def generatePlots(indicators):
	plotTitles = ['Execution Frequency', 'Execution Frequency: case','Execution Time', 
					'Inverse Stability', 'Prior Follow Variability']
	plotFilenames = ['ExecutionFrequency', 'ExecutionFreqCase', 'ExecutionTime',
					'InverseStability', 'PriorFollowVar']
	##############  plot ############
	for i,j in zip(range(2, indicators.shape[1]), range(len(plotTitles))):
		plt.style.use('seaborn-whitegrid')
		plt.figure()
		xindex = np.arange(len(indicators))
		colname = indicators.columns[i]
		indicators.sort_values(by=[colname], inplace=True)
		plt.bar(xindex.astype('U'), indicators[colname], color = 'indigo',width=1)
		plt.xticks(np.arange(0, len(indicators), 50)) 
		plt.title(plotTitles[j])
		plt.xlabel('Activities')
		plt.savefig('plots/'+plotFilenames[j]+'.png');


def main():
	## input parameter data 
	filexes = sys.argv[1:][0]
	##############  import ############
	path = ".."
	os.chdir(path)
	log = xes_import_factory.apply(os.path.join(os.getcwd(), "data", filexes))
	print("Event log loaded with number of traces:", len(log))
	##############  trace clustering ############
	# Count the arguments excluding the filename, first is necessary 
	activityKey = sys.argv[1:][1]
	if (len(sys.argv) - 2) > 1:
		clusterParams = {"str_ev_attr" : activityKey, "str_tr_attr" :sys.argv[1:][2]}
	else:
		clusterParams = {"str_ev_attr" : activityKey}
	clusters = clusterlog(log,clusterParams)
	##############  calculate indicators ############
	indicators = getIndicators(clusters, activityKey)
	print("Generating Indicator plots and saving the files")
	##############  plot ############
	generatePlots(indicators)
	



if __name__ == "__main__":
	main()

