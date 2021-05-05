import pyreadstat
import numpy as np
import pandas as pd

folder = "~/Downloads/UKDA-5050-spss/spss/spss24/"

med = []

# wave 0 1998
med.append(['medcinbp','diur','beta','aceinh','calciumb','obpdrug','medcindi','lipid'])

# wave 0 1999
med.append(['medcinbp','diur','beta','aceinh','calciumb','obpdrug','medcindi','lipid'])

# wave 0 2001
med.append(['ager','bpmedd','diur','beta','aceinh','calciumb','obpdrug','lipid'])

#wave 1
med.append(['hemda', 'hemdb', 'heama'])

#wave 2
# blood pressure, diabetes, antigoagulation, lung condition, asthma, osteoporosis, knee/hip medication or exercise
# anaticoglant, Warfarin
med.append(['Hemda','HeMdb', 'Hehrtb', 'Hehrtc','HeLng', 'HeAma', 'HeOstec', 
            'HePad', 'Hehrtb2', 'Hehrtc2', 'HeBetb', 'HeAcea'])

#wave3
# blood pressure, blood pressure returning, diabetes, blood thinning, to lower cholesterol, prevent high chol, lung
# asthma, knee/hip drug or exercise, chest pain drug
med.append(['hemda', 'hemdab','hemdb','hehrtmd', 'hechmd', 'hechme',
            'helng', 'heama', 'hepad', 'herosmd', 'heacea', 'heamb'])

#wave 4
med.append(['hemda', 'hemdab','hemdb','hehrtmd', 'hechmd', 'hechme',
            'helng', 'heama', 'herosmd', 'heacea', 'heamb',
           'hemda1', 'hekned', 'hehipb', 'hepmed'])

#wave 5
med.append(['hemda', 'hemdab','hemdb','hebetb','HeHrtMd', 'hehrtb', 'hehrtc', 
            'hehrtb2', 'hehrtc2', 'hechmd', 'hechme', 
            'helng', 'heama', 'herosmd', 'heacea', 'heamb',
           'hemda1', 'heknec','hekned', 'hehipb', 'hepmed', 'heostec', 'heostea'])

#wave 6
med.append(['HeMDa','HeMdab','HeAcea','HEMDA1','HeMdb','Hehrtb','Hehrtc',
            'HeChMd','HeChMe','HeLng','HeLngB','HeAma','HeAmb','HeKnec','HeKned',
            'HeHipB','HePMed','HeOstea','HeOstec'])

#wave 7
med.append(['HeMDa','HeMdab','HeAcea','HEMDA1','HeMdb','Hehrtb','Hehrtc',
            'HeChMd','HeChMe','HeLng','HeLngB','HeAma','HeAmb','HeOstea','HeOstec'])

#wave 8
med.append(['hemda','hemdab','heacea','hemda1','hemdb','hehrtb',
            'hehrtc','hechmd','hechme',
            'helng','helngb','heama','heamb','hepad','heostea','heostec'])


# read data
waves = [[]]
wave1998, _ = pyreadstat.read_sav(folder+"wave_0_1998_data.sav",
                                        usecols = ['idauniq','dhager'] + med[0])
waves[0].append(wave1998)


wave1999, _ = pyreadstat.read_sav(folder+"wave_0_1999_data.sav",
                                        usecols = ['idauniq','dhager'] + med[1])
waves[0].append(wave1999)

wave2001, _ = pyreadstat.read_sav(folder+"wave_0_2001_data.sav",
                                        usecols = ['idauniq','dhager'] + med[2])
waves[0].append(wave2001)


wave1, _ = pyreadstat.read_sav(folder+"wave_1_core_data_v3.sav",
                                        usecols = ['idauniq','dhager'] + med[2+1])
waves.append(wave1)


wave2, _ = pyreadstat.read_sav(folder+"wave_2_core_data_v4.sav",
               usecols = ['idauniq','dhager'] + med[2+2])
waves.append(wave2)



wave3, _ = pyreadstat.read_sav(folder+"wave_3_elsa_data_v4.sav",
                                        usecols = ['idauniq','dhager'] + med[2+3])
waves.append(wave3)


wave4, _ = pyreadstat.read_sav(folder+"wave_4_elsa_data_v3.sav",
                                        usecols = ['idauniq','dhager'] + med[2+4])
waves.append(wave4)


wave5, _ = pyreadstat.read_sav(folder+"wave_5_elsa_data_v4.sav",
                                        usecols = ['idauniq','indager'] + med[2+5])
wave5['indager'].replace(-7.0,90,inplace=True)
waves.append(wave5)


wave6, _ = pyreadstat.read_sav(folder+"wave_6_elsa_data_v2.sav",
                                        usecols = ['idauniq','indager'] + med[2+6])
waves.append(wave6)


wave7, _ = pyreadstat.read_sav(folder+"wave_7_elsa_data.sav",
                                        usecols = ['idauniq','indager'] + med[2+7])
wave7['indager'].replace(-7.0,90,inplace=True)
waves.append(wave7)

wave8, _ = pyreadstat.read_sav(folder+"wave_8_elsa_data_eul_v2.sav",
                                        usecols = ['idauniq','indager'] + med[2+8])
wave8['indager'].replace(-7.0,90,inplace=True)
waves.append(wave8)

# replace nans
indexes = [[] for w in range(len(waves))]
for y in range(3):
    waves[0][y][med[y]] = waves[0][y][med[y]].replace([-2.0, -8.0,-9.0],np.nan)
    waves[0][y][med[y]] = waves[0][y][med[y]].replace(3.0,1.0)
    waves[0][y][med[y]] = waves[0][y][med[y]].replace(2.0,0.0)
    waves[0][y][med[y]] = waves[0][y][med[y]].fillna(-1.0)

for w in np.array([1,2,3,4,5,6,7,8],dtype=int)+2:#,4,5,6,7]:
    waves[w-2][med[w]] = waves[w-2][med[w]].replace([-2.0, -8.0,-9.0],np.nan)
    waves[w-2][med[w]] = waves[w-2][med[w]].replace(3.0,1.0)
    waves[w-2][med[w]] = waves[w-2][med[w]].replace(2.0,0.0)
    waves[w-2][med[w]] = waves[w-2][med[w]].fillna(-1.0)
    indexes[w-2] = [x for x in waves[w-2].index.values]


# fix names

# wave0 1998
waves[0][0].rename(columns={'ager':'dhager',
                         'medcinbp':'blood pressure medication 1',
                         'diur':'blood pressure medication 2',
                         'beta':'blood pressure medication 3',
                         'aceinh':'blood pressure medication 4',
                         'calciumb':'blood pressure medication 5',
                         'obpdrug':'blood pressure medication 6',
                         'medcindi':'diabetes medication',
                         'lipid':'cholesterol medication 1'}, inplace=True)

# wave0 1999
waves[0][1].rename(columns={'ager':'dhager',
                         'medcinbp':'blood pressure medication 1',
                         'diur':'blood pressure medication 2',
                         'beta':'blood pressure medication 3',
                         'aceinh':'blood pressure medication 4',
                         'calciumb':'blood pressure medication 5',
                         'obpdrug':'blood pressure medication 6',
                         'medcindi':'diabetes medication',
                         'lipid':'cholesterol medication 1'}, inplace=True)

# wave0 2001
waves[0][2].rename(columns={'ager':'dhager',
                         'bpmedd':'blood pressure medication 1',
                         'diur':'blood pressure medication 2',
                         'beta':'blood pressure medication 3',
                         'aceinh':'blood pressure medication 4',
                         'calciumb':'blood pressure medication 5',
                         'obpdrug':'blood pressure medication 6',
                         'lipid':'cholesterol medication 1'}, inplace=True)


# wave 1
waves[1].rename(columns={'ager':'dhager', 
                         'hemda':'blood pressure medication 1',
                         'hemdb':'diabetes medication',
                         'helng':'lung medication 1',
                         'heama':'asthma medication 1',
                         'dobyear':'dob'}, inplace=True)

# waves 2
waves[2].rename(columns={'ager':'dhager', 
                         'Hemda':'blood pressure medication 1',
                         'HeBetb':'blood pressure medication 2',
                         'HeAcea':'blood pressure medication 3',
                         'HeMdb': 'diabetes medication',
                         'Hehrtb': 'anticoagulent 1',
                         'Hehrtc': 'anticoagulent 2',
                         'Hehrtb2': 'anticoagulent 3',
                         'Hehrtc2': 'anticoagulent 4',
                         'HeLng':'lung medication 1',
                         'HeAma':'asthma medication 1',
                         'HeOstec':'osteoporosis medication',
                         'HePad':'hip/knee treatment 1'
                         }, inplace=True)

# wave 3
waves[3].rename(columns={'ager':'dhager', 
                         'hemda':'blood pressure medication 1',
                         'hemdab':'blood pressure medication 2',
                         'heacea':'blood pressure medication 3',
                         'hehrtmd':'anticoagulent 1',
                         'hemdb':'diabetes medication',
                         'helng':'lung medication 1',
                         'helngb':'lung medication 2',
                         'heama':'asthma medication 1',
                         'heamb':'asthma medication 2',
                         'hepad':'hip/knee treatment 1',
                         'hechmd':'cholesterol medication 1',
                         'hechme':'cholesterol medication 2'
                         }, inplace=True)

# wave 4
waves[4].rename(columns={'ager':'dhager', 
                         'hemda':'blood pressure medication 1',
                         'hemdab':'blood pressure medication 2',
                         'heacea':'blood pressure medication 3',
                         'hemda1':'blood pressure medication 4',
                         'hemdb':'diabetes medication',
                         'hehrtmd':'anticoagulent 1',
                         'hechmd':'cholesterol medication 1',
                         'hechme':'cholesterol medication 2',
                         'helng':'lung medication 1',
                         'helngb':'lung medication 2',
                         'heama':'asthma medication 1',
                         'heamb':'asthma medication 2',
                         'hekned':'hip/knee treatment 1',
                         'hehipb':'hip/knee treatment 2',
                         'hepmed':'hip/knee treatment 3'
                        }, inplace=True)

# wave 5
waves[5].rename(columns={'ager':'dhager', 
                         'hemda':'blood pressure medication 1',
                         'hemdab':'blood pressure medication 2',
                         'hebetb':'blood pressure medication 3',
                         'heacea':'blood pressure medication 4',
                         'hemda1':'blood pressure medication 5', 
                         'hemdb':'diabetes medication',
                         'hehrtb':'anticoagulent 1',
                         'hehrtc':'anticoagulent 2',
                         'hehrtb2':'anticoagulent 3',
                         'hehrtc2':'anticoagulent 4',
                         'HeHrtMd':'anticoagulent 5',
                         'hechmd':'cholesterol medication 1',
                         'hechme':'cholesterol medication 2',
                         'helng':'lung medication 1',
                         'helngb':'lung medication 2',
                         'heama':'asthma medication 1',
                         'heamb':'asthma medication 2',
                         'heknec':'hip/knee treatment 1',
                         'hekned':'hip/knee treatment 2',
                         'hehipb':'hip/knee treatment 3',
                         'hepmed':'hip/knee treatment 4',
                         'heostea':'osteoporosis medication 1',
                         'heostec':'osteoporosis medication 2'
                        }, inplace=True)

# wave 6
waves[6].rename(columns={'ager':'dhager', 
                         'HeMDa':'blood pressure medication 1',
                         'HeMdab':'blood pressure medication 2',
                         'HeAcea':'blood pressure medication 3',
                         'HEMDA1':'blood pressure medication 4', 
                         'HeMdb':'diabetes medication',
                         'Hehrtb':'anticoagulent 1',
                         'Hehrtc':'anticoagulent 2',
                         'HeChMd':'cholesterol medication 1',
                         'HeChMe':'cholesterol medication 2',
                         'HeLng':'lung medication 1',
                         'HeLngB':'lung medication 2',
                         'HeAma':'asthma medication 1',
                         'HeAmb':'asthma medication 2',
                         'HeKnec':'hip/knee treatment 1',
                         'HeKned':'hip/knee treatment 2',
                         'HeHipB':'hip/knee treatment 3',
                         'HePMed':'hip/knee treatment 4',
                         'HeOstea':'osteoporosis medication 1',
                         'HeOstec':'osteoporosis medication 2'
                        }, inplace=True)

# wave 7
waves[7].rename(columns={'ager':'dhager', 
                         'HeMDa':'blood pressure medication 1',
                         'HeMdab':'blood pressure medication 2',
                         'HeAcea':'blood pressure medication 3',
                         'HEMDA1':'blood pressure medication 4', 
                         'HeMdb':'diabetes medication',
                         'Hehrtb':'anticoagulent 1',
                         'Hehrtc':'anticoagulent 2',
                         'HeChMd':'cholesterol medication 1',
                         'HeChMe':'cholesterol medication 2',
                         'HeLng':'lung medication 1',
                         'HeLngB':'lung medication 2',
                         'HeAma':'asthma medication 1',
                         'HeAmb':'asthma medication 2',
                         'HeOstea':'osteoporosis medication 1',
                         'HeOstec':'osteoporosis medication 2'
                        }, inplace=True)

# wave 8
waves[8].rename(columns={'ager':'dhager', 
                         'hemda':'blood pressure medication 1',
                         'hemdab':'blood pressure medication 2',
                         'hebetb':'blood pressure medication 3',
                         'heacea':'blood pressure medication 4',
                         'hemda1':'blood pressure medication 5', 
                         'hemdb':'diabetes medication',
                         'hehrtb':'anticoagulent 1',
                         'hehrtc':'anticoagulent 2',
                         'hechmd':'cholesterol medication 1',
                         'hechme':'cholesterol medication 2',
                         'helng':'lung medication 1',
                         'helngb':'lung medication 2',
                         'heama':'asthma medication 1',
                         'heamb':'asthma medication 2',
                         'hepad':'hip/knee treatment 1',
                         'heostea':'osteoporosis medication 1',
                         'heostec':'osteoporosis medication 2'
                        }, inplace=True)

waves[0] = pd.concat([waves[0][0],waves[0][1],waves[0][2]], sort=False)
indexes[0] = [x for x in waves[0].index.values]

# combine all waves
data = pd.concat(waves, 
                 keys=[0,1,2,3,4,5,6,7,8], sort=False)
data.fillna(-1.0, inplace=True)


# determine if taking any medicines by combining same types
def taking_any(row,labels):
    
    x = -1.0
    if np.any(row[labels] == 1.0) or np.any(row[labels] == 3.0):
        x = 1.0
    elif np.any(row[labels] == 2.0) or np.any(row[labels] == 0.0):
        x = 0.0
    else:
        x = -1.0
    return x

data['BP med'] = data.apply(lambda row: taking_any(row,['blood pressure medication %d'%i \
                                                  for i in range(1,7)]), axis=1)

data['anticoagulent med'] = data.apply(lambda row: taking_any(row,['anticoagulent %d'%i \
                                                  for i in range(1,6)]), axis=1)

data['chol med'] = data.apply(lambda row: taking_any(row,['cholesterol medication 1', 
                                                    'cholesterol medication 2']), axis=1)

data['hip/knee treat'] = data.apply(lambda row: taking_any(row,['hip/knee treatment %d'%i \
                                                  for i in range(1,5)]),axis=1)

data['lung/asthma med'] = data.apply(lambda row: taking_any(row,['lung medication 1', 
                                                                'lung medication 2',
                                                                'asthma medication 1',
                                                                'asthma medication 2']),axis=1)

data[['idauniq','dhager','BP med', 'anticoagulent med', 'chol med', 'hip/knee treat', 'lung/asthma med']] = data[['idauniq','dhager','BP med', 'anticoagulent med', 'chol med', 'hip/knee treat','lung/asthma med']].replace(2.0,0.0)

# output
for w in range(9): 
    data.xs(w,level=0)[['idauniq','BP med', 'anticoagulent med', 'chol med', 'hip/knee treat','lung/asthma med']].to_csv('../Data/ELSA_Med_cleaned_wave'+str(w)+'.csv',index=False)
