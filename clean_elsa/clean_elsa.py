from pyreadstat import read_sav
import numpy as np
import pandas as pd


"""
Creates main elsa data
"""




#Read files. Column names vary wildly per file, so these need to be selected manually.
folder = "~/Downloads/UKDA-5050-spss/spss/spss24/"

files = ["wave_0_1998_data.sav", "wave_0_1999_data.sav", "wave_0_2001_data.sav", 
        "wave_1_core_data_v3.sav", "wave_2_core_data_v4.sav", "wave_2_nurse_data_v2.sav",
        "wave_3_elsa_data_v4.sav", "wave_4_elsa_data_v3.sav", "wave_4_nurse_data.sav",
        "wave_5_elsa_data_v4.sav",   
         "wave_6_elsa_data_v2.sav", "wave_6_elsa_nurse_data_v2.sav","wave_7_elsa_data.sav", 
         "wave_8_elsa_data_eul_v2.sav", "wave_8_elsa_nurse_data_eul_v1.sav"]

cols = []

# wave0 1998
cols.append(['idauniq', 'ager','sex','dobyear','sys1','sys2','sys3','dias1','dias2','dias3',
                   'crpval','hdlval', 'glucval','haemval','fibval','ferval', 'bmi', 'pulse1',
            'pulse2','pulse3', 'docinfo1', 'ethnicr', 'htval', 'genhelf', 'longill', 'limitact',
             'smkevr', 'cignow'])

# wave0 1999
cols.append(['idauniq', 'ager','sex','dobyear','sys1','sys2','sys3','dias1','dias2','dias3',
                   'ftrigval','crpval','hdlval','fldlval','glucval','haemval','fibval','ferval','bmi', 'cholval',
            'tglyval', 'pulse1','pulse2','pulse3','docinfo1', 'ethnicr','htval', 'genhelf',
            'longill', 'limitact', 'smkevr', 'cignow'])

# wave0 2001
cols.append(['idauniq', 'ager','sex','dobyear','sys1','sys2','sys3','dias1','dias2','dias3',
                   'glucval','bmi','pulse1','pulse2','pulse3', 'ethnicr','htval', 
                    'genhelf', 'longill', 'limitact', 'smkevr', 'cignow']) #'illsm1'

# wave1
cols.append(['idauniq','dhager','dhsex','mmwlka','mmwlkb','dhdobyr', 'mmschs', *['hedia%02d'%i for i in range(1,11)],
             'fqethnr', 'hehelf','pscedb', 'heill', 'helim','heacta', 'heactb', 'heactc', 
             'hesmk', 'heska', 'heeye', 'hehear', 'hefunc', 'heala', 'fqcbthr', 'hefrac', 
             'heji'])

# wave2
cols.append(['idauniq','indsex','MMWlkA','MMWlkB','dhdobyr', 'MmSchs','HeACd', 'fqethnr',
             'Hehelf', 'PScedB', 'Heill', 'Helim','heActa', 'heActb', 'heActc', 
             'HeSmk', 'HESka', 'Heeye', 'Hehear', 'HeFunc', 'scako', 'fqcbthr',
             'HeFrac', 'HeJi'])

# wave2 nurse
cols.append(['idauniq','confage','mmgsd1','mmgsn1','mmgsd2','mmgsn2','mmgsd3','mmgsn3',
              'sys1','dias1','sys2','dias2','sys3','dias3','trig','hscrp','hdl',
                'ldl','fglu','hgb','cfib','rtin','bmi', 'chol', 'apoe','hba1c','pulse1','pulse2','pulse3',
            'mmrrfti', 'htval', 'mmloti', 'mmftti'])

# wave3 
cols.append(['idauniq','dhager','dhsex','mmwlka','mmwlkb','dhdobyr', 'mmschs','dheacd', 'fqethnr',
            'hegenh', 'pscedb', 'heill', 'helim', 'heacta', 'heactb', 'heactc', 
             'hesmk', 'heska', 'heeye','hehear','hefunc', 'scako', 'fqcbthr', 'hefrac',
             'heji'])

# wave4
cols.append(['idauniq','dhager','dhsex','mmwlka','mmwlkb', 'mmschs', 'heacd', 'fqethnr',
            'hehelf', 'pscedb', 'heill', 'helim', 'heacta', 'heactb', 'heactc',
             'hesmk', 'heska', 'heeye', 'hehear', 'hefunc', 'scako', 'fqcbthr', 'hefrac',
             'heji'])

# wave4 nurse
cols.append(['idauniq','confage','mmgsd1','mmgsn1','mmgsd2','mmgsn2','mmgsd3','mmgsn3',
            'sys1','dias1','sys2','dias2','sys3','dias3','dobyear','trig','hscrp',
             'hdl','ldl','igf1','fglu','hgb','cfib','rtin','bmi','chol','dheas', 'hba1c', 'wbc', 'mch',
            'pulse1','pulse2','pulse3','mmrrfti', 'htval', 'mmloti', 'mmftti'])

# wave5
cols.append(['idauniq','indager','dhsex','indobyr','mmwlka','mmwlkb', 'mmschs', 'heacd', 'fqethnr',
            'hehelf', 'pscedb', 'heill', 'helim', 'heacta', 'heactb', 'heactc',
             'hesmk', 'heska', 'heeye', 'hehear', 'HeFunc', 'scako', 'fqcbthr', 'hefrac',
             'heji'])

# wave6
cols.append(['idauniq','indager','DhSex','MMWlkA','MMWlkB','Indobyr', 'MmSchs','HeACd', 'Fqethnr',
            'Hehelf', 'PScedB', 'Heill', 'Helim', 'heActa', 'heActb', 'heActc',
             'HeSmk', 'HESka', 'HeEye', 'HeHear', 'HeFunc', 'scako', 'Fqcbthr',
             'HeFrac', 'HeJi'])

# wave6 nurse
cols.append(['idauniq','mmgsd1','mmgsn1','mmgsd2','mmgsn2','mmgsd3','mmgsn3',
              'sys1','dias1','sys2','dias2','sys3','dias3','trig','hscrp',
               'hdl','ldl','igf1','fglu','hgb','cfib','rtin','BMI','wbc','mch','hba1c','VITD','chol',
            'pulse1','pulse2','pulse3','mmrrfti','HTVAL', 'mmloti', 'mmftti'])

# wave7
cols.append(['idauniq','indager','DhSex','MMWlkA','MMWlkB','Indobyr', 'MmSchs','HeACd', 'Fqethnr',
            'Hehelf', 'PScedB', 'Heill', 'Helim','heActa', 'heActb', 'heActc',
             'HeSmk', 'HESka', 'HeEye', 'HeHear','HeFunc', 'scako', 'Fqcbthr', 'HeFrac',
             'HeJi'])

# wave8
cols.append(['idauniq','indager','mmwlka','mmwlkb','indobyr', 'mmschs', 'heacd', 'fqethnmr',
            'hehelf', 'pscedb', 'heill', 'helim','heacta', 'heactb', 'heactc',
             'hesmk', 'heska', 'heeye', 'hehear', 'hefunc', 'scako', 'fqcbthmr', 'hefrac',
             'heji'])

# wave8 nurse
cols.append(['idauniq','indsex','mmgsd1','mmgsn1','mmgsd2','mmgsn2','mmgsd3','mmgsn3',
              'sys1','dias1','sys2','dias2','sys3','dias3','trig','hscrp',
              'hdl','ldl','igf1','fglu','hgb','cfib','rtin','wbc','mch','hba1c','vitd','chol',
            'pulse1','pulse2','pulse3'])

# wave9
cols.append(['idauniq','indager','mmwlka','mmwlkb','indobyr', 'mmschs', 'heacd', 'fqethnmr',
            'hehelf', 'pscedb', 'heill', 'helim','heacta', 'heactb', 'heactc',
             'hesmk', 'heska', 'heeye', 'hehear', 'hefunc', 'scalcm', 'fqcbthmr', 
             'hefrac', 'heji'])

keys = ['wave0_1998','wave0_1999', 'wave0_2001', 'wave1', 'wave2', 'wave2_nurse',
       'wave3','wave4', 'wave4_nurse', 'wave5', 'wave6', 'wave6_nurse', 'wave7',
       'wave8','wave8_nurse']

waves = dict()
for i,f in enumerate(files):
    waves[keys[i]] = read_sav(folder+f, usecols=cols[i])[0]
    
# fixing birth years
#Individuals of age 90+ have their actual age and birth year censored due to privacy reasons. We approximate this missing birthyear with, birthyear = (approximate year of interview) - 90.
waves['wave0_1998']['dobyear'].replace(-7.0,1908.0, inplace=True)
waves['wave0_1999']['dobyear'].replace(-7.0,1909.0, inplace=True)
waves['wave0_2001']['dobyear'].replace(-7.0,1911.0, inplace=True)
waves['wave1']['dhdobyr'].replace(-7.0,1912.0,inplace=True)
waves['wave2']['dhdobyr'].replace(-7.0,1914.0,inplace=True)
waves['wave3']['dhdobyr'].replace(-7.0,1916.0,inplace=True)
waves['wave4_nurse']['dobyear'].replace(-7.0,1918.0,inplace=True)
waves['wave5']['indobyr'].replace(-7.0,1920.0,inplace=True)
waves['wave5']['indager'].replace(-7.0,90,inplace=True)
waves['wave6']['Indobyr'].replace(-7.0,1922.0,inplace=True)
waves['wave7']['Indobyr'].replace(-7.0,1924.0,inplace=True)
waves['wave7']['indager'].replace(-7.0,90,inplace=True)
waves['wave8']['indobyr'].replace(-7.0,1926.0,inplace=True)
waves['wave8']['indager'].replace(-7.0,90,inplace=True)

# some hba1c values are in mmol/mol instead of % values. Convert them all to %
# this information is available in the documentation provided by ELSA
waves['wave6_nurse']['hba1c'].fillna(-1.0, inplace=True)
waves['wave6_nurse']['hba1c'] = waves['wave6_nurse']['hba1c'].apply(lambda x: (x/10.929)+2.15 if x >= 0 else x)
waves['wave8_nurse']['hba1c'].fillna(-1.0, inplace=True)
waves['wave8_nurse']['hba1c'] = waves['wave8_nurse']['hba1c'].apply(lambda x: (x/10.929)+2.15 if x >= 0 else x)

# ADL and IADL scores were created in a separate file and input here for simplicity
waves_FI = dict()
for i in [1,2,3,4,6,7,8]:
    waves_FI['wave'+str(i)] = pd.read_csv('../Data/ELSA_Frailty_cleaned_wave'+str(i)+'.csv')
    waves_FI['wave'+str(i)].fillna(-1.0, inplace=True)

# Medication data was also done separately
waves_med = dict()
for i in range(9):
    waves_med['wave'+str(i)] = pd.read_csv('../Data/ELSA_Med_cleaned_wave'+str(i)+'.csv')

# end of life data
eol2 = read_sav(folder+"elsa_eol_w2_archive_v1.sav", usecols=['idauniq','EiDateY'])[0]
eol2['idauniq'] = eol2['idauniq'].astype(int)
eol2 = eol2.set_index('idauniq')

eol3 = read_sav(folder+"elsa_eol_w3_archive_v1.sav",usecols=['idauniq','EiDateY'])[0]
eol3['idauniq'] = eol3['idauniq'].astype(int)
eol3 = eol3.set_index('idauniq')

eol4 = read_sav(folder+"elsa_eol_w4_archive_v1.sav",usecols=['idauniq','EiDateY'])[0]
eol4['idauniq'] = eol4['idauniq'].astype(int)
eol4 = eol4.set_index('idauniq')

eol6 = read_sav(folder+"elsa_endoflife_w6archive.sav",usecols=['idauniq','EiDateY'])[0]
eol6['idauniq'] = eol6['idauniq'].astype(int)
eol6 = eol6.set_index('idauniq')

# Now the data for each wave is consolidated and individual indexes set.

for key in keys:
    waves[key]['idauniq'] = waves[key]['idauniq'].astype(int)
    waves[key] = waves[key].set_index('idauniq')

for key in waves_FI.keys():
    waves_FI[key]['idauniq'] = waves_FI[key]['idauniq'].astype(int)
    waves_FI[key] = waves_FI[key].set_index('idauniq')
    
for key in waves_med.keys():
    waves_med[key]['idauniq'] = waves_med[key]['idauniq'].astype(int)
    waves_med[key] = waves_med[key].set_index('idauniq')

combined_waves = dict()
combined_waves['wave0'] = pd.concat([waves['wave0_1998'],
                                     waves['wave0_1999'],
                                     waves['wave0_2001']], 
                                    sort=False)

combined_waves['wave0'] = pd.concat([combined_waves['wave0'],
                                    waves_med['wave0']], 
                                    axis = 1, sort=False)


combined_waves['wave1'] = pd.concat([waves['wave1'], 
                                     waves_FI['wave1'],
                                    waves_med['wave1']], 
                                    axis = 1, sort=False)

combined_waves['wave2'] = pd.concat([waves['wave2_nurse'], 
                                     waves['wave2'], 
                                     waves_FI['wave2'],
                                     waves_med['wave2'],
                                     eol2], axis=1, sort=False)

combined_waves['wave3'] = pd.concat([waves['wave3'], 
                                     waves_FI['wave3'], 
                                     waves_med['wave3'],
                                     eol3], axis=1, sort=False)

combined_waves['wave4'] = pd.concat([waves['wave4_nurse'], 
                                     waves['wave4'], 
                                     waves_FI['wave4'], 
                                     waves_med['wave4'],
                                     eol4], axis=1, sort=False)

combined_waves['wave5'] = pd.concat([waves['wave5'], 
                                    waves_med['wave5']],
                                    axis = 1, sort=False)

combined_waves['wave6'] = pd.concat([waves['wave6_nurse'], 
                                     waves['wave6'], 
                                     waves_FI['wave6'], 
                                     waves_med['wave6'],
                                     eol6], axis=1, sort=False)

combined_waves['wave7'] = pd.concat([waves['wave7'], 
                                     waves_FI['wave7'],
                                    waves_med['wave7']],
                                    axis = 1, sort=False)

combined_waves['wave8'] = pd.concat([waves['wave8_nurse'], 
                                     waves['wave8'], 
                                     waves_FI['wave8'],
                                    waves_med['wave8']], 
                                    axis=1, sort=False)

combined_waves['wave8'] = pd.concat([waves['wave8_nurse'], 
                                     waves['wave8'], 
                                     waves_FI['wave8'],
                                    waves_med['wave8']], 
                                    axis=1, sort=False)

# make varaible names consistent across all waves
combined_waves['wave0'].rename(columns={'ager':'age','dobyear':'dob', 'ftrigval':'trig', 'crpval':'crp',
                      'hdlval':'hdl','fldlval':'ldl','glucval':'glucose','haemval':'hgb',
                     'fibval':'fib','ferval':'fer', 'cholval':'chol', 'docinfo1':'diabetes', 'ethnicr':'ethnicity', 
                                       'tglyval':'hba1c', 'htval':'height', 'genhelf':'srh','cignow':'smknow'
                                       }, inplace=True)

combined_waves['wave1'].rename(columns={'dhager':'age', 'dhsex':'sex','mmwlka':'walka',
                      'mmwlkb':'walkb','dhdobyr':'dob','fqethnr':'ethnicity', 'mmschs': 'mobility',
                       'hehelf':'srh', 'pscedb':'effort','heill':'longill', 
                       'helim':'limitact', 'hesmk':'smkevr', 
                       'heska':'smknow', 
                       'heeye':'eye', 'hehear':'hear', 'hefunc':'func',
                       'heala':'alcohol', 'fqcbthr':'country', 'hefrac':'fractures',
                                        'heji':'jointrep'}, inplace=True)


combined_waves['wave2'].rename(columns={'confage':'age','indsex':'sex','mmgsd1':'grip_dom_1','mmgsd2':'grip_dom_2','mmgsd3':'grip_dom_3', 
                      'mmgsn1':'grip_ndom_1','mmgsn2':'grip_ndom_2','mmgsn3':'grip_ndom_3',
                      'MMWlkA':'walka','MMWlkB':'walkb','dhdobyr':'dob',
                     'EiDateY':'dod','hscrp':'crp','fglu':'glucose',
                     'cfib':'fib','rtin':'fer', 'HeACd':'diabetes', 'fqethnr':'ethnicity',
                                       'mmrrfti':'chair', 'MmSchs': 'mobility', 'htval':'height',
                                       'mmftti':'full tandem', 'mmloti': 'leg raise',
                                       'Hehelf':'srh', 'PScedB':'effort','Heill':'longill','Helim':'limitact',
                                       'HeSmk':'smkevr', 'HESka':'smknow', 'Heeye':'eye', 
                                        'Hehear':'hear', 'HeFunc':'func',
                       'scako':'alcohol', 'fqcbthr':'country', 'HeFrac':'fractures',
                            'HeJi':'jointrep'}, inplace=True)

combined_waves['wave3'].rename(columns={'dhager':'age','dhsex':'sex','mmwlka':'walka',
                       'mmwlkb':'walkb','dhdobyr':'dob',
                     'EiDateY':'dod', 'dheacd':'diabetes', 'fqethnr':'ethnicity', 
                                        'mmschs': 'mobility', 
                                        'hegenh':'srh', 'pscedb':'effort','heill':'longill',
                                       'helim':'limitact', 'hesmk':'smkevr', 'heska':'smknow', 
                                        'heeye':'eye', 'hehear':'hear', 'hefunc':'func',
                       'scako':'alcohol', 'fqcbthr':'country', 'hefrac':'fractures',
                            'heji':'jointrep'}, inplace=True)

combined_waves['wave4'].rename(columns={'confage':'age', 'dhsex':'sex','mmgsd1':'grip_dom_1','mmgsd2':'grip_dom_2','mmgsd3':'grip_dom_3', 
                      'mmgsn1':'grip_ndom_1','mmgsn2':'grip_ndom_2','mmgsn3':'grip_ndom_3',
                      'mmwlka':'walka','mmwlkb':'walkb','dobyear':'dob',
                      'EiDateY':'dod','hscrp':'crp','fglu':'glucose',
                     'cfib':'fib','rtin':'fer','heacd':'diabetes', 'fqethnr':'ethnicity',
                     'mmrrfti':'chair', 'mmschs': 'mobility', 'htval':'height',
                     'mmftti':'full tandem', 'mmloti': 'leg raise',
                                       'hehelf':'srh', 'pscedb':'effort','heill':'longill',
                                       'helim':'limitact', 'hesmk':'smkevr', 'heska':'smknow', 
                                        'heeye':'eye', 'hehear':'hear', 'hefunc':'func',
                       'scako':'alcohol', 'fqcbthr':'country', 'hefrac':'fractures',
                       'heji':'jointrep'}, inplace=True)

combined_waves['wave5'].rename(columns={'indager':'age','dhsex':'sex','mmwlka':'walka',
                       'mmwlkb':'walkb','indobyr':'dob', 'heacd':'diabetes',
                     'EiDateY':'dod', 'fqethnr':'ethnicity', 'mmschs': 'mobility',
                                       'hehelf':'srh', 'pscedb':'effort','heill':'longill',
                                       'helim':'limitact',
                                       'hesmk':'smkevr', 'heska':'smknow', 'heeye':'eye', 
                                        'hehear':'hear', 'HeFunc':'func',
                       'scako':'alcohol', 'fqcbthr':'country', 'hefrac':'fractures',
                       'heji':'jointrep'}, inplace=True)

combined_waves['wave6'].rename(columns={'indager':'age', 'DhSex':'sex','Sex':'sex','mmgsd1':'grip_dom_1','mmgsd2':'grip_dom_2','mmgsd3':'grip_dom_3', 
                      'mmgsn1':'grip_ndom_1','mmgsn2':'grip_ndom_2','mmgsn3':'grip_ndom_3',
                      'MMWlkA':'walka','MMWlkB':'walkb','Indobyr':'dob',
                      'EiDateY':'dod','hscrp':'crp','fglu':'glucose',
                     'cfib':'fib','rtin':'fer','BMI':'bmi','VITD':'vitd', 'HeACd':'diabetes', 
                                       'Fqethnr':'ethnicity', 'mmrrfti':'chair', 'MmSchs': 'mobility',
                                       'HTVAL':'height', 'mmftti':'full tandem', 'mmloti': 'leg raise',
                                       'Hehelf':'srh', 'PScedB':'effort','Heill':'longill',
                                       'Helim':'limitact',
                                       'HeSmk':'smkevr', 'HESka':'smknow', 'HeEye':'eye', 
                                        'HeHear':'hear', 'HeFunc':'func',
                       'scako':'alcohol', 'Fqcbthr':'country', 'HeFrac':'fractures',
                       'HeJi':'jointrep'}, inplace=True)

combined_waves['wave7'].rename(columns={'indager':'age','DhSex':'sex','MMWlkA':'walka',
                       'MMWlkB':'walkb','Indobyr':'dob', 'HeACd':'diabetes', 'Fqethnr':'ethnicity',
                                       'MmSchs': 'mobility',
                                       'Hehelf':'srh', 'PScedB':'effort','Heill':'longill',
                                       'Helim':'limitact',
                                       'HeSmk':'smkevr', 'HESka':'smknow', 'HeEye':'eye', 
                                        'HeHear':'hear', 'HeFunc':'func',
                       'scako':'alcohol', 'Fqcbthr':'country', 'HeFrac':'fractures',
                       'HeJi':'jointrep'}, inplace=True)

combined_waves['wave8'].rename(columns={'indager':'age', 'indsex':'sex','mmgsd1':'grip_dom_1','mmgsd2':'grip_dom_2','mmgsd3':'grip_dom_3', 
                      'mmgsn1':'grip_ndom_1','mmgsn2':'grip_ndom_2','mmgsn3':'grip_ndom_3',
                      'mmwlka':'walka','mmwlkb':'walkb','indobyr':'dob','hscrp':'crp','fglu':'glucose',
                     'cfib':'fib','rtin':'fer', 'heacd':'diabetes', 'fqethnmr':'ethnicity',
                                       'mmschs': 'mobility',
                                       'hehelf':'srh', 'pscedb':'effort','heill':'longill',
                                       'helim':'limitact', 'hesmk':'smkevr', 'heska':'smknow', 
                                        'heeye':'eye', 'hehear':'hear', 'hefunc':'func',
                       'scako':'alcohol', 'fqcbthmr':'country', 'hefrac':'fractures',
                       'heji':'jointrep'}, inplace=True)


# put all waves together to create 1 big dataframe, with a multiindex for wave and index
indexes = [x for x in combined_waves['wave0'].index.values]
data = pd.concat(list(combined_waves.values()), keys=[0,1,2,3,4,5,6,7,8], sort=False)

# some variables have specific values codifying special cases, this is dealt with here
# additionally, we collapse all distinct types of missingness into the same with code -1.
data['dod'].fillna(-1,inplace=True)
data['dod'] = data['dod'].astype(int)

data = data[(data['dob'] != -9.0)] #drop refusal
data = data[(data['dob'] != -8.0)] #drop don't know
data['dob'].fillna(-1,inplace=True)
data['dob'] = data['dob'].astype(int)

data['age'].replace(99.0,90,inplace=True) #collapsed at 90
data['age'].fillna(-1,inplace=True)
data['age'] = data['age'].astype(int)
data = data[(data['age'] != -9.0)] #drop refusal
data = data[(data['age'] != -8.0)] #drop don't know

data['walka'].fillna(-1.0,inplace=True)
data['walkb'].fillna(-1.0,inplace=True)

data['dias1'].replace(999.0,-1,inplace=True)
data['dias2'].replace(999.0,-1,inplace=True)
data['dias3'].replace(999.0,-1,inplace=True)
data['dias1'].fillna(-1,inplace=True)
data['dias2'].fillna(-1,inplace=True)
data['dias3'].fillna(-1,inplace=True)

data['sys1'].replace(999.0,-1,inplace=True)
data['sys2'].replace(999.0,-1,inplace=True)
data['sys3'].replace(999.0,-1,inplace=True)
data['sys1'].fillna(-1,inplace=True)
data['sys2'].fillna(-1,inplace=True)
data['sys3'].fillna(-1,inplace=True)

data['grip_dom_1'].replace(99.0,-1,inplace=True)
data['grip_dom_2'].replace(99.0,-1,inplace=True)
data['grip_dom_3'].replace(99.0,-1,inplace=True)
data['grip_dom_1'].fillna(-1,inplace=True)
data['grip_dom_2'].fillna(-1,inplace=True)
data['grip_dom_3'].fillna(-1,inplace=True)

data['grip_ndom_1'].replace(99.0,-1,inplace=True)
data['grip_ndom_2'].replace(99.0,-1,inplace=True)
data['grip_ndom_3'].replace(99.0,-1,inplace=True)
data['grip_ndom_1'].fillna(-1,inplace=True)
data['grip_ndom_2'].fillna(-1,inplace=True)
data['grip_ndom_3'].fillna(-1,inplace=True)

data['trig'].replace(9998.0,-1,inplace=True)
data['trig'].replace(9999.0,-1,inplace=True)
data['trig'].fillna(-1,inplace=True)

data['crp'].replace(99998.0,-1,inplace=True)
data['crp'].replace(99999.0,-1,inplace=True)
data['crp'].fillna(-1,inplace=True)

data['hdl'].fillna(-1,inplace=True)

data['ldl'].fillna(-1,inplace=True)

data['igf1'].fillna(-1,inplace=True)

data['glucose'].fillna(-1,inplace=True)

data['hgb'].fillna(-1,inplace=True)

data['fib'].fillna(-1,inplace=True)

data['fer'].fillna(-1,inplace=True)

data['bmi'].fillna(-1,inplace=True)

data['FI ADL'].fillna(-1, inplace=True)
data['FI IADL'].fillna(-1, inplace=True)
data['ADL count'].fillna(-1, inplace=True)
data['IADL count'].fillna(-1, inplace=True)

data['sex'].fillna(-1, inplace=True)
data['sex'].replace(1.0, 0.0, inplace=True)
data['sex'].replace(2.0, 1.0, inplace=True)

data['wbc'].fillna(-1,inplace=True)
data['wbc'].replace([-11.0,-8.0,-7.0,-6.0, -2.0],-1.0,inplace=True)

data['mch'].fillna(-1,inplace=True)
data['mch'].replace([-11.0,-8.0,-7.0,-6.0, -2.0],-1.0,inplace=True)

data['vitd'].fillna(-1,inplace=True)
data['vitd'].replace([-6.0,-2.0],-1.0,inplace=True)

data['dheas'].fillna(-1,inplace=True)
data['dheas'].replace([-11.0,-6.0,-2.0],-1.0,inplace=True)

data['hba1c'].fillna(-1,inplace=True)
data['hba1c'].replace([-11.0,-8.0,-7.0,-6.0,-3.0,-2.0],-1.0,inplace=True)

data['apoe'].fillna(-1,inplace=True)
data['apoe'].replace([-11.0,-7.0,-6.0,-2.0],-1.0,inplace=True)

data['chol'].fillna(-1,inplace=True)
data['chol'].replace([-11.0,-7.0,-6.0,-3.0,-2.0],-1.0,inplace=True)

data['pulse1'].fillna(-1.0,inplace=True)
data['pulse2'].fillna(-1.0,inplace=True)
data['pulse3'].fillna(-1.0,inplace=True)

data['ethnicity'].fillna(-1.0, inplace=True)
data['ethnicity'].replace(-8.0, 2.0, inplace = True)
data['ethnicity'] = data['ethnicity'].apply(lambda x: -1 if x < -1.0 else x)
data['ethnicity'].replace(1.0, 0.0, inplace=True) #white -> 0
data['ethnicity'].replace(2.0, 1.0, inplace=True) #non-white -> 0

data['chair'].fillna(-1.0, inplace=True)

data['mobility'].fillna(-1.0, inplace=True)
data['mobility']= data['mobility'].apply(lambda x: -1.0 if x < -1 else x)

data['height'].fillna(-1.0, inplace=True)
data['height'] = data['height'].apply(lambda x: -1.0 if x < -1.0 else x)

data['leg raise'].fillna(-1.0, inplace=True)
data['leg raise'] = data['leg raise'].apply(lambda x: -1.0 if x < -1.0 else x)

data['full tandem'].fillna(-1.0, inplace=True)
data['full tandem'] = data['full tandem'].apply(lambda x: -1.0 if x < -1.0 else x)

data['longill'].fillna(-1.0, inplace=True)
data['longill'] = data['longill'].apply(lambda x: -1 if x < -1.0 else x)
data['longill'].replace(2.0, 0.0, inplace=True) #no -> 0

data['srh'].fillna(-1.0, inplace=True)
data['srh'] = data['srh'].apply(lambda x: -1 if x < -1.0 else x)

data['effort'].fillna(-1.0, inplace=True)
data['effort'] = data['effort'].apply(lambda x: -1 if x < -1.0 else x)
data['effort'].replace(2.0, 0.0, inplace=True) #no -> 0

data['limitact'].fillna(-1.0, inplace=True)
data['limitact'] = data['limitact'].apply(lambda x: -1 if x < -1.0 else x)
data['limitact'].replace(2.0, 0.0, inplace=True) #no -> 0

data['smkevr'].fillna(-1.0, inplace=True)
data['smkevr'] = data['smkevr'].apply(lambda x: -1 if x < -1.0 else x)
data['smkevr'].replace(2.0, 0.0, inplace=True) #no -> 0

data['smknow'].fillna(-1.0, inplace=True)
data['smknow'] = data['smknow'].apply(lambda x: -1 if x < -1.0 else x)
data['smknow'].replace(2.0, 0.0, inplace=True) #no -> 0

data['alcohol'].fillna(-1.0, inplace=True)
data['alcohol'] = data['alcohol'].apply(lambda x: -1 if x < -1.0 else x)
data['alcohol'].replace(2.0, 0.0, inplace=True) #no -> 0

data['jointrep'].fillna(-1.0, inplace=True)
data['jointrep'] = data['jointrep'].apply(lambda x: -1 if x < -1.0 else x)
data['jointrep'].replace(2.0, 0.0, inplace=True) #no -> 0

data['country'].fillna(-1.0, inplace=True)
data['country'] = data['country'].apply(lambda x: -1 if x < -1.0 else x)
data['country'].replace(1.0, 0.0, inplace=True) #white -> 0
data['country'].replace(2.0, 1.0, inplace=True) #non-white -> 1

data['eye'].fillna(-1.0, inplace=True)
data['eye'] = data['eye'].apply(lambda x: -1 if x < -1.0 else x)

data['hear'].fillna(-1.0, inplace=True)
data['hear'] = data['hear'].apply(lambda x: -1 if x < -1.0 else x)

data['func'].fillna(-1.0, inplace=True)
data['func'] = data['func'].apply(lambda x: -1 if x < -1.0 else x)

data.loc[(data['smkevr'] == 0.0) & (data['smknow'] != 1.0), 'smknow'] = 0.0

"""
Now fix other aspects of the missing data. The multiple attempts of walking speed, grip strength, and blood pressure measurements are averaged to get a single value (and less noisy). 
Walking speed is measured in seconds to walk 8 feet, so the speed is computed with a distance of 8ft = 2.438m.
Hemoglobin is usually measured in g/dL, sometimes there are measurements in g/L. These are detected by selecting values that are too large to be in g/dL (>= 30 g/dL).
Discrete variables are converted to within [0,1]. This is done for self-rated health, hearing difficulty, eyesight difficulty, and walking ability.
"""

data['mobility'] = data['mobility'].apply(lambda x: 3 if x == 4 else x)
data['mobility'] = data['mobility'].apply(lambda x: -1.0 if x == 5 else x)

def average(row,labels):
    count = 0
    sum = 0.
    for l in labels:
        if row[l] >= 0:
            sum += row[l]
            count += 1
    if count > 0:
        return sum/count
    else:
        return -1.0

# convert from 
data['gait speed'] = 2.438/data.apply(lambda row: average(row,['walka','walkb']), axis=1)
data['gait speed'] = data['gait speed'].apply(lambda x: -1.0 if x < 0 else x)
data['grip dom'] = data.apply(lambda row: average(row,['grip_dom_1','grip_dom_2','grip_dom_3']), axis=1)
data['grip ndom'] = data.apply(lambda row: average(row,['grip_ndom_1','grip_ndom_2','grip_ndom_3']), axis=1)
data['dias'] = data.apply(lambda row: average(row,['dias1','dias2','dias3']), axis=1)
data['sys'] = data.apply(lambda row: average(row,['sys1','sys2','sys3']), axis=1)
data['trig'] = data['trig'].apply(lambda x: -1.0 if x < 0 else x)
data['crp'] = data['crp'].apply(lambda x: -1.0 if x < 0 else x)
data['hdl'] = data['hdl'].apply(lambda x: -1.0 if x < 0 else x)
data['ldl'] = data['ldl'].apply(lambda x: -1.0 if x < 0 else x)
data['igf1'] = data['igf1'].apply(lambda x: -1.0 if x < 0 else x)
data['glucose'] = data['glucose'].apply(lambda x: -1.0 if x < 0 else x)
data['hgb'] = data['hgb'].apply(lambda x: -1.0 if x < 0 else x)
data['hgb'] = data['hgb'].apply(lambda x: x/10 if x > 30 else x) # fix units for hgb
data['fib'] = data['fib'].apply(lambda x: -1.0 if x < 0 else x)
data['fer'] = data['fer'].apply(lambda x: -1.0 if x < 0 else x)
data['bmi'] = data['bmi'].apply(lambda x: -1.0 if x < 8 else x)
data['pulse1'] = data['pulse1'].apply(lambda x: -1 if x > 200 else x)
data['pulse2'] = data['pulse2'].apply(lambda x: -1 if x > 200 else x)
data['pulse3'] = data['pulse3'].apply(lambda x: -1 if x > 200 else x)
data['pulse'] = data.apply(lambda row: average(row, ['pulse1','pulse2','pulse3']), axis=1)
data['hba1c'] = data['hba1c'].apply(lambda x: -1 if x < 0 else x)
data['ethnicity'] = data['ethnicity'].apply(lambda x: -1 if x < 0 else x)
data['chair'] = data['chair'].apply(lambda x: -1 if x < 0 else x)
data['mobility'] = data['mobility'].apply(lambda x: x-1 if x >= 1 else x)
data['srh'] = data['srh'].apply(lambda x: x if x < 0 else (x-1)/4)
data['eye'] = data['eye'].apply(lambda x: x if x < 0 else (x-1)/5)
data['hear'] = data['hear'].apply(lambda x: x if x < 0 else (x-1)/4)
data['func'] = data['func'].apply(lambda x: x if x < 0 else (x-1)/3)

# remove very fast walkers, probably an error
data['gait speed'] = data['gait speed'].apply(lambda x: x if x < 4 else -1.0)

# impute sex if some are missing
data['sex'].replace(-1.0,np.nan,inplace=True)
data['sex'] = data.groupby('idauniq')['sex'].transform(lambda x: x.fillna(method='ffill'))

# impute ethnicity if some are missing
data['ethnicity'].replace(-1.0,np.nan,inplace=True)
data['ethnicity'] = data.groupby('idauniq')['ethnicity'].transform(lambda x: x.fillna(method='ffill'))
data['ethnicity'] = data.groupby('idauniq')['ethnicity'].transform(lambda x: x.fillna(method='bfill'))

# impute country of birth, if some are missing
data['country'].replace(-1.0,np.nan,inplace=True)
data['country'] = data.groupby('idauniq')['country'].transform(lambda x:
                                                               x.fillna(method='ffill'))
data['country'] = data.groupby('idauniq')['country'].transform(lambda x:
                                                               x.fillna(method='bfill'))

data['ethnicity'] = data['ethnicity'].fillna(-1)
data['country'] = data['country'].fillna(-1)

# impute height
data['height'].replace(-1.0,np.nan,inplace=True)
data['height'] = data.groupby('idauniq')['height'].transform(lambda x: x.fillna(method='ffill'))
data['height'] = data.groupby('idauniq')['height'].transform(lambda x: x.fillna(method='bfill'))
data['height'] = data['height'].fillna(-1)

# Fill missing age values by using the previous age and the fact that waves are ~2 years apart.
data['new age'] = -1
for index, group in data.groupby('idauniq'):
    if np.any(np.diff(group['age']) <= 0):
        waves = group.xs(index,level=1).index.values
        for w, wave in enumerate(waves):
            if w > 0:
                #print(wave, index, data.loc[(waves[w-1],index),'age'])
                if data.loc[(wave,index),'age'] <= data.loc[(waves[w-1],index),'age'] and data.loc[(wave,index),'age'] > 0 and data.loc[(waves[w-1],index),'age'] > 0:
                    data.loc[(wave,index),'age'] = data.loc[(waves[w-1],index),'age'] + 2*(waves[w] - waves[w-1])

                if data.loc[(waves[w-1],index),'age'] > 0 and data.loc[(wave,index),'age'] < 0:
                    data.loc[(wave,index),'age'] = data.loc[(waves[w-1],index),'age'] + 2*(waves[w] - waves[w-1])


data['new age'] = -1
for index, group in data.groupby('idauniq'):
    if np.any(np.diff(group['age']) <= 0):
        waves = group.xs(index,level=1).index.values[::-1]
        for w, wave in enumerate(waves):
            if w < len(waves)-1:
                
                if data.loc[(wave,index),'age'] >= data.loc[(waves[w-1],index),'age'] and data.loc[(wave,index),'age'] > 0 and data.loc[(waves[w-1],index),'age'] > 0:
                    data.loc[(wave,index),'age'] = data.loc[(waves[w-1],index),'age'] - 2*np.abs(waves[w] - waves[w-1])

                if data.loc[(waves[w-1],index),'age'] > 0 and data.loc[(wave,index),'age'] < 0:
                    data.loc[(wave,index),'age'] = data.loc[(waves[w-1],index),'age'] - 2*np.abs(waves[w] - waves[w-1])


for index, group in data.groupby('idauniq'):
    if np.any(group['age'] < 0):
        waves = group.xs(index,level=1).index.values
        for w, wave in enumerate(waves):
            if w == 0 and len(waves) > 1 and data.loc[(wave,index),'age'] < 0 and data.loc[(waves[w+1],index),'age'] > 0:
                data.loc[(wave, index), 'age'] = data.loc[(waves[w+1],index),'age'] - 2*np.abs(waves[w] - waves[w-1])


# Calculate death ages from date of birth (dob) and date of death (dod).
data['death age'] = int(-1)
for index, group in data.groupby('idauniq'):
    
    selected = data.xs(index,level=1)
    
    dod = selected['dod'].max()
    dob = selected['dob'].max()
    
    for w,wave in enumerate(selected.index.values):
        data.loc[(wave,index),'death age'] = int(dod-dob) if dob>0 and dod>0 else -1
        data.loc[(wave,index),'status'] = int(1) if dob>0 and dod>0 else 0 # 1 if dead

# There could be issues with last age > death age/censoring age, due to the approximations with birth age and year of interview. If that occurs, set death age/censoring age to the maximum measured age. 
for index, group in data.groupby('idauniq'):
    selected = data.xs(index,level=1).index.values
    if np.any(group['death age'] > 0):
        for wave in selected:
            data.loc[(wave,index),'death age'] = max(data.loc[(selected[-1],index),'age'],
                                                    data.loc[(selected[-1],index),'death age'])

# variables we want
deficits = ['gait speed','grip dom', 'grip ndom', 'FI ADL', 'FI IADL', 'srh',
            'eye', 'hear', 'func', 'chair', 'leg raise', 
            'full tandem','dias', 'sys','bmi', 'pulse', 'trig','crp','hdl','ldl','glucose','igf1','hgb',
            'fib','fer', 'chol', 'wbc', 'mch', 'hba1c', 'vitd', 'dheas', 'apoe']

data[deficits] = data[deficits].fillna(-1.0)
data['sex'] = data['sex'].fillna(-1)
data['height'] = data['height'].fillna(-1)


# Remove all individuals who don't have at least 1 measurement, have sex missing, have ethnicity missing, or have missing ages that couldn't be approximated through the year of interview.
indexes = [x[1] for x in data.index.values]
dropping = []
for i,index in enumerate(np.unique(indexes)):
    
    selected = data.xs(index,level=1)
    
    remove = True
    for w,wave in enumerate(selected.index.values):
        current = data.loc[(wave,index),deficits]
        
        if np.any(current.values > -1.0) and np.all(selected['age'].values > 0) \
        and np.all(selected['sex'].values >= 0) and np.all(selected['ethnicity'].values >= 0):
            remove = False
            break
            
    if remove:
        dropping.append(index)
data.drop(dropping,level=1,inplace=True)
indexes_count = [x[1] for x in data.index.values]

# Set missing values to -1000 because we will be log-scalling some variables (=> possible negatives).
for d in deficits + ['srh', 'effort', 'longill', 'height', 'BP med', 'anticoagulent med', 
                     'chol med', 'hip/knee treat', 'lung/asthma med']:
    data[d] = data[d].apply(lambda x: -1000 if x < 0 else x)


# Log-scale ferritin, triglycerides, C-reactive protein, white blood cell counts, mean corpuscular hemoglobin, vitamin-D, and DHEAS. These distributions look very skewed and have big ranges of possible values.
for d in ['fer','trig','crp', 'wbc', 'mch', 'vitd', 'dheas']:
    data[d] = data[d].apply(lambda x: np.log(x) if x > -1000 else -1000)

data['leg raise'] = data['leg raise'].apply(lambda x: np.log(x) if x > -1000 else -1000)
data['full tandem'] = data['full tandem'].apply(lambda x: np.log(x+1) if x > -1000 else -1000)

# reset index to just wave and id as columns instead of the index
data = data.reset_index()
data.rename(columns={'level_0':'wave','idauniq':'id'}, inplace=True)

# columns to output
columns = ['id','wave','age','gait speed', 'grip dom', 'grip ndom', 'FI ADL', 'FI IADL',
           'chair','leg raise', 'full tandem', 'dias', 'sys','bmi','pulse','trig','crp',
           'hdl','ldl','glucose','igf1','hgb','fib','fer','chol','wbc','mch','hba1c',
           'vitd','dheas','apoe','BP med', 'anticoagulent med', 'chol med', 
           'hip/knee treat', 'lung/asthma med', 'srh', 
           'longill', 'limitact', 'effort', 'smkevr', 'smknow', 'eye', 'hear',
           'func', 'height', 'alcohol', 'country', 'jointrep', 'fractures',
           'mobility', 'ethnicity','sex','death age']

data.fillna(-1000.0, inplace=True)
final_data = data[columns].sort_values(by=['id','wave'])
final_data = final_data.dropna(subset=['age'], how = 'any')

# fix any missing
final_data['eye'] = final_data['eye'].apply(lambda x: -1000 if x < 0 else x)
final_data['hear'] = final_data['hear'].apply(lambda x: -1000 if x < 0 else x)
final_data['func'] = final_data['func'].apply(lambda x: -1000 if x < 0 else x)
final_data['smkevr'] = final_data['smkevr'].apply(lambda x: -1000 if x < 0 else x)
final_data['smknow'] = final_data['smknow'].apply(lambda x: -1000 if x < 0 else x)
final_data['limitact'] = final_data['limitact'].apply(lambda x: -1000 if x < 0 else x)
final_data['alcohol'] = final_data['alcohol'].apply(lambda x: -1000 if x < 0 else x)
final_data['fractures'] = final_data['fractures'].apply(lambda x: -1000 if x < 0 else x)
final_data['country'] = final_data['country'].apply(lambda x: -1000 if x < 0 else x)
final_data['jointrep'] = final_data['jointrep'].apply(lambda x: -1000 if x < 0 else x)
final_data['mobility'] = final_data['mobility'].apply(lambda x: -1000 if x < 0 else x)

final_data[columns].to_csv('../Data/ELSA_cleaned.csv',index=False)
