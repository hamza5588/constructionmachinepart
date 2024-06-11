import numpy as np
import pandas as pd
import streamlit as st
import joblib
import pickle




with open(r"C:\Users\PMLS\Desktop\machinelearningproject\machinepartprediction\lrmodel.pkl", "rb") as f:
    model = pickle.load(f)


filename = r'C:\Users\PMLS\Desktop\machinelearningproject\machinepartprediction\machinemodel.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

filename2 = r'C:\Users\PMLS\Desktop\machinelearningproject\machinepartprediction\descrition.pkl'
dload = pickle.load(open(filename2, 'rb'))
filename3 = r'C:\Users\PMLS\Desktop\machinelearningproject\machinepartprediction\partno.pkl'
pload = pickle.load(open(filename3, 'rb'))

# with open(r"C:\Users\PMLS\Desktop\New folder (3)\machinemodel.pkl", "rb") as f:
#     machine_model = pickle.load(f)

# print(machine_model)
# with open(r"C:\Users\PMLS\Desktop\New folder (3)\partno.pkl", "rb") as f:
#     part_no = pickle.load(f)

# with open(r"C:\Users\PMLS\Desktop\New folder (3)\descrition.pkl", "rb") as f:
#     description= pickle.load(f)


# Sidebar
machine_models = ['FD50AT-7', 'KOM-Mach', 'HD465-7', 'HD325-6', 'GD663A-2',
       'WA200-5', 'WA320-5', 'HD325-7R', 'PC200-8NI', 'PC200LC-8MO',
       'HD465-7R', 'D155A-5', 'D9R', 'PC200-8', 'PC200-6', 'PC600-7',
       'FD30JC-14', 'D85EX-15', 'D155A-6R', 'HD405-7', 'PC1250SP-8R',
       'D155A-2', 'PC220-8MO', 'FD60-7', 'WA500-1', 'TP500', 'FB20EX-II',
       'WA500-3', 'WA600-3A', 'PC200/300-8MO', 'PC200-7', '6D102E-1',
       'WA380-3', 'WA380-6', 'Astec', 'WA500-3A', 'Amman', 'PC240-8MO',
       '5700C', 'Shop Manuals', 'MC70', 'D65E-12', 'PC220-8', 'D155A-1',
       'Terex (TP500)', 'PC300-8MO', 'PC350-8MO', 'FD50AT-17',
       'Kom-Ge EG85-1', 'D85ESS-2A', 'WA30-2', 'Kom-Mach', 'WA420-3',
       'FD15T-20', 'D85A-18', 'Crane RT-35', 'Kom Mach', 'WA-200-5',
       'D45 & PC460', 'D80A-18', 'Kom Mchines', 'FD50AYT-10',
       'KOM Mach Shop  Manuals', 'ECP34-25-4', 'Ohatsu Genset',
       'FB25RL-4', 'Donalsand Fltrs', 'Kom-Genset', 'Lub-Kom-Gen', 'D8N',
       'WL-966211', 'Mechanical Sweeb (Turkish)', 'WA470-6', 'HD3255-6A',
       'WB93S-5E', 'Kom-Exca', 'D155A-3', 'D155', 'FD30-10',
       'Power Curber 5700B', '5700B', 'FD30JC-11', 'PC200',
       'Terex\n1412TP', 'FD30T-16', 'D155A-15', 'Donaldson Fltr', '-',
       'GD605A-3', 'PC800-7', 'FD70-7', 'KOM Machine',
       'PC200/600-7\nD155A-3', 'Manitou MT940L', 'TP: XH500', 'D150A-1',
       'WA180-3', 'PC200-8 & PC200-8MO', 'WB97R-5', 'WA600', 'WA420-3A',
       'HD325-6A', 'PC800SE-7', 'KOM Machines', 'Zega Drill\nD355',
       'TP\n1412UPF', 'Misc', 'Kom Machines', 'Misc Mach', 'D85A-21',
       'PC200-/300', 'FD15T-10', 'Manitou 200 ATJ', 'EG220-1', 'GP33',
       'D41E-6', 'Genset', 'WB93S-5', 'PC200-8 SR#6', 'Terex 1412TP',
       'Oil', 'PC240LC-6K', 'WA500--3', 'XH500', 'PC200LC-8', 'PC300-8',
       'D8R (CAT)', 'CAT\n988G', 'PC200-8NI (JRM-7)', 'D65EX-16',
       'FD50-7', 'GD663A-3', '10\nDrums', 'Zega Drill Rigs', 'WA600-3',
       '30DF-7', 'CAT-1W3830', 'HD465-7\n&\nHD465-7R', 'PC220-6',
       'HD405-7C', 'M50-4', 'Kom -Mach', 'WA500-1/\nD155A-1',
       'TP1412/500', 'WA500-1/ D155A-5', 'PC200/300LC',
       'PC300-8MO/\nPC200LC-8MO', 'TP1412', 'WA500-6',
       'PC200LC-8MO, PC300-8MO', 'FD20T-16', 'FD50E-6', 'FD35AT-17',
       'MC:50', 'PC750-6', '5700-C', 'Manitou-200ATJ',
       'PC200LC/PC300-8MO', 'FB20EX-11', 'WB97R-2', 'PC200-8MO',
       'PC2/300-8MO', 'PC200-10', 'PC200-7/8/10', 'PC110-7',
       'ZX500-LCH / ZX520-LCR-3', 'EX550-LCE  ', 'D355A-3', 'D8K', 'D9G',
       'D8H', 'Misc Models', 'Kom-Forklift', 'WA100-1         ',
       'WA350-3', 'GD505A-3                     ',
       'GD405A-3                     ', 'GH320-2                     ',
       'D275A-2', 'D375A-2', 'D2/375A-2', 'Parkin Part', 'D155AX-3',
       'WA470-3', 'ZX600', 'PC3020-8MO', 'Kom Exc', 'WA320-3',
       'D85EX-15R', 'WB97R-5EO', 'PC200/300', 'BD', 'MT940L',
       '80KVA Genset', 'FD25T-17', '5700C-PC', 'Kom-Lub', 'D275A-2    ',
       'D85P-18    ', 'Under Carriage', 'Work Eqpt', 'D155A-3    ',
       'D375A-3    ', 'D375A-3', 'D85A-21    ', 'WA380-3/ PC300-8MO',
       'D40A-3/D41A-3', 'WA200-6', 'Power Curber', 'WA500-6R',
       'PC200-8MO\nPC300LC-8MO', 'D50A-17', 'D40A-3', 'D37E-5',
       'D40A-3/D37E-5', 'Terex Crane', 'A-356B6100', 'FLT']

part_nos = [
 '01252-60545',
 '6218-11-5830',
 '04120-21737',
 '6217-51-8161',
 '21M-03-15110',
 '06037-06007',
 '6217-K1-9900',
 '6217-K2-9900',
 '20Y-979-6121',
 '17M-911-3530',
 '04120-21740',
 '6505-65-5091',
 '6217-51-8110',
 '6218-K1-9900',
 '6218-K2-9900',
 '285-01-12411',
 '569-01-62410', 
 '(6218-71-1111)\n6218-71-1112',
 '6151-31-2150',
 '(6150-31-2033)\n6150-32-2030',
 '6150-41-4111',
 '6150-41-4210',
 '6150-12-1341',
 '6150-12-1351',
 '(6151-21-2220)\n6151-22-2220',
 '(6151-K1-1002)\n6151-K1-9901',
 '(6151-K2-0404)\n6151-K2-9901',
 '6150-31-3040']

descriptions = [
'Valve Assembly',
 'Solenoid Valve',
 'Bolt',
 'Gasket',
 'V- Belt',
 'Tube',
 'Radiator Core',
 'Bearing Ball',
 'Gasket Kit, Cylinder Head',
 'Gasket Kit, Cylinder Block',
 'Compressor Assy',
 'A/C Fresh Air Filter',
 'Turbocharger',
 'Rubber',
 'Fuel Pump Assembly',
 'Piston',
 'Piston Ring Assembly',
 'Valve Intake',
 'Valve Exhaust',
 'Guide Intake Valve',
 'Guide Exhaust Valve',
 'Liner Cylinder',
 'Crank Pin Metal Assy',
 'Main Metal Assembly',
 'Thrust Metal Assembly',
 'Accumulator',
 'Spider',
 'Komatsu Coolant (200-Litre Packing)',
 'Cartridge, Fuel']

# Sidebar
st.sidebar.title('Input Parameters')
machine_model = st.sidebar.selectbox('Machine Model', machine_models)
part_no = st.sidebar.selectbox('Part No.', part_nos)
description = st.sidebar.selectbox('Description', descriptions)
date = st.sidebar.date_input('Date')
unit_price = st.sidebar.number_input('Unit Price')
a=pd.DataFrame({"Machine Model":machine_model,"Description":description,"Part No.":part_no},index=[0])


machine_model= loaded_model.transform(a["Machine Model"])
machine_model=machine_model.values
description= dload.transform(a["Description"])
description=description.values
part_no = pload.transform(a["Part No."])
part_no=part_no.values

date=str(date)
date=date.split("-")
year=date[0]
month=date[1]
day=date[2]

unit_price=np.log1p(unit_price+1)

input=[[float(machine_model[0][0]),float(part_no[0][0]),float(description[0][0]),int(year),int(month),int(day),float(unit_price)]]
print(input)
print(type(input))
ypredict=model.predict(input)
print(ypredict)
ypredict=int(ypredict)
ypredict=np.abs(ypredict)




# year=date.dt.year
# month=date.dt.month
# date=date.dt.day

# Predict Button
if st.button('Predict'):
    # Dummy Prediction (Replace with your actual prediction logic)
    predicted_price = 1000  # Example prediction
    
    # Display Prediction
    st.write('### Prediction:')
    st.write('Predicted qunatity Price:', ypredict)

st.write('### Prediction:')
    # st.write('Predicted Unit Price:', prediction)
