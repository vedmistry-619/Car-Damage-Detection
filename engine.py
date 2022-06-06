import os
import json
import urllib
import pandas as pd

import h5py
import numpy as np
import pickle as pk

from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.utils.data_utils import get_file

first_gate = VGG16(weights='imagenet')
print("First gate loaded")
second_gate = load_model('static/models/d1_ft_model.h5')
print("Second gate loaded")
location_model = load_model('static/models/d2_ft_model.h5')
print("Location model loaded")
severity_model = load_model('static/models/d3_ft_model.h5')
print("Severity model loaded")
with open('static/models/vgg16_cat_list.pk', 'rb') as f:
    cat_list = pk.load(f)
print("Cat list loaded")

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

features_columns = ['YEAR',
                        'no_of_damages',
                        'score',
                        'BMW',
                        'Ford',
                        300,
                        501,
                        502,
                        503,
                        507,
                        700,
                        1500,
                        2002,
                        2800,
                        '1-Series',
                        '1500, 1600, 1800, 1800TI',
                        '1500, 1800',
                        '1600, 1600-2, 1600TI, 1800, 1800TI, 2000, 2000C, 2000CS, 2000TI',
                        '1600, 1600-2, 1800, 1800TI, 2000, 2000C, 2000CS, 2000TI',
                        '1600, 1600GT, 1600TI, 1800, 2000, 2000C, 2000CS, 2000TI',
                        '1600, 1800, 1800TI, 2000C, 2000CS',
                        '1600, 1800, 2000, 2000C, 2000CS',
                        '1600, 2000',
                        '1600, 2000, 2000 Touring, 2000TII',
                        '1941 Ford: Deluxe / Super Deluxe',
                        '1949 Ford: Deluxe / Custom',
                        '1952 Ford: Mainline / Customline / Crestline',
                        '2-Series',
                        '2-Series Gran Coupé',
                        '2-Series, M2',
                        '2000 Touring, 2000TII',
                        '2002, 2000TI',
                        '2002, 2002TI',
                        '2002, 2002TII',
                        '250 Isetta',
                        '2500, 2800, 2800 Bavaria, 2800CS',
                        '2500, 2800, 2800CS',
                        '2600, 2600L, 3200CS, 3200L, 3200S',
                        '2600, 2600L, 3200L, 3200S',
                        '2600L, 3200CS, 3200S',
                        '2800CS',
                        '3-Series',
                        '3-Series Gran Turismo',
                        '3-Series, M3',
                        '3.0 Bavaria, 3.0CS, 3.0CSL, 3.0S',
                        '3.0CS, 3.0CSL, 3.0S',
                        '3.0CSI, 3.0S, 3.0SIO',
                        '3.0SI',
                        '300 Isetta',
                        '3200CS',
                        '4-Series',
                        '4-Series, M4',
                        '5-Series',
                        '5-Series Gran Turismo',
                        '5-Series, M5',
                        '6-Series',
                        '6-Series Gran Turismo',
                        '6-Series, M6',
                        '600 Isetta',
                        '7-Series',
                        '700, 700 LS',
                        '8-Series',
                        'Aerostar',
                        'Aspire',
                        'Bronco',
                        'Bronco II',
                        'C-Max',
                        'Contour',
                        'Country Squire',
                        'Crown Victoria',
                        'Custom 500',
                        'Durango',
                        'E-Series',
                        'EXP',
                        'EcoSport',
                        'Edge',
                        'Escape',
                        'Escort',
                        'Excursion',
                        'Expedition',
                        'Explorer',
                        'Explorer Sport Trac',
                        'F-Series',
                        'F-Series Super Duty',
                        'Fairlane',
                        'Fairmont',
                        'Falcon',
                        'Falcon Ranchero',
                        'Festiva',
                        'Fiesta',
                        'Five Hundred',
                        'Flex',
                        'Focus',
                        'Freestar',
                        'Freestyle',
                        'Fusion',
                        'GT',
                        'Galaxie',
                        'Gran Torino',
                        'Granada',
                        'LTD',
                        'LTD Crown Victoria',
                        'LTD Fox',
                        'LTD II',
                        'M1',
                        'Maverick',
                        'Mustang',
                        'Mustang Mach-E',
                        'Pinto',
                        'Probe',
                        'Ranchero',
                        'Ranger',
                        'Taurus',
                        'Taurus X',
                        'Tempo',
                        'Thunderbird',
                        'Torino',
                        'Transit',
                        'Transit Connect',
                        'Windstar',
                        'X1',
                        'X2',
                        'X3',
                        'X4',
                        'X5',
                        'X6',
                        'X7',
                        'Z3',
                        'Z3, M',
                        'Z4',
                        'Z8',
                        'i3',
                        'i8']


def get_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    l = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        indexes = [tuple(CLASS_INDEX[str(i)]) + (pred[i],)
                   for i in top_indices]
        indexes.sort(key=lambda x: x[2], reverse=True)
        l.append(indexes)
    return l


def prepare_img_224(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def car_categories_gate(img_224, model):
    print("Validating that this is a picture of your car...")
    out = model.predict(img_224)
    top = get_predictions(out, top=5)
    for j in top[0]:
        if j[0:2] in cat_list:
            # print j[0:2]
            return True
    return False


def prepare_img_256(img_path):
    img = load_img(img_path, target_size=(256, 256))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)/255
    return x


def car_damage_gate(img_256, model):
    print("Validating that damage exists...")
    pred = model.predict(img_256)
    if pred[0][0] <= .5:
        return True
    else:
        return False


def location_assessment(img_256, model):
    print("Determining location of damage...")
    pred = model.predict(img_256)
    pred_label = np.argmax(pred, axis=1)
    d = {0: ('Front', 0.75), 1: ('Rear', 0.5), 2: ('Side', 0.25)}
    for key in d.keys():
        if pred_label[0] == key:
            return d[key]


def severity_assessment(img_256, model):
    print("Determining severity of damage...")
    pred = model.predict(img_256)
    pred_label = np.argmax(pred, axis=1)
    d = {0: ('Minor', 0.25), 1: ('Moderate', 0.5), 2: ('Severe', 0.75)}
    for key in d.keys():
        if pred_label[0] == key:
            return d[key]

def predict_cost(predict_dict):
        print("Predicting cost...")
        model = pk.load(open('static/models/finalized_model.sav', 'rb'))
        test_df = pd.DataFrame(predict_dict)
        dummy_df = pd.get_dummies(test_df['MAKE'])
        test_df = test_df.merge(dummy_df, left_index=True, right_index=True)
        del test_df['MAKE']
        dummy_df = pd.get_dummies(test_df['MODEL'])
        test_df = test_df.merge(dummy_df, left_index=True, right_index=True)
        del test_df['MODEL']
        del dummy_df
        new_df = pd.DataFrame({}, columns=features_columns)
        new_test_df = test_df.merge(new_df, how='left')
        new_test_df.fillna(0, inplace=True)
        predic_df = pd.DataFrame({}, columns=features_columns)
        predic_df[features_columns] = new_test_df[features_columns]
        predict_data = model.predict(predic_df)
        predicted_price = predict_data[0]
        return predicted_price

def engine(img_path, make, car_model, year):
    img_224 = prepare_img_224(img_path)
    g1 = car_categories_gate(img_224, first_gate)

    if g1 is False:
        result = {'gate1': 'Car validation check: ',
                  'gate1_result': 0,
                  'gate1_message': {0: 'Are you sure this is a picture of your car? Please retry your submission.',
                                    1: 'Hint: Try zooming in/out, using a different angle or different lighting'},
                  'gate2': None,
                  'gate2_result': None,
                  'gate2_message': {0: None, 1: None},
                  'location': None,
                  'severity': None,
                  'final': 'Damage assessment unsuccessful!'}
        return result

    img_256 = prepare_img_256(img_path)
    g2 = car_damage_gate(img_256, second_gate)

    if g2 is False:
        result = {'gate1': 'Car validation check: ',
                  'gate1_result': 1,
                  'gate1_message': {0: None, 1: None},
                  'gate2': 'Damage presence check: ',
                  'gate2_result': 0,
                  'gate2_message': {0: 'Are you sure that your car is damaged? Please retry your submission.',
                                    1: 'Hint: Try zooming in/out, using a different angle or different lighting.'},
                  'location': None,
                  'severity': None,
                  'final': 'Damage assessment unsuccessful!'}
        return result

    x, damage_score_x = location_assessment(img_256, location_model)
    y, damage_score_y = severity_assessment(img_256, severity_model)

    dictionary_to_test = {'MAKE': [make], 'YEAR': [year], 'MODEL': [
        car_model], 'no_of_damages': [2], 'score': [(damage_score_x+damage_score_y)/2]}
    print(dictionary_to_test)
    cost = predict_cost(dictionary_to_test)/2
    
    print(cost)

    result = {'gate1': 'Car validation check: ',
              'gate1_result': 1,
              'gate1_message': {0: None, 1: None},
              'gate2': 'Damage presence check: ',
              'gate2_result': 1,
              'gate2_message': {0: None, 1: None},
              'location': x,
              'severity': y,
              'cost': cost,
              'final': 'Damage assessment complete!'}
    return result
