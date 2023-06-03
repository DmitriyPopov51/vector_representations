

import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import classification_report
from scipy import spatial

import datetime
import pytz

import json

import argparse

import time

from art import tprint


# парсер аргументов для скрипта 
def arg_parser():
    parser = argparse.ArgumentParser(description='Testing vectors script')
    parser.add_argument("path_vect", help="path to the file with vectors")
    parser.add_argument("mode", 
                        help="vector formation method (1-5 base-4096, 6-10 large-4096, 11-12 sbert)", 
                        type=int)
    parser.add_argument("path_test", help="path to the file with test values")
    parser.add_argument("path_eval", help="path to the directory for output with evaluations")
    parser.add_argument("start", help="initial threshold value", type=float)
    parser.add_argument("stop", help="final threshold value", type=float)
    parser.add_argument("step", help="step of increasing the threshold", type=float)
    parser.add_argument("-d", "--dataset", 
                        help="name of the dataset by which the vectors were formed (default='dataset')", 
                        default='dataset')
    parser.add_argument("-ft", "--finetune",
                        action='store_true',
                        help="Use fine tune model")
    return(parser.parse_args())


# Cosine Similarity
# vector1, vector2 - векторы в numpy
# НА ВЫХОД:
# result - cosine similarity
def cosine_similarity(vector1, vector2):
#     result = np.dot(vector1, vector2)/(norm(vector1)*norm(vector2))
    result = float(1 - spatial.distance.cosine(vector1, vector2))
    return result


# преобразование датафрейма (id , vector) в словарь result_dict
# df - датафрейм (id , vector)
# НА ВЫХОД:
# result_dict - преобразованный к словарю датафрейм id : vector(список из float)
def df_to_dict(df):
    doc_ids = df['doc_id'].values
    vectors = df['vector'].values
    result_dict = {doc_ids[i]: json.loads(vectors[i]) for i in range(len(doc_ids))}
    return result_dict


# создание списков целевых и предполагаемых значений
# test_df - датафрейм test.csv
# result_dict - датафрейм result.csv один из режимов
# threshold - порог значения. Если cosine similarity > threshold, то predict_label = 1
# НА ВЫХОД:
# true_label - целевые значение в виде списка
# predict_label - предполагаемые цели в виде списка
def prediction_label(test_df, result_dict, threshold):
    
    
    true_label = []
    predict_label = []
    for i in test_df.index:
        doc_id_1 = test_df.take([i]).values[0][0] #doc_id_1
        doc_id_2 = test_df.take([i]).values[0][2] #doc_id_2
        label = test_df.take([i]).values[0][4] #label
        
        vector1 = result_dict.get(doc_id_1)
        vector2 = result_dict.get(doc_id_2)
        
        if type(vector1) is not list:
            print(f'Error: we have not vector in file with vectors by index = {doc_id_1}. Check the correctness of the used files with vectors and test values','\n')
        if type(vector2) is not list:
            print(f'Error: we have not vector in file with vectors by index = {doc_id_2}. Check the correctness of the used files with vectors and test values','\n')
        
       
        cos_sim = cosine_similarity(vector1, vector2)
        if cos_sim > threshold:
            prediction_label = 1
        else:
            prediction_label = 0
        
        predict_label.append(prediction_label)
        true_label.append(label)
    return true_label, predict_label


# classification report
def classif_report(true_label, predict_label):
#     return classification_report(true_label, predict_label, output_dict=True)
# чтобы не было предупреждения о делении на ноль можно добавить zero_division=0
    return classification_report(true_label, predict_label, zero_division=0, output_dict=True)


# тестирование порогового значения от START до END с шагом STEP

# start, end, step - начало, конец, шаг
# size - размер модели (4096 и тд)
# path_test_file - название тестового файла .csv 
# path_result_file - путь к файлу с векторами .csv
# mode - режим работы программы
#    LONGFORMER (base-4096, large-4096)
#    1, 6 - способ суммирования, когда текст делится на части и каждая часть обрабатывается отдельно 
#        (был текст 10000, поделили на 3 части, обработали и получили общий результат)
#    2, 7 - способ суммирования, когда текст обрезается и модель обрабатывает только начало (напр. только 4096 токенов)
#    3, 8 - способ максимального в столбце, когда текст делится на части и каждая часть обрабатывается отдельно 
#    4, 9 - способ максимального в столбце, когда текст обрезается и модель обрабатывает только начало (напр. только 4096 токенов)
#    5, 10 - способ <cls> токен, когда текст токенизируется с добавлением спец токенов, и после работы модели берется первый вектор,
#        который соответствует <cls> токену
#        в tokenizer(text, True) обязательно должно быть TRUE
#    SBERT
#    11 - способ суммирования
#    12 - способ максимального в столбце

# короткие названия режимов
#    1, 6 - sum_vectors_full_text
#    2, 7 - sum_vectors_begin_text
#    3, 8 - max_in_column_full_text
#    4, 9 - max_in_column_begin_text
#    5, 10 - cls_token
#    11 - sum_vectors
#    12 - max_in_column


def threshold_testing(start, end, step, mode, size, path_test_file, path_result_file, model, dataset, method):
    print('\n', 'reading test file...', end=' ')
    test_df = pd.read_csv(path_test_file)
    print('done')
    print('\n', 'reading file with vectors...', end=' ')
    result_df = pd.read_csv(path_result_file)
    print('done')
    print('\n', 'making dictionary with vectors...', end=' ')
    result_dict = df_to_dict(result_df)
    print('done')
    result_list = []
    
    modes_in_text = {1: 'sum_vectors_full_text', 
                     2: 'sum_vectors_begin_text',
                     3: 'max_in_column_full_text', 
                     4: 'max_in_column_begin_text',
                     5: 'cls_token',
                     6: 'sum_vectors_full_text', 
                     7: 'sum_vectors_begin_text',
                     8: 'max_in_column_full_text', 
                     9: 'max_in_column_begin_text',
                     10: 'cls_token',
                     11: 'sum_vectors',
                     12: 'max_in_column'
    }
    
    thr = start
    while thr < end:
        dict1 = {}
        
        dict1['model'] = model

        model_settings = {}
        model_settings['mode'] = modes_in_text.get(mode)
        if size is not None:
            model_settings['size'] = size
        dict1['model_settings'] = model_settings

        moscow_time = datetime.datetime.now(pytz.timezone('Europe/Moscow'))
        time_now = moscow_time.strftime("%d-%m-%Y %H:%M:%S")
        dict1['date'] = time_now

        dict1['dataset'] = dataset

        task = {}
        task['file'] = path_test_file
        task['method'] = method
        task['threshold'] = thr
        dict1['task'] = task

        metrics = {}
        true_label, predict_label = prediction_label(test_df, result_dict, thr)
        cl_rep = classif_report(true_label, predict_label)
        label0 = cl_rep.get('0')
        label0.pop('support')
        label1 = cl_rep.get('1')
        label1.pop('support')
        metrics['0'] = label0
        metrics['1'] = label1
        dict1['metrics'] = metrics
        
        result_list.append(dict1)
        
        thr += step
        
        # для отображения процесса работы программы в консоли
#         print('\r', 'processing...', str(thr), '/', str(end), end='')
        # в процентах
        percent = int(((thr - start) / (end - start)) * 100)
        if percent > 100: percent = 100
        print('\r', 'processing...', str(percent), '% / 100 %', end='')
    return result_list


# запись в json файл
# result_list - список из словарей
# path_json - путь к файлу, в который записываем
def write_to_json(result_list, path_json):
    with open(path_json, 'w') as file:
        json.dump(result_list, file, indent=4)

# из пути достаем название файла csv
# ВХОД
# path_vectors - путь до векторов строкой
# ВЫХОД
# model_name - название модели
def get_model_name(path_vectors):
    l = list(path_vectors)
    f = []
    for i in range(len(l)-1, -1, -1):
#             print(l[i],end='')
        if l[i] == '/':
            break
        f.append(l[i])
    n = ''
    for i in range(len(f)-1, -1, -1):
        n += f[i]
    # print(n)
    model_name = n[:-4]
    return model_name
        
        
def main():
    args = arg_parser()
    
    path_vectors = args.path_vect
    model_mode = args.mode
    path_test_file = args.path_test
    path_json = args.path_eval + '/evaluation_balanced_' + str(model_mode) + '.json'
    finetune = args.finetune
    
    if finetune:
        if model_mode > 0 and model_mode <= 10:
            model_name = get_model_name(path_vectors)
            model_size = 4096
            path_json = args.path_eval + '/evaluation-' + str(model_name) + '.json'
        elif model_mode > 10 and model_mode <= 12:
            model_name = get_model_name(path_vectors)
            model_size = None
            path_json = args.path_eval + '/evaluation-' + str(model_name) + '.json'
    else:
        if model_mode > 0 and model_mode <= 5:
            model_name = 'base-4096'
            model_size = 4096
        elif model_mode > 5 and model_mode <= 10:
            model_name = 'large-4096'
            model_size = 4096
        elif model_mode > 10 and model_mode <= 12:
            model_name = 'sbert'
            model_size = None
    dataset = args.dataset
    method = 'cosine similarity'
    
    start = args.start
    stop = args.stop
    step = args.step
    
    tprint('Doc-Sem-Match', font='slant')
    
    print("The script is running...")
    print('__')
    print("The path to the file with vectors:", path_vectors)
    print("The vector formation method:", model_mode)
    print("The path to the file with test values:", path_test_file)
    print("The path to the directory for output with evaluations:", args.path_eval)
    print("The threshold value from {} to {} with step {}".format(start, stop, step))
    print("The name of the dataset by which the vectors were formed:", dataset)
    print("Using fine tune model:", finetune)
    print('__')
    print('After successful execution of the program, the result of the program will be in the file:', path_json)
    print('__')
    
    moscow_time1 = datetime.datetime.now(pytz.timezone('Europe/Moscow'))
    time_now1 = moscow_time1.strftime("%d-%m-%Y %H:%M:%S")
    print("The script was launched at", time_now1)
    
    result_list = threshold_testing(start, stop, step,
                                    model_mode, model_size, path_test_file, path_vectors, model_name, dataset, method)
    
    write_to_json(result_list, path_json)
    
    moscow_time2 = datetime.datetime.now(pytz.timezone('Europe/Moscow'))
    time_now2 = moscow_time2.strftime("%d-%m-%Y %H:%M:%S")
    print("\nThe script is completed at", time_now2)
    execution_time = moscow_time2 - moscow_time1
    print('Execution time =', execution_time)
    
    
if __name__ == "__main__":
    main()
