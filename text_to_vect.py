
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer

from transformers import RobertaTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig


import numpy as np
import csv

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

import os

import argparse

import datetime
import pytz

import time

from art import tprint


# парсер аргументов для скрипта
def arg_parser():
    parser = argparse.ArgumentParser(description='Text to vector script')
    parser.add_argument("path_text", 
                        help="path of the dataset with texts", 
                        type=str,
                        default='../notebooks/test_texts')
    parser.add_argument("path_vect", 
                        help="path to the directory for output", 
                        type=str,
                        default='../notebooks')
    parser.add_argument("mode", 
                        help="text processing method (1-5 base-4096, 6-10 large-4096, 11-12 sbert)", 
                        type=int,
                        default=12)
    parser.add_argument("-ug", "--usegpu",
                        action='store_true',
                        help="Use GPU or CPU (if the flag is set - GPU, if not set - CPU)")
    parser.add_argument("-ft", "--finetune",
                        action='store',
#                         action='store_true',
                        help="Use fine tune model (write name, example: -ft longformer-base-4096-tune1)")
    return(parser.parse_args())


# открыть файл
# id - id файла
# folder - путь к папке ('text')
# на ВЫХОД:
# text_in_string текст в формате строки
def open_file(d_id, folder):
    path = folder + '/' + str(d_id) + '.txt'
    text_file = open(path,'r')
    text_in_string = text_file.read()
    text_file.close()
    return text_in_string


# загрузка модели
# model_name - имя модели (base-4096, large-4096, sbert)
# usegpu - TRUE или FALSE, использовать для обработки GPU или CPU
# НА ВЫХОД
# model - подгруженная модель и конфиг
# config
def load_model(model_name, usegpu, mode):
    if model_name == 'base-4096':
        config = LongformerConfig.from_pretrained('/workspace/src/models/longformer-base-4096/') 
        config.attention_mode = 'sliding_chunks'
        model = Longformer.from_pretrained('/workspace/src/models/longformer-base-4096/', config=config)
        # если usegpu == True, то используем GPU
        if usegpu:
            model = model.cuda()
    elif model_name == 'large-4096':
        config = LongformerConfig.from_pretrained('/workspace/src/models/longformer-large-4096/') 
        config.attention_mode = 'sliding_chunks'
        model = Longformer.from_pretrained('/workspace/src/models/longformer-large-4096/', config=config)
        # если usegpu == True, то используем GPU
        if usegpu:
            model = model.cuda()
    elif model_name == 'sbert':
        # если usegpu == True, то используем GPU
        if usegpu:
#             model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')
#             model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
            model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cuda')
        else:
#             model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
#             model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
        # конфиг равен -1 потому что он не нужен в дальнейшей работе
        config = -1
    else: # значит finetune модель
        if mode > 0 and mode <= 10: # ft longformer
            config = LongformerConfig.from_pretrained('/workspace/src/models/'+model_name+'/') 
            config.attention_mode = 'sliding_chunks'
            model = Longformer.from_pretrained('/workspace/src/models/'+model_name+'/', config=config)
            # если usegpu == True, то используем GPU
            if usegpu:
                model = model.cuda()
        else: # ft sbert
            if usegpu:
                model = SentenceTransformer('/workspace/src/models/'+model_name+'/', device='cuda')
            else:
                model = SentenceTransformer('/workspace/src/models/'+model_name+'/', device='cpu')
            # конфиг равен -1 потому что он не нужен в дальнейшей работе
            config = -1
        
    return model, config


# модель sbert
# text - текст в формате строки
# model - подгруженная заранее модель
# НА ВЫХОД
# output - numpy массив с векторами, добавленный в список
# output - [[array[embeddings]]]
# такой формат вывода нужен для удобства использования функций от longformer
def model_sbert(text, model):
#     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)
    output = []
    output.append([embeddings])
    return output


# удаление списка токенов, у которого длина 21 и меньше
# иначе attention_mask в модели при обращении к 21 индексу будет выдавать
# ошибку IndexError: index 21 is out of bounds
# new_tokens - список из списков с токенами (результат работы split_tokens)
# НА ВЫХОД:
# new_tokens - список из списков с токенами, но без последнего списка, если в нем меньше 22 элементов
# если последний список больше 21 элемента, то вернется тот же самый список, что и на входе был
def remove_short_list(new_tokens):
    if len(new_tokens[-1]) < 22:
        new_tokens.pop(-1)
    return new_tokens


# деление списка из токенов, на списки <= 4096 токенов
# tokens_list - список токенов 
# new_size - размерность списков, на которые надо разбить
# далее удалять пустые (где токен = 1, т.е. где закончился текст) строки можно только у первого с конца вложенного списка
# на ВЫХОД:
# new_tokens - список из списков с токенами
# new_tokens - список со списками < new_size
#     выглядит в формате [[0,1,2], [3,4,5], [6,7]]
def split_tokens(tokens_list, new_size):
    new_tokens = []
    i = 0
    number_list = -1
    while i != len(tokens_list):
        if i % new_size == 0:
            new_tokens.append([])
            number_list += 1
            new_tokens[number_list].append(tokens_list[i])
        else:
            new_tokens[number_list].append(tokens_list[i])
        i += 1
    new_tokens = remove_short_list(new_tokens)
    return new_tokens


# сокращение списка токенов до списка <= 4096 
# tokens_list - список токенов 
# new_size - размерность списка, до которого надо сократить
# НА ВЫХОД:
# new_tokens - обрезанный список с токенами (<= new_size)
#     выглядит в формате вложенного списка [[0,1,2]]
def cut_tokens(tokens_list, new_size):
    new_tokens = []
    tok = tokens_list[:new_size]
    new_tokens.append(tok)
    return new_tokens


# запуск модели longformer-base-4096
# tokens - список из токенов (после обработки длины, то есть меньше 4096)
# pad_token_id - токен необходимый в модели(tokenizer.pad_token_id)
# model - модель подгруженная заранее
# config - конфиг модели подгруженный заранее
# usegpu - TRUE или FALSE, использовать для обработки GPU или CPU
# mode - номер режима. Необходим для правильного выбора attention_mask
# на ВЫХОД:
# output - tensor со значениями
def model_base_4096(tokens, pad_token_id, model, config, usegpu, mode):
    input_ids = torch.tensor(tokens).unsqueeze(0)  # batch of size 1
    # если usegpu == True, то используем GPU
    if usegpu:
        input_ids = input_ids.cuda()
    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
    if mode == 5:
        attention_mask[:, [0]] =  2
    else:
        attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
                                         # classification: the <s> token
                                         # QA: question tokens
    # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
    input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, config.attention_window[0], pad_token_id)
    
    output = model(input_ids, attention_mask=attention_mask)[0]
    return output


# запуск модели longformer-large-4096
# tokens - список из токенов (после обработки длины, то есть меньше 4096)
# pad_token_id - токен необходимый в модели(tokenizer.pad_token_id)
# model - модель подгруженная заранее
# config - конфиг модели подгруженный заранее
# usegpu - TRUE или FALSE, использовать для обработки GPU или CPU
# mode - номер режима. Необходим для правильного выбора attention_mask
# на ВЫХОД:
# output - tensor со значениями
def model_large_4096(tokens, pad_token_id, model, config, usegpu, mode):
    input_ids = torch.tensor(tokens).unsqueeze(0)  # batch of size 1
    # если usegpu == True, то используем GPU
    if usegpu:
        input_ids = input_ids.cuda()
    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
    if mode == 10:
        attention_mask[:, [0]] =  2
    else:
        attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
                                         # classification: the <s> token
                                         # QA: question tokens
    # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
    input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, config.attention_window[0], pad_token_id)
    
    output = model(input_ids, attention_mask=attention_mask)[0]
    return output


# запуск finetune модели (model_longformer_4096_finetune), подгруженной ранее
# tokens - список из токенов (после обработки длины, то есть меньше 4096)
# pad_token_id - токен необходимый в модели(tokenizer.pad_token_id)
# model - модель подгруженная заранее
# config - конфиг модели подгруженный заранее
# usegpu - TRUE или FALSE, использовать для обработки GPU или CPU
# mode - номер режима. Необходим для правильного выбора attention_mask
# на ВЫХОД:
# output - tensor со значениями
def model_longformer_4096_finetune(tokens, pad_token_id, model, config, usegpu, mode):
    input_ids = torch.tensor(tokens).unsqueeze(0)  # batch of size 1
    # если usegpu == True, то используем GPU
    if usegpu:
        input_ids = input_ids.cuda()
    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
    if mode == 5 or mode == 10:
        attention_mask[:, [0]] =  2
    else:
        attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
                                         # classification: the <s> token
                                         # QA: question tokens
    # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
    input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, config.attention_window[0], pad_token_id)
    
    output = model(input_ids, attention_mask=attention_mask)[0]
    return output


# токенизируем текст
# text - текст в формате строки
# add_special_tokens - True or False, добавлять специальные токены или нет (<cls> <s>)
# НА ВЫХОД:
# tokens - список из токенов
# pad_token_id - токен необходимый в модели
def tokenize_text(text, add_special_tokens, tokenizer):
    if add_special_tokens == True:
        tokens = tokenizer.encode(text, add_special_tokens=True)
    else:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens, tokenizer.pad_token_id


# загрузка tokenizer
# НА ВЫХОД
# tokenizer - подгруженный токенайзер
def load_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#     tokenizer = AutoTokenizer.from_pretrained('roberta-base')
#     tokenizer = AutoTokenizer.from_pretrained('../models/longformer-base-4096-epoch1-lr3e-4-sigm/')
#     tokenizer.model_max_length = model.config.max_position_embeddings
    tokenizer.model_max_length = 1000000
    return tokenizer


# функция прогона через модель токенов в виде списка со списками
# (запуск модели) для base-4096 и large-4096

# model_name - строка с названием модели (base-4096, large-4096)
#tokens_list - список со списками из токенов (после обработки длины, то есть меньше 4096)
# pad_token_id - токен необходимый в модели
# model - модель подгруженная заранее
# config - конфиг модели подгруженный заранее
# usegpu - TRUE или FALSE, использовать для обработки GPU или CPU
# mode - номер режима. Необходим для правильного выбора attention_mask
# КАК РАБОТАЕТ:
# получаем на вход список со списками токенов
# запускаем цикл по длине списка (сколько частей в тексте)
# в цикле запускаем выбранную модель, получаем output в tensor формате
# преобразуем output к numpy
# создаем список и добавляем в него output в numpy
# НА ВЫХОД:
# outputs - список из numpy arrays
# outputs = [array[output1], array[output2], array[output3]]

def launch_model(model_name, tokens_list, pad_token_id, model, config, usegpu, mode):
    outputs = []
    for tokens in tokens_list:
        if model_name == 'base-4096':
            output_tensor = model_base_4096(tokens, pad_token_id, model, config, usegpu, mode)
            output_numpy = output_tensor.cpu().detach().numpy()
        elif model_name == 'large-4096':
            output_tensor = model_large_4096(tokens, pad_token_id, model, config, usegpu, mode)
            output_numpy = output_tensor.cpu().detach().numpy()
        else:
            output_tensor = model_longformer_4096_finetune(tokens, pad_token_id, model, config, usegpu, mode)
            output_numpy = output_tensor.cpu().detach().numpy()
        del output_tensor
        torch.cuda.empty_cache()        
        outputs.append(output_numpy)
    return outputs


# удаление векторов с пустыми словами
# outputs - результат работы модели, добавленный в список
#    (получается [array[output1], array[output2], array[output3]])
#    (нас интересует первый с конца, потому что остальные будут максимально заполнены)
#    (доработать можно, не копируя весь output а в функцию брать сразу первый с конца)
# split_tokens - список со списками токенов (разделенные по частям)
#    нас интересует len(split_tokens[-1]) (длина первого с конца токенизированного списка)
# НА ВЫХОД:
# список [array[output1], array[output2], array[output3]], но в array[output3] нет пустых слов 
def delete_empty_vectors(outputs, split_tokens):
    new_output = outputs
    new_output[-1] = new_output[-1][:len(split_tokens[-1])]
    return new_output


# суммирование поэлементно всех векторов -> получаем 1 вектор
# суммирование векторов (создание 1 вектора для текста)
# output_numpy - список из numpy_array с результатами работы модели, преобразованный к numpy
#     output_numpy = [array[output1], array[output2], array[output3]]
#     обязательно передавать сюда output_numpy с удаленными пустыми строками у последнего array[output3]
#     (то есть после функции delete_empty_vectors)
# на ВЫХОД:
# result_vector - numpy array с итоговым вектором для текста. его можно добавлять в файл
def sum_by_element(output_numpy):
    sums = []
    for i in output_numpy:
        summ = np.sum(i[0], axis=0)
        sums.append(summ)
    sums = np.array(sums)
    result_vector = np.sum(sums, axis=0)
    return result_vector


# максимальное значение по столбцу в массиве -> получаем 1 вектор
# берем каждый output, в нем находим максимальное значение по столбцу
# и затем находим максимальное значение среди максимальных в каждом куске текста (создание 1 вектора для текста)
# 
# output_numpy - список из numpy_array с результатами работы модели, преобразованный к numpy
#     output_numpy = [array[output1], array[output2], array[output3]]
#     обязательно передавать сюда output_numpy с удаленными пустыми строками у последнего array[output3]
#     (то есть после функции delete_empty_vectors)
# на ВЫХОД:
# result_vector - numpy array с итоговым вектором для текста. его можно добавлять в файл
def max_in_column(output_numpy):
    maxs = []
    for i in output_numpy:
        max_in_column = np.amax(i[0], axis=0)
        maxs.append(max_in_column)
    maxs = np.array(maxs)
    result_vector = np.amax(maxs, axis=0)
    return result_vector


# взятие вектора, соответствующего <cls> (<s>) токену
# берем первый вектор в output - он соответствует <cls>

# output_numpy - список из numpy_array с результатами работы модели, преобразованный к numpy
#     output_numpy = [array[output1], array[output2], array[output3]]
#     можно передавать без удаленных пустых строк
# на ВЫХОД:
# result_vector - numpy array с итоговым вектором для текста. его можно добавлять в файл
def take_cls_token(output_numpy):
    result_vector = output_numpy[0][0][0]
    return result_vector


# создание csv файла
# name_csv - имя файла
# def create_csv(name_csv):
#     path = name_csv + '.csv'
#     with open(path, mode='a') as file:
#         file_writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
def create_csv(name_csv):
    path = name_csv + '.csv'
    with open(path, mode='w') as file:
        file_writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['doc_id', 'vector'])


# добавление в файл id и result_vector
# csv {doc_id,vector}
# name_csv - имя файла
# doc_id - id документа
# vector - итоговый вектор в NUMPY для текста
# открывается файл и дозаписывается в конец

def write_to_csv(name_csv, doc_id, vector):
    path = name_csv + '.csv'
    with open(path, mode='a') as file:
        file_writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow([doc_id, vector])
        

# функция запускающая прогон для одного текста
# d_id - id текста
# folder - папка, в которой лежат все тексты
# name_csv - имя файла csv, в котором будут результаты работы программы
# model_size - 4096 и т.д. размер модели
# model_name - выбираем модель 'base-4096' или 'large-4096' (только для лонгформера)
# model - подгруженная модель
# config - конфиг подгруженной модели (только для 'base-4096' или 'large-4096', иначе просто передаём 0)
# tokenizer - подгруженный токенайзер (только для 'base-4096' или 'large-4096', иначе просто передаём 0)
# usegpu - TRUE или FALSE, использовать для обработки GPU или CPU
# mode - режим работы программы
#    LONGFORMER (base-4096, large-4096)
#    1, 6 - способ суммирования, когда текст делится на части и каждая часть обрабатывается отдельно 
#        (был текст 10000, поделили на 3 части, обработали и получили общий результат)
#    2, 7 - способ суммирования, когда текст обрезается и модель обрабатывает только начало (напр. только 4096 токенов)
#    3, 8 - способ максимального в столбце, когда текст делится на части и каждая часть обрабатывается отдельно 
#    4, 9 - способ максимального в столбце, когда текст обрезается и модель обрабатывает только начало 
#        (напр. только 4096 токенов)
#    5, 10 - способ <cls> токен, когда текст токенизируется с добавлением спец токенов, 
#        и после работы модели берется первый вектор,
#        который соответствует <cls> токену
#        в tokenizer(text, True) обязательно должно быть TRUE
#    SBERT
#    11 - способ суммирования
#    12 - способ максимального в столбце
def start_for_one_text(d_id, folder, name_csv, model_size, model_name, mode, model, config, tokenizer, usegpu):
    if mode == 1 or mode == 6:
        text = open_file(d_id, folder)
        tokens, pad_token_id = tokenize_text(text, False, tokenizer)
        split_tok = split_tokens(tokens, model_size)
        outputs = launch_model(model_name, split_tok, pad_token_id, model, config, usegpu, mode)
        new_outputs = delete_empty_vectors(outputs, split_tok)
        result_vector = sum_by_element(new_outputs)
        result_vector_list = result_vector.tolist()
        write_to_csv(name_csv, d_id, result_vector_list)
    elif mode == 2 or mode == 7:
        text = open_file(d_id, folder)
        tokens, pad_token_id = tokenize_text(text, False, tokenizer)
        cut_tok = cut_tokens(tokens, model_size)
        outputs = launch_model(model_name, cut_tok, pad_token_id, model, config, usegpu, mode)
        new_outputs = delete_empty_vectors(outputs, cut_tok)
        result_vector = sum_by_element(new_outputs)
        result_vector_list = result_vector.tolist()
        write_to_csv(name_csv, d_id, result_vector_list)
    elif mode == 3 or mode == 8:
        text = open_file(d_id, folder)
        tokens, pad_token_id = tokenize_text(text, False, tokenizer)
        split_tok = split_tokens(tokens, model_size)
        outputs = launch_model(model_name, split_tok, pad_token_id, model, config, usegpu, mode)
        new_outputs = delete_empty_vectors(outputs, split_tok)
        result_vector = max_in_column(new_outputs)
        result_vector_list = result_vector.tolist()
        write_to_csv(name_csv, d_id, result_vector_list)
    elif mode == 4 or mode == 9:
        text = open_file(d_id, folder)
        tokens, pad_token_id = tokenize_text(text, False, tokenizer)
        cut_tok = cut_tokens(tokens, model_size)
        outputs = launch_model(model_name, cut_tok, pad_token_id, model, config, usegpu, mode)
        new_outputs = delete_empty_vectors(outputs, cut_tok)
        result_vector = max_in_column(new_outputs)
        result_vector_list = result_vector.tolist()
        write_to_csv(name_csv, d_id, result_vector_list)
    elif mode == 5 or mode == 10:
        text = open_file(d_id, folder)
        tokens, pad_token_id = tokenize_text(text, True, tokenizer)
        cut_tok = cut_tokens(tokens, model_size)
        outputs = launch_model(model_name, cut_tok, pad_token_id, model, config, usegpu, mode)
        result_vector = take_cls_token(outputs)
        result_vector_list = result_vector.tolist()
        write_to_csv(name_csv, d_id, result_vector_list)
    elif mode == 11:
        text = open_file(d_id, folder)
        outputs = model_sbert(text, model)
        result_vector = sum_by_element(outputs)
        result_vector_list = result_vector.tolist()
        write_to_csv(name_csv, d_id, result_vector_list)
    elif mode == 12:
        text = open_file(d_id, folder)
        outputs = model_sbert(text, model)
        result_vector = max_in_column(outputs)
        result_vector_list = result_vector.tolist()
        write_to_csv(name_csv, d_id, result_vector_list)


# итерация по текстам
# path_of_file - путь до файла со списком id документов
# НА ВЫХОД:
# doc_ids - все id текстов в виде списка
def read_list_of_texts(path_of_file):
    file_with_ids = open(path_of_file,'r')
    doc_ids = file_with_ids.readlines()
    file_with_ids.close()
    return doc_ids


# анализ папки с текстами и сохранение их названий в список
def get_doc_ids_from_dir(path_text):
    doc_ids = []
    for root, dirs, files in os.walk(path_text):  
        for filename in files:
            if filename.endswith('.txt'):
                doc_ids.append(filename[:-4])
    return doc_ids


def main():
    args = arg_parser()
    
    folder_of_texts = args.path_text
    folder_of_vectors = args.path_vect
    mode = args.mode
    usegpu = args.usegpu
    finetune = args.finetune
    
    tprint('TEXT > TO > VECT', font='slant')
    
    print("The script is running...")
    print('__')
    print("The path of the dataset with texts:", folder_of_texts)
    print("The path to the directory for output:", folder_of_vectors)
    print("Text processing method:", mode)
    print("Using GPU:", usegpu)
    print("Using fine tune model:", finetune)
    print('__')
    
    if finetune:
        name_csv = folder_of_vectors + '/vectors-' + finetune
    else:
        name_csv = folder_of_vectors + '/vectors' + str(mode)
    
    print('After successful execution of the program, the result of the program will be in the file:', name_csv + '.csv')
    print('__')
    print('The model is loading...')
    print('__')
    
    if finetune:
        if mode > 0 and mode <= 10:
#             model_name = 'base-4096-tune1'
            model_name = finetune
            model_size = 4096
            model, config = load_model(model_name, usegpu, mode)
            tokenizer = load_tokenizer()
        elif mode > 10 and mode <= 12:
            model_name = finetune
            model_size = -1
            model, config = load_model(model_name, usegpu, mode)
            # токенайзер равен -1 потому что он не нужен в дальнейшей работе
            tokenizer = -1
        else:
            print(f'Error: for finetune model you need to select mode number from 0 to 10. But you chose mode = {mode}')
    else:
        if mode > 0 and mode <= 5:
            model_name = 'base-4096'
            model_size = 4096
            model, config = load_model(model_name, usegpu, mode)
            tokenizer = load_tokenizer()
        elif mode > 5 and mode <= 10:
            model_name = 'large-4096'
            model_size = 4096
            model, config = load_model(model_name, usegpu, mode)
            tokenizer = load_tokenizer()
        elif mode > 10 and mode <= 12:
            model_name = 'sbert'
            model_size = -1
            model, config = load_model(model_name, usegpu, mode)
            # токенайзер равен -1 потому что он не нужен в дальнейшей работе
            tokenizer = -1
    
#     import pandas as pd
#     print('\n', 'reading file with vectors...', end=' ')
#     result_df = pd.read_csv(name_csv + '.csv')
#     print('done')
#     list_processed_id = list(result_df['doc_id'])
#     set_processed_id = set()
#     for el in list_processed_id:
#         set_processed_id.add(str(el))
#     del list_processed_id
    
    
    moscow_time1 = datetime.datetime.now(pytz.timezone('Europe/Moscow'))
    time_now1 = moscow_time1.strftime("%d-%m-%Y %H:%M:%S")
    print("The script was launched at", time_now1)
    
    
    # doc_ids хранит список id текстов которые надо прогнать по модели (их берем из директории с текстами)
    doc_ids = get_doc_ids_from_dir(folder_of_texts)
    # создаем файл csv в который будем добавлять результаты
    create_csv(name_csv)
    # берем каждый id из списка и запускаем модель для текста
    i = 0
    for doc_id in doc_ids:
        d_id = doc_id.strip()
#         i += 1
#         if d_id in set_processed_id:
#             continue
        start_for_one_text(d_id, folder_of_texts, name_csv, model_size, model_name, mode, model, config, tokenizer, usegpu)
        i += 1
        print('\r', 'processing...', str(i), '/', len(doc_ids), end='')
    
    
    
    moscow_time2 = datetime.datetime.now(pytz.timezone('Europe/Moscow'))
    time_now2 = moscow_time2.strftime("%d-%m-%Y %H:%M:%S")
    print("\nThe script is completed at", time_now2)
    execution_time = moscow_time2 - moscow_time1
    print('Execution time =', execution_time)


if __name__ == "__main__":
    main()
