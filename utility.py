import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from operator import itemgetter
from scipy.stats import entropy
from math import log

top1 = 1
top2 = 5
top3 = 10
top4 = 15


# calculate NDCG@k
def NDCG_at_k(predicted_list, ground_truth, k):
    dcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(predicted_list[:k])]
    dcg = np.sum(dcg_value)
    if len(ground_truth) < k:
        ground_truth += [0 for i in range(k - len(ground_truth))]
    idcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(ground_truth[:k])]
    idcg = np.sum(idcg_value)
    return dcg / idcg


# calculate precision@k, recall@k, NDCG@k, where k = 1,5,10,15
def user_precision_recall_ndcg(new_user_prediction, test):
    dcg_list = []

    # compute the number of true positive items at top k
    count_1, count_5, count_10, count_15 = 0, 0, 0, 0
    for i in range(15):# if the item in test set fall to top 1,5,10,15 positions --> add to the the count_1 number
        if i == 0 and new_user_prediction[0][i] in test:
            count_1 = 1.0
        if i < 5 and new_user_prediction[0][i] in test:
            count_5 += 1.0
        if i < 10 and new_user_prediction[0][i] in test:
            count_10 += 1.0
        if new_user_prediction[0][i] in test:
            count_15 += 1.0
            dcg_list.append(1)
        else:
            dcg_list.append(0)
            #dcg list returns the topk list (for 15 items), position with 1 means there is an item in test list makes it to top-i position

    # calculate NDCG@k
    idcg_list = [1 for i in range(len(test))] #generate a list idcg with length = utest and all value = 1
    ndcg_tmp_1 = NDCG_at_k(dcg_list, idcg_list, 1)
    ndcg_tmp_5 = NDCG_at_k(dcg_list, idcg_list, 5)
    ndcg_tmp_10 = NDCG_at_k(dcg_list, idcg_list, 10)
    ndcg_tmp_15 = NDCG_at_k(dcg_list, idcg_list, 15)

    # precision@k no of test items in top k divided by no. of relevant items
    precision_1 = count_1
    precision_5 = count_5 / 5.0
    precision_10 = count_10 / 10.0
    precision_15 = count_15 / 15.0

    l = len(test)
    if l == 0:
        l = 1
    # recall@k % movie in the test set that make it to topk
    recall_1 = count_1 / l
    recall_5 = count_5 / l
    recall_10 = count_10 / l
    recall_15 = count_15 / l

    # return precision, recall, ndcg_tmp
    return np.array([precision_1, precision_5, precision_10, precision_15]), \
           np.array([recall_1, recall_5, recall_10, recall_15]), \
           np.array([ndcg_tmp_1, ndcg_tmp_5, ndcg_tmp_10, ndcg_tmp_15])


# calculate precision@k, recall@k, NDCG@k, where k = 1,5,10,15
def user_recall(new_user_prediction, test, item_idd_genre_list, key_genre):

    #initialize dictionary to store result
    test_dict = dict()
    count_1_dict = dict()
    count_5_dict = dict()
    count_10_dict = dict()
    count_15_dict = dict()
    recall_1_dict = dict()
    recall_5_dict = dict()
    recall_10_dict = dict()
    recall_15_dict = dict()

    #initialize dictionary to store result, for each key genre
    for k in key_genre:
        test_dict[k] = 0.0
        count_1_dict[k] = 0.0
        count_5_dict[k] = 0.0
        count_10_dict[k] = 0.0
        count_15_dict[k] = 0.0
        recall_1_dict[k] = 0.0
        recall_5_dict[k] = 0.0
        recall_10_dict[k] = 0.0
        recall_15_dict[k] = 0.0


    #for each item in the test list, if the item has the key genre concerned --> count and append to the test_dict dictionary
    #test dict will return the number of movie in each key genre for each test list (list of movies associated with an user in the test data)
    for t in test:
        gl = item_idd_genre_list[t]
        for g in gl:
            if g in key_genre:
                test_dict[g] += 1.0

    #for each item in top 15, if they are in the test set, and belong to one of the key genre specified,
    #we append them to the dictionary that track no. of movie in key_genre that make it to topk
    for i in range(top4):
        item_id = int(new_user_prediction[1][i])
        if item_id in test:
            gl = item_idd_genre_list[item_id]
            if i < 10:
                for g in gl:
                    if g in key_genre:
                        count_10_dict[g] += 1.0
                if i < 5:
                    for g in gl:
                        if g in key_genre:
                            count_5_dict[g] += 1.0
                    if i < 1:
                        for g in gl:
                            if g in key_genre:
                                count_1_dict[g] += 1.0
            for g in gl:
                if g in key_genre:
                    count_15_dict[g] += 1.0

    # recall@k
    for k in key_genre:
        l = test_dict[k]
        #check the number of key genre movie in the top k list, if a key genre does not appear, then set l = 1 and tmp = -1
        if l == 0:
            tmp = -1
            l = 1
        else:
            tmp = 0
        #recall of a genre = number of movie in topk belong to that genre over total number of movie belongs to that genre in test list
        recall_1_dict[k] = count_1_dict[k] / l + tmp
        recall_5_dict[k] = count_5_dict[k] / l + tmp
        recall_10_dict[k] = count_10_dict[k] / l + tmp
        recall_15_dict[k] = count_15_dict[k] / l + tmp

    # return precision, recall, ndcg_tmp
    return recall_1_dict, recall_5_dict, recall_10_dict, recall_15_dict, \
           count_1_dict, count_5_dict, count_10_dict, count_15_dict, test_dict


# calculate the metrics of the result
def test_model_all(Rec, test_df, train_df):
    Rec = copy.copy(Rec)
    precision = np.array([0.0, 0.0, 0.0, 0.0])
    recall = np.array([0.0, 0.0, 0.0, 0.0])
    ndcg = np.array([0.0, 0.0, 0.0, 0.0])
    user_num = Rec.shape[0]

    for i in range(user_num):
        like_item = (train_df.loc[train_df['user_id'] == i, 'item_id']).tolist()
        Rec[i, like_item] = -100000.0

    for u in range(user_num):  # iterate each user
        u_test = (test_df.loc[test_df['user_id'] == u, 'item_id']).tolist()
        u_pred = Rec[u, :]


        #while they combine 2 tensors here?
        top15_item_idx_no_train = np.argpartition(u_pred, -15)[-15:] #take out top 15 movies only
        top15 = (np.array([top15_item_idx_no_train, u_pred[top15_item_idx_no_train]])).T #u_pred[top15_item_idx_no_train]= np.partition(u_pred, -1 * top4)[-1 * top4:]
        top15 = sorted(top15, key=itemgetter(1), reverse=True)

        # calculate the metrics
        if not len(u_test) == 0:
            precision_u, recall_u, ndcg_u = user_precision_recall_ndcg(top15, u_test)
            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
        else:
            user_num -= 1

    # compute the average over all users
    precision /= user_num
    recall /= user_num
    ndcg /= user_num
    print('precision_1\t[%.7f],\t||\t precision_5\t[%.7f],\t||\t precision_10\t[%.7f],\t||\t precision_15\t[%.7f]' \
          % (precision[0], precision[1], precision[2], precision[3]))
    print('recall_1   \t[%.7f],\t||\t recall_5   \t[%.7f],\t||\t recall_10   \t[%.7f],\t||\t recall_15   \t[%.7f]' \
          % (recall[0], recall[1], recall[2], recall[3]))
    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]) if not precision[0] + recall[
        0] == 0 else 0
    f_measure_5 = 2 * (precision[1] * recall[1]) / (precision[1] + recall[1]) if not precision[1] + recall[
        1] == 0 else 0
    f_measure_10 = 2 * (precision[2] * recall[2]) / (precision[2] + recall[2]) if not precision[2] + recall[
        2] == 0 else 0
    f_measure_15 = 2 * (precision[3] * recall[3]) / (precision[3] + recall[3]) if not precision[3] + recall[
        3] == 0 else 0
    print('f_measure_1\t[%.7f],\t||\t f_measure_5\t[%.7f],\t||\t f_measure_10\t[%.7f],\t||\t f_measure_15\t[%.7f]' \
          % (f_measure_1, f_measure_5, f_measure_10, f_measure_15))
    f_score = [f_measure_1, f_measure_5, f_measure_10, f_measure_15]
    print('ndcg_1     \t[%.7f],\t||\t ndcg_5     \t[%.7f],\t||\t ndcg_10     \t[%.7f],\t||\t ndcg_15     \t[%.7f]' \
          % (ndcg[0], ndcg[1], ndcg[2], ndcg[3]))
    return precision, recall, f_score, ndcg


def negative_sample(train_df, num_user, num_item, neg):
    user = []
    item_pos = []
    item_neg = []
    item_set = set(range(num_item))
    for i in range(num_user):
        like_item = (train_df.loc[train_df['user_id'] == i, 'item_id']).tolist()
        unlike_item = list(item_set - set(like_item))
        if len(unlike_item) < neg:
            tmp_neg = len(unlike_item)
        else:
            tmp_neg = neg
        for l in like_item: #for like item, take out neg (default) negative items to compare
            neg_samples = (np.random.choice(unlike_item, size=tmp_neg, replace=False)).tolist()
            user += [i] * tmp_neg
            item_pos += [l] * tmp_neg
            item_neg += neg_samples
    num_sample = len(user)
    return num_sample, np.array(user).reshape((num_sample, 1)),\
           np.array(item_pos).reshape((num_sample, 1)), np.array(item_neg).reshape((num_sample, 1))


def ranking_analysis(Rec, test_df, train_df, key_genre, item_idd_genre_list, user_genre_count):
    Rec = copy.copy(Rec)

    count1_dict = dict()
    count5_dict = dict()
    count10_dict = dict()
    count15_dict = dict()
    test_count = dict()

    recall1_dict = dict()
    recall5_dict = dict()
    recall10_dict = dict()
    recall15_dict = dict()
    user_count_dict = dict()

    num_user = Rec.shape[0]
    num_item = Rec.shape[1]

    top1_dict = dict()
    top5_dict = dict()
    top10_dict = dict()
    top15_dict = dict()
    avg_top1_dict = dict()
    avg_top5_dict = dict()
    avg_top10_dict = dict()
    avg_top15_dict = dict()
    tmp_top1_dict = dict()
    tmp_top5_dict = dict()
    tmp_top10_dict = dict()
    tmp_top15_dict = dict()
    genre_rank_count = dict()
    rank_count = np.ones(num_item) * 1e-10

    genre_to_be_rank = dict()

    for k in key_genre:
        genre_rank_count[k] = np.zeros(num_item)
        top1_dict[k] = 0.0
        top5_dict[k] = 0.0
        top10_dict[k] = 0.0
        top15_dict[k] = 0.0
        avg_top1_dict[k] = 0.0
        avg_top5_dict[k] = 0.0
        avg_top10_dict[k] = 0.0
        avg_top15_dict[k] = 0.0
        tmp_top1_dict[k] = 0.0
        tmp_top5_dict[k] = 0.0
        tmp_top10_dict[k] = 0.0
        tmp_top15_dict[k] = 0.0

        recall1_dict[k] = 0.0
        recall5_dict[k] = 0.0
        recall10_dict[k] = 0.0
        recall15_dict[k] = 0.0
        user_count_dict[k] = 0.0

        count1_dict[k] = 0.0
        count5_dict[k] = 0.0
        count10_dict[k] = 0.0
        count15_dict[k] = 0.0

        genre_to_be_rank[k] = 0.0
        test_count[k] = 0.0

    for u in range(num_user):
        #for each user list all the items that he likes, mark that to -10000
        like_item = (train_df.loc[train_df['user_id'] == u, 'item_id']).tolist()
        Rec[u, like_item] = -100000.0

    for u in range(num_user):  # iterate each user
        #extract item that user actually match during testing, extract the prediction
        u_test = (test_df.loc[test_df['user_id'] == u, 'item_id']).tolist()
        u_pred = Rec[u, :]


        #get top 15 items
        top15_item_idx_no_train = np.argpartition(u_pred, -1 * top4)[-1 * top4:]
        top15 = (np.array([top15_item_idx_no_train, u_pred[top15_item_idx_no_train]])).T
        top15 = sorted(top15, key=itemgetter(1), reverse=True)

        # calculate the recall for different genres
        if not len(u_test) == 0:
            recall_1_tmp_dict, recall_5_tmp_dict, recall_10_tmp_dict, recall_15_tmp_dict, \
            count_1_tmp_dict, count_5_tmp_dict, count_10_tmp_dict, count_15_tmp_dict, test_tmp_dict\
                = user_recall(top15, u_test, item_idd_genre_list, key_genre)
            for k in key_genre:
                #for each item in top 15, if they are in the test set, and belong to one of the key genre specified,
                #we append them to the dictionary that track no. of movie in key_genre that make it to topk
                count1_dict[k] += count_1_tmp_dict[k]
                count5_dict[k] += count_5_tmp_dict[k]
                count10_dict[k] += count_10_tmp_dict[k]
                count15_dict[k] += count_15_tmp_dict[k]
                test_count[k] += test_tmp_dict[k]
                if recall_1_tmp_dict[k] == -1:
                    continue
                #recall of a genre = number of movie in topk belong to that genre over total number of movie belongs to that genre in test list
                recall1_dict[k] += recall_1_tmp_dict[k]
                recall5_dict[k] += recall_5_tmp_dict[k]
                recall10_dict[k] += recall_10_tmp_dict[k]
                recall15_dict[k] += recall_15_tmp_dict[k]
                user_count_dict[k] += 1.0

        # calculate ranking probability
        rank = 1
        for r in top15[0]:#top 15 items sorted
        # extract the genre of each topk movie
            gl = item_idd_genre_list[int(r)]
            for g in gl:
                if g in key_genre:
                    #if the movie belongs to a key genre ==> add 1 to the dict
                    genre_rank_count[g][rank - 1] += 1.0 #size = no. of items
                    rank_count[rank - 1] += 1.0
                    if rank <= top4:
                        tmp_top15_dict[g] += 1.0 #no of movie in top15 for each key genre
                        if rank <= top3:
                            tmp_top10_dict[g] += 1.0
                            if rank <= top2:
                                tmp_top5_dict[g] += 1.0
                                if rank <= top1:
                                    tmp_top1_dict[g] += 1.0
            rank += 1 #15 rank
        for k in key_genre:
            top1_dict[k] += tmp_top1_dict[k] #no of movie in key_genre k in topk for an user u
            top5_dict[k] += tmp_top5_dict[k]
            top10_dict[k] += tmp_top10_dict[k]
            top15_dict[k] += tmp_top15_dict[k]
            avg_top1_dict[k] += (1.0 * tmp_top1_dict[k] / user_genre_count[u][k]) #user_genre_count = no. of more in each key_genre that the user has not intereacted with in training
            avg_top5_dict[k] += (1.0 * tmp_top5_dict[k] / user_genre_count[u][k]) #fraction of movies for each key genre appear in top K/ total movies of that genre that user u has not interacted with before
            avg_top10_dict[k] += (1.0 * tmp_top10_dict[k] / user_genre_count[u][k])
            avg_top15_dict[k] += (1.0 * tmp_top15_dict[k] / user_genre_count[u][k])
            tmp_top1_dict[k] = 0.0 #reset tmp dict
            tmp_top5_dict[k] = 0.0
            tmp_top10_dict[k] = 0.0
            tmp_top15_dict[k] = 0.0

            genre_to_be_rank[k] += user_genre_count[u][k] #total number of movies that all users have not interacted with during training
            #user_genre_count = no. of movies in each key_genre that the user has not intereacted with in training

    # compute the average recall for different genres, and print out the results
    for k in key_genre:
        #count1_dict track no. of movie in key_genre that make it to topk (in topk list predicted)
        count1_dict[k] /= test_count[k]
        count5_dict[k] /= test_count[k]
        count10_dict[k] /= test_count[k]
        count15_dict[k] /= test_count[k]    #test dict will return the number of movie in each key genre for each test list (list of movies associated with an user in the test data)
        recall1_dict[k] /= user_count_dict[k] #no of user interact with at least an item from a key_genre (divided for average recall for user)
        recall5_dict[k] /= user_count_dict[k]
        recall10_dict[k] /= user_count_dict[k]
        recall15_dict[k] /= user_count_dict[k]
    print('')
    print('#' * 100)
    print('# System-level Recall:')
    print('# \t\t\tRecall@%d\tRecall@%d\tRecall@%d\tRecall@%d' % (top1, top2, top3, top4))
    for k in key_genre:
        print('# ' + k + '\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' % (count1_dict[k], count5_dict[k], count10_dict[k], count15_dict[k]))
    recall1_std = relative_std(count1_dict) #count1_dict = % movie in a key_genre in test list make it to top1
    recall5_std = relative_std(count5_dict)
    recall10_std = relative_std(count10_dict)
    recall15_std = relative_std(count15_dict)
    print('# relative std\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' % (recall1_std, recall5_std, recall10_std, recall15_std))
    print('#' * 100)

    print('# User-level Recall:')
    print('# \t\t\tRecall@%d\tRecall@%d\tRecall@%d\tRecall@%d' % (top1, top2, top3, top4))
    for k in key_genre:
        print('# ' + k + '\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' % (
            recall1_dict[k], recall5_dict[k], recall10_dict[k], recall15_dict[k]))
    recall1_std_user = relative_std(recall1_dict)#recall of a genre = number of movie in topk belong to that genre over total number of movie belongs to that genre in test list
    recall5_std_user = relative_std(recall5_dict)
    recall10_std_user = relative_std(recall10_dict)
    recall15_std_user = relative_std(recall15_dict)
    print('# relative std\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' % (recall1_std_user, recall5_std_user, recall10_std_user, recall15_std_user))
    print('#' * 100)

    # calculate the average genre ranking probability across users, and calculate system-level ranking probability
    for k in key_genre:
        avg_top1_dict[k] /= num_user#avg probability in top k of a item belonging to a key genre for each user
        avg_top5_dict[k] /= num_user
        avg_top10_dict[k] /= num_user
        avg_top15_dict[k] /= num_user

        top1_dict[k] /= genre_to_be_rank[k] #total movies belongs to a keygenre belongs to top k /#total number of movies that all users have not interacted with during training
        top5_dict[k] /= genre_to_be_rank[k]
        top10_dict[k] /= genre_to_be_rank[k]
        top15_dict[k] /= genre_to_be_rank[k]

    print('# System-level top ranking probability:')
    print('# \t\t\t@%d\t\t@%d\t\t@%d\t\t@%d' % (top1, top2, top3, top4))
    for k in key_genre:
        print('# ' + k + '\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' % (top1_dict[k], top5_dict[k], top10_dict[k], top15_dict[k]))
    top1_std = relative_std(top1_dict)
    top5_std = relative_std(top5_dict)
    top10_std = relative_std(top10_dict)
    top15_std = relative_std(top15_dict)
    print('# relative std\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' % (top1_std, top5_std, top10_std, top15_std))
    print('#' * 100)

    print('# User-level top ranking probability:')
    print('# \t\t\t@%d\t\t@%d\t\t@%d\t\t@%d' % (top1, top2, top3, top4))
    for k in key_genre:
        print('# ' + k + '\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' % (avg_top1_dict[k], avg_top5_dict[k], avg_top10_dict[k], avg_top15_dict[k]))
    top1_std_user = relative_std(avg_top1_dict)
    top5_std_user = relative_std(avg_top5_dict)
    top10_std_user = relative_std(avg_top10_dict)
    top15_std_user = relative_std(avg_top15_dict)
    print('# relative std\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f' % (top1_std_user, top5_std_user, top10_std_user, top15_std_user))
    print('#' * 100)

    return np.array([top1_std, top5_std, top10_std, top15_std]), np.array([recall1_std, recall5_std, recall10_std, recall15_std])


def relative_std(dictionary):
    tmp = []
    for key, value in sorted(dictionary.items(), key = lambda x: x[0]):
        tmp.append(value)
    rstd = np.std(tmp) / (np.mean(tmp) + 1e-10)
    return rstd
