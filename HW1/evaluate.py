import time
import json
from functools import lru_cache

import cv2
import numpy as np
from scipy.spatial.distance import cdist

# from gradescope_utils.autograder_utils.decorators import partial_credit, number, leaderboard

from ticket_to_ride import predict_image


# class TestTicketToRide(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     @partial_credit(10.0)
#     @number("test case 0")
#     def test_case_0(self, set_score=None):
#         set_score(evaluate_test_case(0))
#
#     @partial_credit(10.0)
#     @number("test case 1")
#     def test_case_1(self, set_score=None):
#         set_score(evaluate_test_case(1))
#
#     @partial_credit(10.0)
#     @number("test case 2")
#     def test_case_2(self, set_score=None):
#         set_score(evaluate_test_case(2))
#
#     @partial_credit(10.0)
#     @number("test case 3")
#     def test_case_3(self, set_score=None):
#         set_score(evaluate_test_case(3))
#
#     @partial_credit(10.0)
#     @number("test case 4")
#     def test_case_4(self, set_score=None):
#         set_score(evaluate_test_case(4))
#
#     @partial_credit(10.0)
#     @number("test case 5")
#     def test_case_5(self, set_score=None):
#         set_score(evaluate_test_case(5))
#
#     @partial_credit(10.0)
#     @number("test case 6")
#     def test_case_6(self, set_score=None):
#         set_score(evaluate_test_case(6))
#
#     @partial_credit(10.0)
#     @number("test case 7")
#     def test_case_7(self, set_score=None):
#         set_score(evaluate_test_case(7))
#
#     @partial_credit(10.0)
#     @number("test case 8")
#     def test_case_8(self, set_score=None):
#         set_score(evaluate_test_case(8))
#
#     @partial_credit(10.0)
#     @number("test case 9")
#     def test_case_9(self, set_score=None):
#         set_score(evaluate_test_case(9))
#
#     @leaderboard("final score")
#     def test_leaderboard_final(self, set_leaderboard_value=None):
#         set_leaderboard_value(np.sum([evaluate_test_case(i) for i in range(10)]))


COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAIN_CASES = {i: fname for i, fname in enumerate(('all', 'black_blue_green', 'black_red_yellow',
                                                   'red_green_blue_inaccurate', 'red_green_blue'))}


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_test_case(i):
    time_eval_start = time.time()

    predict_fn = predict_image

    gt_centers = load_json(f'train/{TRAIN_CASES[i]}_centers.json')
    gt_n_trains = load_json(f'train/{TRAIN_CASES[i]}_n_trains.json')
    gt_scores = load_json(f'train/{TRAIN_CASES[i]}_scores.json')

    img = cv2.imread(f'train/{TRAIN_CASES[i]}.jpg')

    time_predict_start = time.time()
    centers, n_trains, scores = predict_fn(img)
    time_predict_end = time.time()

    score1 = score_centers(gt_centers, centers)
    score2 = score_n_trains(gt_n_trains, n_trains)
    score3 = score_scores(gt_scores, scores)
    score = np.round(10 * (0.25 * score1 + 0.5 * score2 + 0.25 * score3), decimals=2)

    time_eval_end = time.time()

    return score, time_predict_end - time_predict_start, time_eval_end - time_eval_start


def score_centers(centers_true, centers_pred):
    _max_hit_dist = 25
    # 1 pred couldn't hit >1 true because of the min dist
    # so check only 2 conditions: (is the closest pred) & (is closer than `_max_hit_dist`)
    dist_matrix = cdist(np.float32(centers_pred), np.float32(centers_true))
    dist_hit_condition = dist_matrix < _max_hit_dist
    pred_argmin_idxs = np.ravel_multi_index(np.array([np.argmin(dist_matrix, axis=0), np.arange(dist_matrix.shape[1])]),
                                            dims=dist_matrix.shape)
    pred_argmin_condition = np.ravel(np.zeros(dist_matrix.shape))
    pred_argmin_condition[pred_argmin_idxs] = 1
    pred_argmin_condition = np.bool_(np.reshape(pred_argmin_condition, dist_matrix.shape))
    hit_matrix = dist_hit_condition & pred_argmin_condition

    tp = np.sum(hit_matrix)
    fp = hit_matrix.shape[0] - tp
    fn = hit_matrix.shape[1] - tp
    return tp / (tp + fp + fn)


def score_n_trains(n_trains_true, n_trains_pred):
    # +-1 = 1.0; +-2 = 0.8; +-3 = 0.6; +-4 = 0.4; +-5 = 0.2; >+-6 = 0 :: for each player -> average
    return np.mean([np.clip(1.2 - 0.2 * np.abs(n_trains_true[c] - n_trains_pred[c]), 0, 1) for c in COLORS])


def score_scores(scores_true, scores_pred):
    # +-1 = 1.0; +-3 = 0.8; +-5 = 0.6; +-7 = 0.4; +-9 = 0.2; >+-11 = 0 :: for each player -> average
    return np.mean([np.clip(1.1 - 0.1 * np.abs(scores_true[c] - scores_pred[c]), 0, 1) for c in COLORS])


def main():
    final_score = 0
    total_eval_time = 0
    for i, case in TRAIN_CASES.items():
        score, t_pred, t_eval = evaluate_test_case(i)
        final_score += score
        total_eval_time += t_eval

        print(f'### train case `{case}` ###', flush=True)
        print(f'Score = {score:.2f}', flush=True)
        print(f'Prediction time = {t_pred:.2f}s. Evaluation overhead = {t_eval - t_pred:.2f}s.', flush=True)
        print()

    print(f'### Summary ###', flush=True)
    print(f'Final train score = {final_score:.2f} out of 50.00', flush=True)
    print(f'Total evaluation time = {total_eval_time:.2f}s for 5 images.', flush=True)
    print(f'The platform limit is {40*60}s for 10 images.')


if __name__ == '__main__':
    main()
