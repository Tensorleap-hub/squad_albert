import os
import tensorflow as tf
import numpy as np

from leap_binder import preprocess_response, get_input_func, gt_index_encoder_leap, metadata_is_truncated, \
    metadata_length, metadata_dict, calc_txt_statistics, tokens_decoder_leap, \
    tokens_question_decoder_leap, tokens_context_decoder_leap, segmented_tokens_decoder_leap, \
    answer_decoder_leap  # , get_analyzer
from squad_albert.loss import CE_loss
from squad_albert.metrics import exact_match_metric, dict_metrics, CE_start_index, CE_end_index
from squad_albert.utils.utils import get_readibility_score

from code_loader.helpers import visualize
from leap_binder import leap_binder


def check_custom_integration():
    check_generic = True
    plot_vis = True

    if check_generic:
        leap_binder.check()

    print("started custom tests")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/albert.h5'
    albert = tf.keras.models.load_model(os.path.join(dir_path, model_path))

    x = preprocess_response()
    for idx in range(5):
        input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping']
        inputs = []
        for key in input_keys:
            concat = get_input_func(key)(idx, x[0])
            inputs.append(concat)

        metadata_dict_all = metadata_dict(idx, x[0])
        y_pred = albert([inputs]).numpy()
        y_true = gt_index_encoder_leap(idx, x[0])

        #metrics
        f1 = dict_metrics(y_true, y_pred)
        ce_ls = CE_loss(y_true, y_pred)
        match_metric = exact_match_metric(y_true, y_pred)
        CE_start_in = CE_start_index(y_true, y_pred)
        CE_end_in = CE_end_index(y_true, y_pred)

        #vis
        tokens_decoder = tokens_decoder_leap(inputs[0])
        tokens_question_decoder = tokens_question_decoder_leap(inputs[0], inputs[1])
        tokens_context_decoder = tokens_context_decoder_leap(inputs[0], inputs[1])
        segmented_tokens_decoder = segmented_tokens_decoder_leap(inputs[0], inputs[1], y_true, y_pred)
        new_answer_decoder = answer_decoder_leap(y_pred, inputs[0], inputs[1], inputs[-1])
         #TODO
        # leap_binder.set_visualizer(onehot_to_indices, 'prediction_indices', LeapDataType.Text)
        # leap_binder.set_visualizer(onehot_to_indices, 'gt_indices', LeapDataType.Text)

        if plot_vis:
            visualize(tokens_decoder, 'tokens_decoder')
            visualize(tokens_question_decoder, 'tokens_question_decoder')
            visualize(tokens_context_decoder, 'tokens_context_decoder')
            visualize(segmented_tokens_decoder, 'segmented_tokens_decoder')
            visualize(new_answer_decoder, 'new_answer_decoder')


        # ------- Metadata ---------
        doct = metadata_dict(idx, x[0])
        length = metadata_length(idx, x[0])
        is_truncated = metadata_is_truncated(idx, x[0])

        is_truncated = metadata_is_truncated(idx, x[0])
        length = metadata_length(idx, x[0])
        meat_data_all = metadata_dict(idx, x[0])
        # for stat in ['num_letters', 'num_words', 'num_sentences', 'num_polysyllabic_words', 'avg_words_per_sentence',
        #              'avg_syllables_per_word']:
        #     state = get_statistics(stat, idx, x[0], 'context')
        #
        # for score in ['ari', 'coleman_liau', 'dale_chall', 'flesch', 'flesch_kincaid',
        #               'gunning_fog', 'linsear_write', 'smog', 'spache']:
        #     score = get_readibility_score(get_analyzer(idx, x[0]).__getattribute__(score))
    print("Custom tests finished successfully")


if __name__ == '__main__':
    check_custom_integration()
