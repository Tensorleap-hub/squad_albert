import os
import tensorflow as tf
import numpy as np
from code_loader.contract.datasetclasses import PredictionTypeHandler
from leap_binder import preprocess_response, get_input_func, gt_index_encoder_leap, metadata_is_truncated, \
    metadata_length, metadata_dict, tokens_decoder_leap, \
    tokens_question_decoder_leap, tokens_context_decoder_leap, segmented_tokens_decoder_leap, \
    answer_decoder_leap
from squad_albert.loss import CE_loss
from squad_albert.metrics import exact_match_metric, dict_metrics, CE_start_index, CE_end_index
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from code_loader.plot_functions.visualize import visualize

prediction_type1 = PredictionTypeHandler('classes', ["start", "end"])

@tensorleap_load_model([prediction_type1])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/albert_converted.h5'
    cnn = tf.keras.models.load_model(os.path.join(dir_path, model_path))
    return cnn

@tensorleap_integration_test()
def check_custom_integration(idx, subset):
    plot_vis = True
    input_keys = ['input_ids', 'token_type_ids', 'attention_mask']
    inputs = []
    for key in input_keys:
        concat = get_input_func(key)(idx, subset)
        inputs.append(concat)
    albert = load_model()
    y_pred = albert(inputs)
    metadata_dict_all = metadata_dict(idx, subset)
    y_true = gt_index_encoder_leap(idx, subset)
    #
    # #metrics
    f1 = dict_metrics(y_true, y_pred)
    ce_ls = CE_loss(y_true, y_pred)
    match_metric = exact_match_metric(y_true, y_pred)
    CE_start_in = CE_start_index(y_true, y_pred)
    CE_end_in = CE_end_index(y_true, y_pred)
    #
    # #vis
    tokens_decoder = tokens_decoder_leap(inputs[0])
    tokens_question_decoder = tokens_question_decoder_leap(inputs[0], inputs[1])
    tokens_context_decoder = tokens_context_decoder_leap(inputs[0], inputs[1])
    segmented_tokens_decoder = segmented_tokens_decoder_leap(inputs[0], inputs[1], y_true, y_pred)
    new_answer_decoder = answer_decoder_leap(y_pred, inputs[0], inputs[1], inputs[-1])
    if plot_vis:
        visualize(tokens_decoder, 'tokens_decoder')
        visualize(tokens_question_decoder, 'tokens_question_decoder')
        visualize(tokens_context_decoder, 'tokens_context_decoder')
        visualize(segmented_tokens_decoder, 'segmented_tokens_decoder')
        visualize(new_answer_decoder, 'new_answer_decoder')
    #
    # # ------- Metadata ---------
    doct = metadata_dict(idx, subset)
    length = metadata_length(idx, subset)
    is_truncated = metadata_is_truncated(idx, subset)
    #
    meat_data_all = metadata_dict(idx, subset)
    print("Custom tests finished successfully")


if __name__ == '__main__':
    x = preprocess_response()
    for idx in range(5):
        check_custom_integration(idx, x[0])
