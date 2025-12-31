from functools import lru_cache
import tensorflow as tf
import numpy as np
import readability
# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType, DataStateType
from code_loader.contract.visualizer_classes import LeapText, LeapTextMask
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_input_encoder, tensorleap_gt_encoder, \
    tensorleap_metadata, tensorleap_preprocess, tensorleap_unlabeled_preprocess, tensorleap_custom_visualizer
from transformers import AlbertTokenizerFast
from typing import List, Dict, Union
from squad_albert.config import CONFIG
from squad_albert.data.preprocess import load_data
from squad_albert.decoders import get_decoded_tokens, tokenizer_decoder, context_polarity, context_subjectivity, \
    answer_decoder, tokens_decoder, tokens_question_decoder, tokens_context_decoder, segmented_tokens_decoder
from squad_albert.encoders import gt_index_encoder, gt_end_index_encoder, gt_start_index_encoder
from squad_albert.metrics import get_start_end_arrays
from squad_albert.utils.utils import get_context_positions


# -------------------------load_data--------------------------------
@tensorleap_preprocess()
def preprocess_response() -> List[PreprocessResponse]:
    train_idx, train_ds, val_idx, val_ds, _, _, enums_dic = load_data()
    train = PreprocessResponse(length=len(train_idx), data={'ds': train_ds, 'idx': train_idx, **enums_dic}, state=DataStateType.training)
    test = PreprocessResponse(length=len(val_idx), data={'ds': val_ds, 'idx': val_idx, **enums_dic}, state=DataStateType.validation)
    tokenizer = AlbertTokenizerFast.from_pretrained("vumichien/albert-base-v2-squad2")
    leap_binder.cache_container["tokenizer"] = tokenizer
    return [train, test]

@tensorleap_unlabeled_preprocess()
def preprocess_response_unlabeled() -> List[PreprocessResponse]:
    _, _, _, _, test_idx, test_ds, enums_dic = load_data()
    test = PreprocessResponse(length=len(test_idx), data={'ds': test_ds, 'idx': test_idx, **enums_dic}, state=DataStateType.test)
    return test


# ------- Inputs ---------

def convert_index(idx: int, preprocess: PreprocessResponse) -> int:
    if CONFIG['CHANGE_INDEX_FLAG']:
        return int(preprocess.data['idx'][idx])
    return idx


def get_inputs(idx: int, preprocess: PreprocessResponse) -> dict:
    x = preprocess.data['ds'][idx]
    tokenizer = get_tokenizer()
    inputs = tokenizer(
        x["question"],
        x["context"],
        return_tensors="tf",
        padding='max_length',
        max_length=CONFIG['max_sequence_length'],
        return_offsets_mapping=True
    )
    return inputs.data


@lru_cache()
def get_input_func(key: str):
    @tensorleap_input_encoder(key, channel_dim = -1)
    def input_func(idx: int, preprocess: PreprocessResponse):
        idx = convert_index(idx, preprocess)
        x = get_inputs(idx, preprocess)[key].numpy()
        x = x.squeeze()
        #VERIFY
        x = x.astype(np.float32)
        return x[:CONFIG['max_sequence_length']]

    input_func.__name__ = f"{key}"
    return input_func


# -------------------- gt  -------------------
@tensorleap_gt_encoder('indices_gt')
def gt_index_encoder_leap(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    idx = convert_index(idx, preprocess)
    sample = preprocess.data['ds'][idx]
    inputs = get_inputs(idx, preprocess)
    one_hot = gt_index_encoder(sample, inputs)
    return one_hot.astype(np.float32)


def get_tokenizer():  # V
    return leap_binder.cache_container["tokenizer"]


def gt_end_index_encoder_leap(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample = preprocess.data['ds'][idx]
    inputs = get_inputs(idx, preprocess)
    one_hot = gt_end_index_encoder(sample, inputs)
    return one_hot


def gt_start_index_encoder_leap(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample = preprocess.data['ds'][idx]
    inputs = get_inputs(idx, preprocess)
    one_hot = gt_start_index_encoder(sample, inputs)
    return one_hot


# ---------------------- meta_data  --------------------
@tensorleap_metadata('metadata_length')
def metadata_length(idx: int, preprocess: PreprocessResponse) -> Dict[str, int]:
    token_type_ids = get_input_func("token_type_ids")(idx, preprocess)
    context_start, context_end = get_context_positions(token_type_ids)
    context_length = int(context_end - context_start + 1)
    question_length = int(context_start - 1)

    res = {
        'context_length': context_length,
        'question_length': question_length
    }

    return res

@tensorleap_metadata('metadata_dict')
def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    res = metadata_length(idx, data)
    idx = convert_index(idx, data)
    for section in ["context", "question"]:
        stats_res = calc_txt_statistics(idx, data, section)
        for key in ['readability grades', 'sentence info', 'word usage', 'sentence beginnings']:
            key_stat = {f"{section}_{key}_{k}": v for k, v in stats_res[key].items()}
            res.update(key_stat)

    metadata_functions = {
        "answer_length": metadata_answer_length,
        "title": metadata_title,
        "title_idx": metadta_title_ids,
        "gt_text": metadata_gt_text,
        "context_polarity": metadata_context_polarity,
        "context_subjectivity": metadata_context_subjectivity
    }
    for func_name, func in metadata_functions.items():
        res[func_name] = func(idx, data)

    res.update(metadata_answer_relative(idx, data, res['context_length']))
    return res

def get_decoded_tokens_leap(input_ids: np.ndarray)->List[str]:
    tokenizer = get_tokenizer()
    decoded = get_decoded_tokens(input_ids, tokenizer)
    return decoded


def metadata_answer_length(idx: int, preprocess: PreprocessResponse) -> int:
    start_ind = np.argmax(gt_start_index_encoder_leap(idx, preprocess))
    end_ind = np.argmax(gt_end_index_encoder_leap(idx, preprocess))
    return int(end_ind - start_ind + 1)

def metadata_answer_relative(idx: int, preprocess: PreprocessResponse, context_length: int) -> int:
    res = {}
    start_ind = np.argmax(gt_start_index_encoder_leap(idx, preprocess))
    end_ind = np.argmax(gt_end_index_encoder_leap(idx, preprocess))
    res["answer_length_relative"] = int(end_ind - start_ind + 1)/context_length
    res["answer_start_relative"] = round(start_ind/context_length, 3)
    res["answer_end_relative"] = round(end_ind/context_length, 3)
    return res



def metadata_title(idx: int, preprocess: PreprocessResponse) -> str:
    return preprocess.data['ds'][idx]['title']


def metadta_title_ids(idx: int, preprocess: PreprocessResponse) -> int:
    return preprocess.data['title'][preprocess.data['ds'][idx]['title']].value


def metadata_gt_text(idx: int, preprocess: PreprocessResponse) -> str:
    sample = preprocess.data['ds'][idx]
    return sample['answers']['text'][0]

@tensorleap_metadata('is_truncated')
def metadata_is_truncated(idx: int, preprocess: PreprocessResponse) -> int:
    input_ids = get_input_func("input_ids")(idx, preprocess)
    tokenizer = get_tokenizer()
    decoded = tokenizer_decoder(tokenizer, input_ids.astype(np.int32))
    return int(len(decoded) > CONFIG['max_sequence_length'])


def metadata_context_polarity(idx: int, preprocess: PreprocessResponse) -> float:
    text = preprocess.data['ds'][idx]['context']
    val = context_polarity(text)
    return val


def metadata_context_subjectivity(idx: int, preprocess: PreprocessResponse) -> float:
    text = preprocess.data['ds'][idx]['context']
    val = context_subjectivity(text)
    return val

def calc_txt_statistics(idx: int, subset: PreprocessResponse, section='context') -> float:
    # idx = convert_index(idx, subset)
    text: str = subset.data['ds'][idx][section]
    results = readability.getmeasures(text, lang='en')
    return results


# ------- Visualizers  ---------
@tensorleap_custom_visualizer('new_answer_decoder', LeapDataType.Text)
def answer_decoder_leap(logits: tf.Tensor, input_ids: np.ndarray, token_type_ids, offset_mapping) -> LeapText:
    logits = np.squeeze(logits)
    input_ids = np.squeeze(input_ids)
    tokenizer = get_tokenizer()
    answer = answer_decoder(logits, input_ids, tokenizer)
    return LeapText(answer)

@tensorleap_custom_visualizer('preidction_indices', LeapDataType.Text)
def onehot_to_indices_pred(one_hot: np.ndarray) -> LeapText:
    start_logits, end_logits = get_start_end_arrays(one_hot)
    start_ind = int(tf.math.argmax(start_logits, axis=-1))
    end_ind = int(tf.math.argmax(end_logits, axis=-1))
    return LeapText([start_ind, end_ind])

@tensorleap_custom_visualizer('gt_indices', LeapDataType.Text)
def onehot_to_indices_gt(one_hot: np.ndarray) -> LeapText:
    start_logits, end_logits = get_start_end_arrays(one_hot)
    start_ind = int(tf.math.argmax(start_logits, axis=-1))
    end_ind = int(tf.math.argmax(end_logits, axis=-1))
    return LeapText([start_ind, end_ind])

@tensorleap_custom_visualizer('tokens_decoder', LeapDataType.Text)
def tokens_decoder_leap(input_ids: np.ndarray) -> LeapText:
    input_ids = np.squeeze(input_ids)
    decoded = get_decoded_tokens_leap(input_ids)
    decoded = tokens_decoder(decoded)
    return LeapText(decoded)

@tensorleap_custom_visualizer('tokens_question_decoder', LeapDataType.Text)
def tokens_question_decoder_leap(input_ids: np.ndarray, token_type_ids: np.ndarray) -> LeapText:
    input_ids = np.squeeze(input_ids)
    token_type_ids = np.squeeze(token_type_ids)
    tokenizer = get_tokenizer()
    decoded = tokens_question_decoder(input_ids, token_type_ids, tokenizer)
    return LeapText(decoded)

@tensorleap_custom_visualizer('tokens_context_decoder', LeapDataType.Text)
def tokens_context_decoder_leap(input_ids: np.ndarray, token_type_ids: np.ndarray) -> LeapText:
    input_ids = np.squeeze(input_ids)
    token_type_ids = np.squeeze(token_type_ids)
    tokenizer = get_tokenizer()
    decoded = tokens_context_decoder(input_ids, token_type_ids, tokenizer)
    return LeapText(decoded)

@tensorleap_custom_visualizer('segmented_tokens_decoder', LeapDataType.TextMask)
def segmented_tokens_decoder_leap(input_ids: np.ndarray, token_type_ids: np.ndarray, gt_logits: np.ndarray, pred_logits: np.ndarray) -> LeapTextMask:
    input_ids = np.squeeze(input_ids)
    token_type_ids = np.squeeze(token_type_ids)
    gt_logits = np.squeeze(gt_logits)
    pred_logits = np.squeeze(pred_logits)
    tokenizer = get_tokenizer()
    mask, text, labels = segmented_tokens_decoder(input_ids, token_type_ids, gt_logits, pred_logits, tokenizer)
    return LeapTextMask(mask.astype(np.uint8), text, labels)
