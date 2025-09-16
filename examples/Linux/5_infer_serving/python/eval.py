#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse

import grpc
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
import tritonclient.grpc as grpcclient
import numpy as np
import time
import os
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit

import helpers.tokenization as tokenization
import helpers.data_processing as dp

FLAGS = None

    
def gen_test_data(vocab_file="vocab.txt", max_seq_length=384, doc_stride=128, max_query_length=64, batch_size=1):
    paragraph_text = "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps."
    question_text  = "What is TensorRT?"
    vocab_path = os.path.join(os.path.dirname(__file__), vocab_file)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=True)
    def question_features(tokens, question):
        # Extract features from the paragraph and question
        return dp.convert_example_to_features(tokens, question, tokenizer, max_seq_length, doc_stride, max_query_length)
    doc_tokens = dp.convert_doc_tokens(paragraph_text)
    features = question_features(doc_tokens, question_text)
    feature = features[0]
    if batch_size == 1:
        input_ids = np.expand_dims(feature.input_ids, 0)
        input_mask = np.expand_dims(feature.input_mask, 0)
        segment_ids = np.expand_dims(feature.segment_ids, 0)
    else:
        input_ids = np.stack([feature.input_ids] * batch_size)
        input_mask = np.stack([feature.input_mask] * batch_size)
        segment_ids = np.stack([feature.segment_ids] * batch_size)
    # input_ids = feature.input_ids
    # input_mask = feature.input_mask
    # segment_ids = feature.segment_ids
    def postprocess(output):
        import collections
        _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
                    "NetworkOutput",
                    ["start_logits", "end_logits", "feature_index"])
        networkOutputs = []
        output = output[0]
        networkOutputs.append(_NetworkOutput(
                    start_logits = np.array(output.squeeze()[:, 0]),
                    end_logits = np.array(output.squeeze()[:, 1]),
                    feature_index = 0
            ))
        prediction, nbest_json, scores_diff_json = dp.get_predictions(doc_tokens, features,
                    networkOutputs, 20, 30)
        return prediction
    return input_ids, input_mask, segment_ids, postprocess



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-m', type=str, required=False, default='bert-high', help='Model name')
    parser.add_argument('-t', type=int, required=False, default=1, help='execution time in seconds')
    parser.add_argument('-s', type=float, required=False, default=0, help='sleep time')
    parser.add_argument('-b', type=int, required=False, default=1, help='batch size')
    parser.add_argument('-o', type=str, required=False, default=None, help='output file')
    parser.add_argument('-r', type=int, required=False, default=1, help='last r seconds for performance record')
    FLAGS = parser.parse_args()

    eval_start_time = time.time()

    model_name = FLAGS.m
    model_version = ""
    batch_size = FLAGS.b

    # Test data
    input_ids, input_mask, segment_ids, postprocess = gen_test_data(batch_size=batch_size)
    # print("input_ids: {}\ninput_mask: {}\nsegment_ids: {}".format(input_ids.shape, input_mask.shape, segment_ids.shape))

    # exit(0)

    # Infer
    inputs = []
    inputs.append(grpcclient.InferInput('input_ids', input_ids.shape, "INT32"))
    inputs.append(grpcclient.InferInput('segment_ids', input_mask.shape, "INT32"))
    inputs.append(grpcclient.InferInput('input_mask', segment_ids.shape, "INT32"))
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(segment_ids)
    inputs[2].set_data_from_numpy(input_mask)
    # print(input_ids)
    # print(segment_ids)
    # print(input_mask)
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('cls_squad_logits'))
    # exit(0)
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        exit()
    
    # print(triton_client.get_model_metadata(model_name=model_name))
    request_timestamps = []
    while time.time() - eval_start_time < FLAGS.t:
        start = time.time()
        results = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            # callback=lambda result: print(result.as_numpy('cls_squad_logits'))
        )
        end = time.time()
        if FLAGS.o is None:
            print("time: {}".format(end - start))
        request_timestamps.append((start, end))
        time.sleep(FLAGS.s)

    prediction = postprocess(results.as_numpy('cls_squad_logits'))
    print("prediction: {}".format(prediction))


    final_latency = []
    last_request_time = request_timestamps[-1][0]
    for i in reversed(range(len(request_timestamps))):
        if last_request_time - request_timestamps[i][0] > FLAGS.r:
            break
        final_latency.append(request_timestamps[i][1] - request_timestamps[i][0])
    final_latency = sorted(final_latency)
    throughput = len(final_latency) / (FLAGS.r)
    # print cdf in format of: percentile, latency
    if FLAGS.o is not None:
        with open(FLAGS.o, "w") as f:
            for i in range(len(final_latency)):
                f.write("{}, {}\n".format(i / len(final_latency), final_latency[i] * 1000))
    print("{} throughput: {}, p50: {}, p90: {}, p99: {}".format(FLAGS.m, throughput, final_latency[int(len(final_latency) * 0.5)] * 1000, final_latency[int(len(final_latency) * 0.9)] * 1000, final_latency[int(len(final_latency) * 0.99)] * 1000))

    print("PASS")
