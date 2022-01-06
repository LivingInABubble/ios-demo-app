// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#import "InferenceModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

const int input_width = 640;
const int input_height = 640;
const int threshold = 0.5;


@implementation InferenceModule {
    @protected torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (NSArray<NSNumber*>*)detectImage:(void*)imageBuffer {
    try {
        at::Tensor tensor = torch::from_blob(imageBuffer, { 3, input_width, input_height }, at::kFloat);
        c10::InferenceMode guard;

        std::vector<torch::Tensor> v;
        v.push_back(tensor);


        CFTimeInterval startTime = CACurrentMediaTime();
        auto outputTuple = _impl.forward({
            tensor * 255.0
        }).toTuple();
        CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
        NSLog(@"inference time:%f", elapsedTime);


        auto boxesTensor = outputTuple->elements()[0].toTensor();
        auto labelsTensor = outputTuple->elements()[1].toTensor();
        auto masksTensor = outputTuple->elements()[2].toTensor();
        auto scoresTensor = outputTuple->elements()[3].toTensor();

        float* boxesBuffer = boxesTensor.data_ptr<float>();
        if (!boxesBuffer) {
            return nil;
        }
        float* scoresBuffer = scoresTensor.data_ptr<float>();
        if (!scoresBuffer) {
            return nil;
        }
        int64_t* labelsBuffer = labelsTensor.data_ptr<int64_t>();
        if (!labelsBuffer) {
            return nil;
        }

        NSMutableArray* results = [[NSMutableArray alloc] init];
        long num = scoresTensor.numel();
        for (int i = 0; i < num; i++) {
            if (scoresBuffer[i] < threshold)
                continue;

            [results addObject:@(boxesBuffer[4 * i])];
            [results addObject:@(boxesBuffer[4 * i + 1])];
            [results addObject:@(boxesBuffer[4 * i + 2])];
            [results addObject:@(boxesBuffer[4 * i + 3])];
            [results addObject:@(scoresBuffer[i])];
            [results addObject:@(labelsBuffer[i])];
        }

        return [results copy];

    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end
