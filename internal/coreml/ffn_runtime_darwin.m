//go:build darwin && cgo && coreml

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "ffn_runtime.h"

typedef struct {
	MLModel *model;
	MLMultiArray *input;
	MLMultiArray *output;
	MLDictionaryFeatureProvider *features;
	MLPredictionOptions *options;
	int tokens;
	int dimension;
} GLLMCoreMLFFN;

static void gllm_ffn_error(char **error, NSString *message) {
	if (error != NULL) *error = strdup(message.UTF8String ?: "native FFN operation failed");
}

static MLModelConfiguration *gllm_ffn_configuration(int compute_units) {
	MLModelConfiguration *configuration = [[[MLModelConfiguration alloc] init] autorelease];
	switch (compute_units) {
		case 0:
			if (@available(macOS 13.0, iOS 16.0, *)) configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
			else return nil;
			break;
		case 1: configuration.computeUnits = MLComputeUnitsCPUOnly; break;
		case 2: configuration.computeUnits = MLComputeUnitsCPUAndGPU; break;
		case 3: configuration.computeUnits = MLComputeUnitsAll; break;
		default: return nil;
	}
	return configuration;
}

void gllm_coreml_ffn_release(void *ptr) {
	if (ptr == NULL) return;
	@autoreleasepool {
		GLLMCoreMLFFN *m = ptr;
		[m->features release];
		[m->options release];
		[m->input release];
		[m->output release];
		[m->model release];
		free(m);
	}
}

static bool gllm_ffn_fixed_shape(MLMultiArrayConstraint *constraint) {
	MLMultiArrayShapeConstraint *shape = constraint.shapeConstraint;
	if (shape == nil || shape.type == MLMultiArrayShapeConstraintTypeUnspecified) return true;
	if (shape.type == MLMultiArrayShapeConstraintTypeEnumerated) {
		return shape.enumeratedShapes.count == 1 && [shape.enumeratedShapes[0] isEqualToArray:constraint.shape];
	}
	if (shape.type == MLMultiArrayShapeConstraintTypeRange && shape.sizeRangeForDimension.count == constraint.shape.count) {
		for (NSUInteger i = 0; i < constraint.shape.count; i++) {
			NSRange range = shape.sizeRangeForDimension[i].rangeValue;
			if (range.length != 1 || range.location != constraint.shape[i].unsignedIntegerValue) return false;
		}
		return true;
	}
	return false;
}

static bool gllm_ffn_shape(MLFeatureDescription *feature, int *tokens, int *dimension) {
	if (feature == nil || feature.type != MLFeatureTypeMultiArray || feature.isOptional) return false;
	MLMultiArrayConstraint *constraint = feature.multiArrayConstraint;
	NSArray<NSNumber *> *shape = constraint.shape;
	if (constraint.dataType != MLMultiArrayDataTypeFloat32 || shape.count != 2 || !gllm_ffn_fixed_shape(constraint)) return false;
	long long rows = shape[0].longLongValue, columns = shape[1].longLongValue;
	if (rows < 1 || rows > GLLM_COREML_FFN_MAX_TOKENS || columns < 1 || columns > GLLM_COREML_FFN_MAX_DIMENSION ||
		columns > GLLM_COREML_FFN_MAX_ELEMENTS / rows) return false;
	*tokens = (int)rows;
	*dimension = (int)columns;
	return true;
}

void *gllm_coreml_ffn_open(const char *path, int compute_units, GLLMCoreMLFFNInfo *info, char **error) {
	if (error != NULL) *error = NULL;
	if (info != NULL) memset(info, 0, sizeof(*info));
	if (path == NULL) {
		gllm_ffn_error(error, @"compiled FFN path is required");
		return NULL;
	}
	GLLMCoreMLFFN *m = NULL;
	@autoreleasepool {
		@try {
			m = calloc(1, sizeof(*m));
			if (m == NULL) {
				gllm_ffn_error(error, @"allocate FFN model state");
				return NULL;
			}
			MLModelConfiguration *configuration = gllm_ffn_configuration(compute_units);
			if (configuration == nil) {
				gllm_ffn_error(error, @"unknown or unavailable compute units");
				gllm_coreml_ffn_release(m);
				return NULL;
			}
			NSError *native_error = nil;
			NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path] isDirectory:YES];
			m->model = [[MLModel modelWithContentsOfURL:url configuration:configuration error:&native_error] retain];
			if (m->model == nil) {
				gllm_ffn_error(error, native_error.localizedDescription ?: @"load compiled FFN model");
				gllm_coreml_ffn_release(m);
				return NULL;
			}
			MLModelDescription *description = m->model.modelDescription;
			id creator_metadata = description.metadata[MLModelCreatorDefinedKey];
			id binding = nil;
			if ([creator_metadata isKindOfClass:[NSDictionary class]]) binding = creator_metadata[@"gopherllm.ffn"];
			NSData *manifest_data = nil;
			if (binding != nil) {
				if (![binding isKindOfClass:[NSString class]]) {
					gllm_ffn_error(error, @"FFN source binding must be a UTF-8 string");
					gllm_coreml_ffn_release(m);
					return NULL;
				}
				manifest_data = [(NSString *)binding dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
				if (manifest_data == nil || manifest_data.length > 64 * 1024 ||
					(manifest_data.length != 0 && memchr(manifest_data.bytes, 0, manifest_data.length) != NULL)) {
					gllm_ffn_error(error, @"FFN source binding must be valid UTF-8 without NUL bytes and at most 64 KiB");
					gllm_coreml_ffn_release(m);
					return NULL;
				}
			}
			if (@available(macOS 15.0, iOS 18.0, *)) {
				if (description.stateDescriptionsByName.count != 0) {
					gllm_ffn_error(error, @"FFN model must be stateless");
					gllm_coreml_ffn_release(m);
					return NULL;
				}
			}
			int tokens = 0, dimension = 0, output_tokens = 0, output_dimension = 0;
			if (description.inputDescriptionsByName.count != 1 || description.outputDescriptionsByName.count != 1 ||
				!gllm_ffn_shape(description.inputDescriptionsByName[@"hidden"], &tokens, &dimension) ||
				!gllm_ffn_shape(description.outputDescriptionsByName[@"output"], &output_tokens, &output_dimension) ||
				tokens != output_tokens || dimension != output_dimension) {
				gllm_ffn_error(error, @"FFN must have exactly one hidden input and one output with matching fixed float32 [T,D] shapes, 1 <= T <= 128, D <= 16384, T*D <= 1048576");
				gllm_coreml_ffn_release(m);
				return NULL;
			}
			NSArray<NSNumber *> *shape = @[@(tokens), @(dimension)];
			m->input = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat32 error:&native_error];
			m->output = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat32 error:&native_error];
			if (m->input == nil || m->output == nil) {
				gllm_ffn_error(error, native_error.localizedDescription ?: @"allocate FFN model buffers");
				gllm_coreml_ffn_release(m);
				return NULL;
			}
			m->features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"hidden": [MLFeatureValue featureValueWithMultiArray:m->input]} error:&native_error];
			m->options = [[MLPredictionOptions alloc] init];
			m->options.outputBackings = @{@"output": m->output};
			if (m->features == nil || m->options == nil) {
				gllm_ffn_error(error, native_error.localizedDescription ?: @"create FFN model features");
				gllm_coreml_ffn_release(m);
				return NULL;
			}
			m->tokens = tokens;
			m->dimension = dimension;
			if (info != NULL) {
				info->tokens = tokens;
				info->dimension = dimension;
				if (manifest_data.length != 0) {
					info->manifest = malloc(manifest_data.length + 1);
					if (info->manifest == NULL) {
						gllm_ffn_error(error, @"allocate FFN source binding");
						gllm_coreml_ffn_release(m);
						return NULL;
					}
					memcpy(info->manifest, manifest_data.bytes, manifest_data.length);
					info->manifest[manifest_data.length] = 0;
				}
			}
			return m;
		} @catch (NSException *exception) {
			gllm_ffn_error(error, exception.reason ?: @"exception loading FFN model");
			gllm_coreml_ffn_release(m);
			return NULL;
		}
	}
}

static bool gllm_ffn_fits(NSInteger size, NSInteger row_stride, NSInteger column_stride, int rows, int columns) {
	if (size < 0 || row_stride < 0 || column_stride < 0 || rows <= 0 || columns <= 0 ||
		(rows > 1 && row_stride == 0) || (columns > 1 && column_stride == 0)) return false;
	size_t last_row = (size_t)(rows - 1), last_column = (size_t)(columns - 1);
	if (last_row != 0 && (size_t)row_stride > SIZE_MAX / last_row) return false;
	size_t last = last_row * (size_t)row_stride;
	if (last_column != 0 && (size_t)column_stride > (SIZE_MAX - last) / last_column) return false;
	last += last_column * (size_t)column_stride;
	return last <= (SIZE_MAX - sizeof(float)) / sizeof(float) && last * sizeof(float) + sizeof(float) <= (size_t)size;
}

static bool gllm_ffn_array_matches(MLMultiArray *array, int tokens, int dimension) {
	return array != nil && array.dataType == MLMultiArrayDataTypeFloat32 && array.shape.count == 2 &&
		array.shape[0].longLongValue == tokens && array.shape[1].longLongValue == dimension && array.count == (NSInteger)tokens * dimension;
}

static bool gllm_ffn_fill(MLMultiArray *array, const float *input, int tokens, int dimension) {
	if (!gllm_ffn_array_matches(array, tokens, dimension)) return false;
	__block bool ok = false;
	[array getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *strides) {
		if (bytes == NULL || strides.count != 2) return;
		NSInteger row_stride = strides[0].integerValue, column_stride = strides[1].integerValue;
		if (!gllm_ffn_fits(size, row_stride, column_stride, tokens, dimension)) return;
		if (column_stride == 1 && (tokens == 1 || row_stride == dimension)) {
			memcpy(bytes, input, (size_t)tokens * dimension * sizeof(float));
		} else for (int row = 0; row < tokens; row++) {
			char *destination = (char *)bytes + (size_t)row * row_stride * sizeof(float);
			const float *source = input + (size_t)row * dimension;
			if (column_stride == 1) memcpy(destination, source, (size_t)dimension * sizeof(float));
			else for (int column = 0; column < dimension; column++) {
				memcpy(destination + (size_t)column * column_stride * sizeof(float), source + column, sizeof(float));
			}
		}
		ok = true;
	}];
	return ok;
}

static bool gllm_ffn_read(MLMultiArray *array, float *output, int tokens, int dimension) {
	if (!gllm_ffn_array_matches(array, tokens, dimension)) return false;
	__block bool ok = false;
	[array getBytesWithHandler:^(const void *bytes, NSInteger size) {
		NSArray<NSNumber *> *strides = array.strides;
		if (bytes == NULL || strides.count != 2) return;
		NSInteger row_stride = strides[0].integerValue, column_stride = strides[1].integerValue;
		if (!gllm_ffn_fits(size, row_stride, column_stride, tokens, dimension)) return;
		if (column_stride == 1 && (tokens == 1 || row_stride == dimension)) {
			memcpy(output, bytes, (size_t)tokens * dimension * sizeof(float));
		} else for (int row = 0; row < tokens; row++) {
			const char *source = (const char *)bytes + (size_t)row * row_stride * sizeof(float);
			float *destination = output + (size_t)row * dimension;
			if (column_stride == 1) memcpy(destination, source, (size_t)dimension * sizeof(float));
			else for (int column = 0; column < dimension; column++) {
				memcpy(destination + column, source + (size_t)column * column_stride * sizeof(float), sizeof(float));
			}
		}
		ok = true;
	}];
	return ok;
}

bool gllm_coreml_ffn_predict(void *ptr, const float *input, float *output, char **error) {
	if (error != NULL) *error = NULL;
	if (ptr == NULL || input == NULL || output == NULL) {
		gllm_ffn_error(error, @"invalid FFN prediction buffers");
		return false;
	}
	@autoreleasepool {
		@try {
			GLLMCoreMLFFN *m = ptr;
			if (!gllm_ffn_fill(m->input, input, m->tokens, m->dimension)) {
				gllm_ffn_error(error, @"invalid FFN input layout");
				return false;
			}
			NSError *native_error = nil;
			id<MLFeatureProvider> prediction = [m->model predictionFromFeatures:m->features options:m->options error:&native_error];
			if (prediction == nil) {
				gllm_ffn_error(error, native_error.localizedDescription ?: @"FFN prediction failed");
				return false;
			}
			MLFeatureValue *feature = [prediction featureValueForName:@"output"];
			if (feature.type != MLFeatureTypeMultiArray || !gllm_ffn_read(feature.multiArrayValue, output, m->tokens, m->dimension)) {
				gllm_ffn_error(error, @"invalid FFN output shape, type or strides");
				return false;
			}
			return true;
		} @catch (NSException *exception) {
			gllm_ffn_error(error, exception.reason ?: @"exception during FFN prediction");
			return false;
		}
	}
}
