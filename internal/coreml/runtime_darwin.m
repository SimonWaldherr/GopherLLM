//go:build darwin && cgo && coreml

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <dispatch/dispatch.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "runtime.h"

typedef struct {
	MLModel *model;
	MLMultiArray *ids;
	MLMultiArray *mask;
	MLMultiArray *types;
	MLMultiArray *output;
	MLDictionaryFeatureProvider *features;
	MLPredictionOptions *options;
	NSString *output_name;
	int batch_size;
	int sequence_length;
	int dimension;
} GLLMCoreMLModel;

static void gllm_coreml_error(char **error, NSString *message) {
	if (error != NULL) *error = strdup(message.UTF8String ?: "native operation failed");
}

bool gllm_coreml_supported(void) {
	if (@available(macOS 13.0, iOS 16.0, *)) return true;
	return false;
}

static MLModelConfiguration *gllm_coreml_configuration(int compute_units) {
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

void gllm_coreml_release(void *ptr) {
	if (ptr == NULL) return;
	@autoreleasepool {
		GLLMCoreMLModel *m = ptr;
		[m->features release];
		[m->options release];
		[m->ids release];
		[m->mask release];
		[m->types release];
		[m->output release];
		[m->output_name release];
		[m->model release];
		free(m);
	}
}

// The runtime always uses one fixed shape. Reject flexible shape sets/ranges
// rather than silently choosing a specialization and reshaping between calls.
static bool gllm_coreml_fixed_shape(MLMultiArrayConstraint *constraint) {
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

static bool gllm_coreml_input_shape(MLFeatureDescription *feature, int *batch, int *length) {
	if (feature == nil || feature.type != MLFeatureTypeMultiArray) return false;
	MLMultiArrayConstraint *constraint = feature.multiArrayConstraint;
	NSArray<NSNumber *> *shape = constraint.shape;
	if (constraint.dataType != MLMultiArrayDataTypeInt32 || !gllm_coreml_fixed_shape(constraint) ||
		shape.count != 2) return false;
	long long rows = shape[0].longLongValue, columns = shape[1].longLongValue;
	if (rows < 1 || rows > GLLM_COREML_MAX_BATCH_SIZE || columns < 1 ||
		columns > GLLM_COREML_MAX_BUFFER_ELEMENTS / rows) return false;
	*batch = (int)rows;
	*length = (int)columns;
	return true;
}

static bool gllm_coreml_input_matches(MLFeatureDescription *feature, int batch, int length) {
	int rows = 0, columns = 0;
	return gllm_coreml_input_shape(feature, &rows, &columns) && rows == batch && columns == length;
}

static int gllm_coreml_output_dimension(MLFeatureDescription *feature, int batch) {
	if (feature == nil || feature.type != MLFeatureTypeMultiArray) return 0;
	MLMultiArrayConstraint *constraint = feature.multiArrayConstraint;
	NSArray<NSNumber *> *shape = constraint.shape;
	if (!gllm_coreml_fixed_shape(constraint) ||
		(constraint.dataType != MLMultiArrayDataTypeFloat16 && constraint.dataType != MLMultiArrayDataTypeFloat32) ||
		!((shape.count == 1 && batch == 1) || (shape.count == 2 && shape[0].longLongValue == batch))) return 0;
	long long dimension = shape.lastObject.longLongValue;
	return dimension > 0 && dimension <= GLLM_COREML_MAX_BUFFER_ELEMENTS / batch ? (int)dimension : 0;
}

void *gllm_coreml_open(const char *path, int compute_units, const char *ids,
	const char *mask, const char *types, const char *output, GLLMCoreMLInfo *info, char **error) {
	if (error != NULL) *error = NULL;
	if (info != NULL) memset(info, 0, sizeof(*info));
	if (!gllm_coreml_supported()) {
		gllm_coreml_error(error, @"requires macOS 13/iOS 16 or newer");
		return NULL;
	}
	GLLMCoreMLModel *m = NULL;
	@autoreleasepool {
		@try {
			m = calloc(1, sizeof(*m));
			if (m == NULL) {
				gllm_coreml_error(error, @"allocate model state");
				return NULL;
			}
			MLModelConfiguration *configuration = gllm_coreml_configuration(compute_units);
			if (configuration == nil) {
				gllm_coreml_error(error, @"unknown or unavailable compute units");
				gllm_coreml_release(m);
				return NULL;
			}
			// No FastPrediction hint: it can trade extra memory for latency.
			NSError *native_error = nil;
			NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path] isDirectory:YES];
			m->model = [[MLModel modelWithContentsOfURL:url configuration:configuration error:&native_error] retain];
			if (m->model == nil) {
				gllm_coreml_error(error, native_error.localizedDescription ?: @"load compiled model");
				gllm_coreml_release(m);
				return NULL;
			}
			MLModelDescription *description = m->model.modelDescription;
			if (@available(macOS 15.0, iOS 18.0, *)) {
				if (description.stateDescriptionsByName.count != 0) {
					gllm_coreml_error(error, @"embedding model must be stateless");
					gllm_coreml_release(m);
					return NULL;
				}
			}
			NSString *ids_name = [NSString stringWithUTF8String:ids];
			NSString *mask_name = [NSString stringWithUTF8String:mask];
			NSString *types_name = [NSString stringWithUTF8String:types];
			m->output_name = [[NSString stringWithUTF8String:output] copy];
			NSDictionary<NSString *, MLFeatureDescription *> *inputs = description.inputDescriptionsByName;
			NSUInteger expected_inputs = types[0] != 0 ? 3 : 2;
			int batch = 0, length = 0;
			if (inputs.count != expected_inputs || !gllm_coreml_input_shape(inputs[ids_name], &batch, &length) ||
				!gllm_coreml_input_matches(inputs[mask_name], batch, length) ||
				(types[0] != 0 && !gllm_coreml_input_matches(inputs[types_name], batch, length))) {
				gllm_coreml_error(error, @"inputs must exactly match the configured names and use the same fixed int32 [B, S] shape, with 1 <= B <= 8 and B*S <= 1048576");
				gllm_coreml_release(m);
				return NULL;
			}
			MLFeatureDescription *output_feature = description.outputDescriptionsByName[m->output_name];
			int dimension = gllm_coreml_output_dimension(output_feature, batch);
			if (dimension == 0) {
				gllm_coreml_error(error, @"output must be a fixed, pooled float16 or float32 [B, D] embedding (or [D] when B=1), with B*D <= 1048576");
				gllm_coreml_release(m);
				return NULL;
			}
			NSArray<NSNumber *> *input_shape = @[@(batch), @(length)];
			m->ids = [[MLMultiArray alloc] initWithShape:input_shape dataType:MLMultiArrayDataTypeInt32 error:&native_error];
			m->mask = [[MLMultiArray alloc] initWithShape:input_shape dataType:MLMultiArrayDataTypeInt32 error:&native_error];
			if (types[0] != 0) m->types = [[MLMultiArray alloc] initWithShape:input_shape dataType:MLMultiArrayDataTypeInt32 error:&native_error];
			m->output = [[MLMultiArray alloc] initWithShape:output_feature.multiArrayConstraint.shape
				dataType:output_feature.multiArrayConstraint.dataType error:&native_error];
			if (m->ids == nil || m->mask == nil || (types[0] != 0 && m->types == nil) || m->output == nil) {
				gllm_coreml_error(error, native_error.localizedDescription ?: @"allocate model buffers");
				gllm_coreml_release(m);
				return NULL;
			}
			NSMutableDictionary *values = [NSMutableDictionary dictionaryWithObjectsAndKeys:
				[MLFeatureValue featureValueWithMultiArray:m->ids], ids_name,
				[MLFeatureValue featureValueWithMultiArray:m->mask], mask_name, nil];
			if (m->types != nil) values[types_name] = [MLFeatureValue featureValueWithMultiArray:m->types];
			m->features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:values error:&native_error];
			m->options = [[MLPredictionOptions alloc] init];
			m->options.outputBackings = @{m->output_name: m->output};
			if (m->features == nil || m->options == nil) {
				gllm_coreml_error(error, native_error.localizedDescription ?: @"create model features");
				gllm_coreml_release(m);
				return NULL;
			}
			m->batch_size = batch;
			m->sequence_length = length;
			m->dimension = dimension;
			if (info != NULL) {
				info->batch_size = batch;
				info->sequence_length = length;
				info->dimension = dimension;
				if (@available(macOS 14.0, iOS 17.0, *)) {
					for (id<MLComputeDeviceProtocol> device in MLModel.availableComputeDevices) {
						if ([device isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
							NSInteger cores = ((MLNeuralEngineComputeDevice *)device).totalCoreCount;
							info->neural_engine_available = true;
							info->neural_engine_cores = cores > 0 && cores <= INT_MAX ? (int)cores : 0;
						}
					}
				}
			}
			return m;
		} @catch (NSException *exception) {
			gllm_coreml_error(error, exception.reason ?: @"exception loading model");
			gllm_coreml_release(m);
			return NULL;
		}
	}
}

bool gllm_coreml_inspect_supported(void) {
	if (@available(macOS 14.4, iOS 17.4, *)) return true;
	return false;
}

API_AVAILABLE(macos(14.4), ios(17.4))
static void gllm_coreml_count_device(MLComputePlanDeviceUsage *usage, GLLMCoreMLPlan *counts) {
	counts->operations++;
	id<MLComputeDeviceProtocol> device = usage.preferredComputeDevice;
	if ([device isKindOfClass:[MLCPUComputeDevice class]]) counts->cpu_operations++;
	else if ([device isKindOfClass:[MLGPUComputeDevice class]]) counts->gpu_operations++;
	else if ([device isKindOfClass:[MLNeuralEngineComputeDevice class]]) counts->neural_engine_operations++;
	else counts->unassigned_operations++;
}

API_AVAILABLE(macos(14.4), ios(17.4))
static void gllm_coreml_count_block(MLComputePlan *plan, MLModelStructureProgramBlock *block,
	GLLMCoreMLPlan *counts, NSHashTable *seen) {
	if (block == nil || [seen containsObject:block]) return;
	[seen addObject:block];
	for (MLModelStructureProgramOperation *operation in block.operations) {
		if ([seen containsObject:operation]) continue;
		[seen addObject:operation];
		gllm_coreml_count_device([plan computeDeviceUsageForMLProgramOperation:operation], counts);
		for (MLModelStructureProgramBlock *nested in operation.blocks) {
			gllm_coreml_count_block(plan, nested, counts, seen);
		}
	}
}

API_AVAILABLE(macos(14.4), ios(17.4))
static void gllm_coreml_count_structure(MLComputePlan *plan, MLModelStructure *structure,
	GLLMCoreMLPlan *counts, NSHashTable *seen) {
	if (structure == nil || [seen containsObject:structure]) return;
	[seen addObject:structure];
	for (MLModelStructureNeuralNetworkLayer *layer in structure.neuralNetwork.layers) {
		if ([seen containsObject:layer]) continue;
		[seen addObject:layer];
		gllm_coreml_count_device([plan computeDeviceUsageForNeuralNetworkLayer:layer], counts);
	}
	for (MLModelStructureProgramFunction *function in structure.program.functions.allValues) {
		gllm_coreml_count_block(plan, function.block, counts, seen);
	}
	for (MLModelStructure *model in structure.pipeline.subModels) {
		gllm_coreml_count_structure(plan, model, counts, seen);
	}
}

bool gllm_coreml_inspect(const char *path, int compute_units, GLLMCoreMLPlan *counts, char **error) {
	if (error != NULL) *error = NULL;
	if (counts != NULL) memset(counts, 0, sizeof(*counts));
	if (!gllm_coreml_inspect_supported() || path == NULL || counts == NULL) {
		gllm_coreml_error(error, @"compute plans require macOS 14.4/iOS 17.4 or newer");
		return false;
	}
	if (@available(macOS 14.4, iOS 17.4, *)) {
		@autoreleasepool {
			__block MLComputePlan *plan = nil;
			__block NSError *native_error = nil;
			dispatch_semaphore_t completion = dispatch_semaphore_create(0);
			@try {
				MLModelConfiguration *configuration = gllm_coreml_configuration(compute_units);
				if (configuration == nil) {
					gllm_coreml_error(error, @"unknown or unavailable compute units");
					return false;
				}
				NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path] isDirectory:YES];
				[MLComputePlan loadContentsOfURL:url configuration:configuration completionHandler:^(MLComputePlan *result, NSError *failure) {
					plan = [result retain];
					native_error = [failure retain];
					dispatch_semaphore_signal(completion);
				}];
				dispatch_semaphore_wait(completion, DISPATCH_TIME_FOREVER);
				if (plan == nil) {
					gllm_coreml_error(error, native_error.localizedDescription ?: @"load compute plan");
					return false;
				}
				NSHashTable *seen = [NSHashTable hashTableWithOptions:NSPointerFunctionsStrongMemory | NSPointerFunctionsObjectPointerPersonality];
				gllm_coreml_count_structure(plan, plan.modelStructure, counts, seen);
				return true;
			} @catch (NSException *exception) {
				gllm_coreml_error(error, exception.reason ?: @"exception inspecting compute plan");
				return false;
			} @finally {
				[plan release];
				[native_error release];
				dispatch_release(completion);
			}
		}
	}
	return false;
}

// Validate every address against the scoped buffer size, including Core ML
// layouts with padding or a non-unit stride in either dimension. Strides are
// in elements; convert only after checking both products and their sum.
static bool gllm_coreml_fits(NSInteger size, NSInteger row_stride, NSInteger column_stride,
	int rows, int columns, size_t width) {
	if (size < 0 || row_stride < 0 || column_stride < 0 || rows <= 0 || columns <= 0 || width == 0 ||
		(rows > 1 && row_stride == 0) || (columns > 1 && column_stride == 0)) return false;
	size_t last_row = (size_t)(rows - 1), last_column = (size_t)(columns - 1);
	if (last_row != 0 && (size_t)row_stride > SIZE_MAX / last_row) return false;
	size_t last = last_row * (size_t)row_stride;
	if (last_column != 0 && (size_t)column_stride > (SIZE_MAX - last) / last_column) return false;
	last += last_column * (size_t)column_stride;
	if (last > (SIZE_MAX - width) / width) return false;
	return last * width + width <= (size_t)size;
}

static bool gllm_coreml_fill(MLMultiArray *array, const int32_t *values, int batch, int length) {
	NSArray<NSNumber *> *shape = array.shape;
	if (array == nil || shape.count != 2 || shape[0].longLongValue != batch || shape[1].longLongValue != length ||
		array.count != (NSInteger)batch * length || array.dataType != MLMultiArrayDataTypeInt32) return false;
	__block bool ok = false;
	[array getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *strides) {
		if (bytes == NULL || strides.count != 2) return;
		NSInteger row_stride = strides[0].integerValue, column_stride = strides[1].integerValue;
		if (!gllm_coreml_fits(size, row_stride, column_stride, batch, length, sizeof(int32_t))) return;
		if (column_stride == 1 && (batch == 1 || row_stride == length)) {
			memcpy(bytes, values, (size_t)batch * (size_t)length * sizeof(int32_t));
		} else for (int row = 0; row < batch; row++) {
			char *destination = (char *)bytes + (size_t)row * (size_t)row_stride * sizeof(int32_t);
			const int32_t *source = values + (size_t)row * (size_t)length;
			if (column_stride == 1) memcpy(destination, source, (size_t)length * sizeof(int32_t));
			else for (int column = 0; column < length; column++) {
				memcpy(destination + (size_t)column * (size_t)column_stride * sizeof(int32_t), source + column, sizeof(int32_t));
			}
		}
		ok = true;
	}];
	return ok;
}

static float gllm_coreml_half_to_float(uint16_t value) {
	uint32_t sign = (uint32_t)(value & 0x8000) << 16;
	uint32_t exponent = (value >> 10) & 31, mantissa = value & 1023, bits;
	if (exponent == 0) {
		if (mantissa == 0) bits = sign;
		else {
			int e = -14;
			while ((mantissa & 1024) == 0) { mantissa <<= 1; e--; }
			bits = sign | ((uint32_t)(e + 127) << 23) | ((mantissa & 1023) << 13);
		}
	} else if (exponent == 31) bits = sign | 0x7f800000 | (mantissa << 13);
	else bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
	float result;
	memcpy(&result, &bits, sizeof(result));
	return result;
}

static bool gllm_coreml_read(MLMultiArray *array, int batch, int dimension, float *output) {
	NSArray<NSNumber *> *shape = array.shape;
	if (array == nil || !((shape.count == 1 && batch == 1) || (shape.count == 2 && shape[0].longLongValue == batch)) ||
		shape.lastObject.longLongValue != dimension || array.count != (NSInteger)batch * dimension ||
		(array.dataType != MLMultiArrayDataTypeFloat16 && array.dataType != MLMultiArrayDataTypeFloat32)) return false;
	__block bool ok = false;
	[array getBytesWithHandler:^(const void *bytes, NSInteger size) {
		NSArray<NSNumber *> *strides = array.strides;
		if (bytes == NULL || strides.count != shape.count) return;
		NSInteger row_stride = shape.count == 1 ? 0 : strides[0].integerValue;
		NSInteger column_stride = strides.lastObject.integerValue;
		size_t width = array.dataType == MLMultiArrayDataTypeFloat16 ? 2 : sizeof(float);
		if (!gllm_coreml_fits(size, row_stride, column_stride, batch, dimension, width)) return;
		if (width == sizeof(float) && column_stride == 1 && (batch == 1 || row_stride == dimension)) {
			memcpy(output, bytes, (size_t)batch * (size_t)dimension * sizeof(float));
		} else for (int row = 0; row < batch; row++) {
			const char *source = (const char *)bytes + (size_t)row * (size_t)row_stride * width;
			float *destination = output + (size_t)row * (size_t)dimension;
			if (width == sizeof(float) && column_stride == 1) memcpy(destination, source, (size_t)dimension * sizeof(float));
			else for (int column = 0; column < dimension; column++) {
				const char *value = source + (size_t)column * (size_t)column_stride * width;
				if (width == 2) {
					uint16_t half;
					memcpy(&half, value, sizeof(half));
					destination[column] = gllm_coreml_half_to_float(half);
				} else memcpy(destination + column, value, sizeof(float));
			}
		}
		ok = true;
	}];
	return ok;
}

bool gllm_coreml_predict(void *ptr, const int32_t *ids, const int32_t *mask,
	const int32_t *types, float *output, char **error) {
	if (error != NULL) *error = NULL;
	if (ptr == NULL || ids == NULL || mask == NULL || output == NULL) {
		gllm_coreml_error(error, @"invalid prediction buffers");
		return false;
	}
	@autoreleasepool {
		@try {
			GLLMCoreMLModel *m = ptr;
			if ((m->types != nil) != (types != NULL) ||
				!gllm_coreml_fill(m->ids, ids, m->batch_size, m->sequence_length) ||
				!gllm_coreml_fill(m->mask, mask, m->batch_size, m->sequence_length) ||
				(m->types != nil && !gllm_coreml_fill(m->types, types, m->batch_size, m->sequence_length))) {
				gllm_coreml_error(error, @"invalid input buffer layout");
				return false;
			}
			NSError *native_error = nil;
			id<MLFeatureProvider> prediction = [m->model predictionFromFeatures:m->features options:m->options error:&native_error];
			if (prediction == nil) {
				gllm_coreml_error(error, native_error.localizedDescription ?: @"prediction failed");
				return false;
			}
			MLFeatureValue *feature = [prediction featureValueForName:m->output_name];
			if (feature.type != MLFeatureTypeMultiArray || !gllm_coreml_read(feature.multiArrayValue, m->batch_size, m->dimension, output)) {
				gllm_coreml_error(error, @"invalid pooled output shape, data type or strides");
				return false;
			}
			return true;
		} @catch (NSException *exception) {
			gllm_coreml_error(error, exception.reason ?: @"exception during prediction");
			return false;
		}
	}
}
