#pragma once

#include <mutex>
#include <vector>
#include <memory>
#include <unordered_map>

#include "xsched/ascend/hal/driver.h"
#include "xsched/ascend/hal/acl_assert.h"
#include "xsched/preempt/hal/hw_command.h"

namespace xsched::ascend
{

#define ACL_COMMAND(name, base, ret_t, func, ...) \
    class name : public base \
    { \
    public: \
        name(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) \
            : base(), FOR_EACH_PAIR_COMMA(DECLARE_COPY_PRIVATE_ARG, __VA_ARGS__) {} \
        virtual ~name() = default; \
    private: \
        FOR_EACH_PAIR_SEMICOLON(DECLARE_PRIVATE_PARAM, __VA_ARGS__) \
        virtual ret_t Launch(aclrtStream stream) override \
        { return func(FOR_EACH_PAIR_COMMA(DECLARE_PRIVATE_ARG, __VA_ARGS__) __VA_OPT__(,) stream); } \
    };

class AclCommand : public preempt::HwCommand
{
public:
    AclCommand(): HwCommand(preempt::kCommandPropertyNone) {}
    virtual ~AclCommand();
    virtual void Synchronize() override;
    virtual bool Synchronizable() override;
    virtual bool EnableSynchronization() override;
    aclError LaunchWrapper(aclrtStream stream);

private:
    virtual aclError Launch(aclrtStream stream) = 0;
    aclrtEvent following_event_ = nullptr;
};

class AclEventRecordCommand : public AclCommand
{
public:
    AclEventRecordCommand(aclrtEvent event);
    virtual ~AclEventRecordCommand() = default;
    virtual void Synchronize() override { ACL_ASSERT(Driver::rtSynchronizeEvent(event_)); }
    virtual bool Synchronizable() override { return true; }
    virtual bool EnableSynchronization() override { return true; }

private:
    aclrtEvent event_;
    virtual aclError Launch(aclrtStream stream) override
    { return Driver::rtRecordEvent(event_, stream); }
};

class TensorDesc
{
public:
    TensorDesc() = default;
    virtual ~TensorDesc() { if (desc_) Driver::DestroyTensorDesc(desc_); }
    const aclTensorDesc *desc() const { return desc_; }

    static std::shared_ptr<TensorDesc> Create(const aclTensorDesc *desc);
    static bool Destroy(const aclTensorDesc *desc);

private:
    const aclTensorDesc *desc_ = nullptr;
    static std::mutex tensor_desc_mutex_;
    static std::unordered_map<const aclTensorDesc *, std::shared_ptr<TensorDesc>> tensor_descs_;
};

class DataBuffer
{
public:
    DataBuffer() = default;
    virtual ~DataBuffer() { if (buffer_) Driver::DestroyDataBuffer(buffer_); }
    const aclDataBuffer *buffer() const { return buffer_; }

    static std::shared_ptr<DataBuffer> Create(const aclDataBuffer *buffer);
    static bool Destroy(const aclDataBuffer *buffer);

private:
    const aclDataBuffer *buffer_ = nullptr;
    static std::mutex data_buffer_mutex_;
    static std::unordered_map<const aclDataBuffer *, std::shared_ptr<DataBuffer>> data_buffers_;
};

class OpAttr
{
public:
    OpAttr() = default;
    virtual ~OpAttr() { if (attr_) Driver::opDestroyAttr(attr_); }
    const aclopAttr *attr() const { return attr_; }

    static std::shared_ptr<OpAttr> Create(const aclopAttr *attr);
    static bool Destroy(const aclopAttr *attr);

private:
    const aclopAttr *attr_ = nullptr;
    static std::mutex op_attr_mutex_;
    static std::unordered_map<const aclopAttr *, std::shared_ptr<OpAttr>> op_attrs_;
};

class AclOpCompileAndExecuteCommand : public AclCommand
{
public:
    AclOpCompileAndExecuteCommand(const char * opType,
                                  std::shared_ptr<std::vector<std::shared_ptr<TensorDesc>>> inputDesc,
                                  std::shared_ptr<std::vector<std::shared_ptr<DataBuffer>>> inputs,
                                  std::shared_ptr<std::vector<std::shared_ptr<TensorDesc>>> outputDesc,
                                  std::shared_ptr<std::vector<std::shared_ptr<DataBuffer>>> outputs,
                                  std::shared_ptr<OpAttr> attr,
                                  aclopEngineType engineType, aclopCompileType compileFlag,
                                  const char * opPath)
        : opType_(opType)
        , inputDesc_(inputDesc), inputs_(inputs), outputDesc_(outputDesc), outputs_(outputs)
        , attr_(attr), engineType_(engineType), compileFlag_(compileFlag), opPath_(opPath)
    {
        XASSERT(inputDesc_->size() == inputs_->size(), "inputDesc and inputs should have the same size");
        XASSERT(outputDesc_->size() == outputs_->size(), "outputDesc and outputs should have the same size");
    }
    virtual ~AclOpCompileAndExecuteCommand() = default;
private:
    const char * opType_;
    std::shared_ptr<std::vector<std::shared_ptr<TensorDesc>>> inputDesc_;
    std::shared_ptr<std::vector<std::shared_ptr<DataBuffer>>> inputs_;
    std::shared_ptr<std::vector<std::shared_ptr<TensorDesc>>> outputDesc_;
    std::shared_ptr<std::vector<std::shared_ptr<DataBuffer>>> outputs_;
    std::shared_ptr<OpAttr> attr_;
    aclopEngineType engineType_;
    aclopCompileType compileFlag_;
    const char * opPath_;
    virtual aclError Launch(aclrtStream stream) override;
};

ACL_COMMAND(AclModelExecuteCommand, AclCommand, aclError, Driver::mdlExecuteAsync, uint32_t, modelId, const aclmdlDataset *, input, aclmdlDataset *, output);
ACL_COMMAND(AclMemcpyCommand, AclCommand, aclError, Driver::rtMemcpyAsync, void *, dst, size_t, destMax, const void *, src, size_t, count, aclrtMemcpyKind, kind);
ACL_COMMAND(AclMemsetCommand, AclCommand, aclError, Driver::rtMemsetAsync, void *, devPtr, size_t, maxCount, int32_t, value, size_t, count);

#define ACL_NN_COMMAND(name) \
    ACL_COMMAND(AclNN##name##Command, AclCommand, aclnnStatus, NnOp::nn##name, void *, workspace, uint64_t, workspaceSize, aclOpExecutor *, executor);

ACL_NN_COMMAND(DropoutBackward);
ACL_NN_COMMAND(Dropout);
ACL_NN_COMMAND(NormalTensorTensor);
ACL_NN_COMMAND(NormalTensorFloat);
ACL_NN_COMMAND(NormalFloatTensor);
ACL_NN_COMMAND(NormalFloatFloat);
ACL_NN_COMMAND(Bernoulli);
ACL_NN_COMMAND(BernoulliTensor);
ACL_NN_COMMAND(InplaceBernoulli);
ACL_NN_COMMAND(InplaceBernoulliTensor);
ACL_NN_COMMAND(Multinomial);
ACL_NN_COMMAND(DropoutDoMask);
ACL_NN_COMMAND(InplaceRandom);
ACL_NN_COMMAND(InplaceUniform);
ACL_NN_COMMAND(DropoutGenMask);
ACL_NN_COMMAND(DropoutGenMaskV2);
ACL_NN_COMMAND(Randperm);
ACL_NN_COMMAND(InplaceNormal);
ACL_NN_COMMAND(IncreFlashAttentionV2);
ACL_NN_COMMAND(BatchNormBackward);
ACL_NN_COMMAND(ForeachPowScalarV2);
ACL_NN_COMMAND(Aminmax);
ACL_NN_COMMAND(DynamicQuant);
ACL_NN_COMMAND(MaxPool3dWithArgmax);
ACL_NN_COMMAND(Sign);
ACL_NN_COMMAND(MoeInitRoutingQuantV2);
ACL_NN_COMMAND(LogSigmoidBackward);
ACL_NN_COMMAND(MatmulReduceScatter);
ACL_NN_COMMAND(MoeFinalizeRoutingV2);
// ACL_NN_COMMAND(CalculateMatmulWeightSizeV2); // not a command
// ACL_NN_COMMAND(CalculateMatmulWeightSize);   // not a command
ACL_NN_COMMAND(TransMatmulWeight);
ACL_NN_COMMAND(UpsampleNearest2d);
ACL_NN_COMMAND(Complex);
ACL_NN_COMMAND(Reciprocal);
ACL_NN_COMMAND(InplaceReciprocal);
ACL_NN_COMMAND(FusedInferAttentionScore);
ACL_NN_COMMAND(VarMean);
ACL_NN_COMMAND(Amin);
ACL_NN_COMMAND(GroupedMatmul);
ACL_NN_COMMAND(ForeachAddcdivList);
ACL_NN_COMMAND(GatherV2);
ACL_NN_COMMAND(SmoothL1Loss);
ACL_NN_COMMAND(MaxPool2dWithMask);
ACL_NN_COMMAND(MaxPool2dWithIndices);
ACL_NN_COMMAND(L1LossBackward);
ACL_NN_COMMAND(IsInf);
ACL_NN_COMMAND(GluBackward);
ACL_NN_COMMAND(UpsampleLinear1dBackward);
ACL_NN_COMMAND(Erfinv);
ACL_NN_COMMAND(InplaceErfinv);
ACL_NN_COMMAND(UpsampleNearestExact3dBackward);
ACL_NN_COMMAND(Hardsigmoid);
ACL_NN_COMMAND(InplaceHardsigmoid);
ACL_NN_COMMAND(QuantMatmulWeightNz);
ACL_NN_COMMAND(ForeachSinh);
ACL_NN_COMMAND(MaxV2);
ACL_NN_COMMAND(ForeachAddcdivScalarList);
ACL_NN_COMMAND(Trace);
ACL_NN_COMMAND(Min);
ACL_NN_COMMAND(HardtanhBackward);
ACL_NN_COMMAND(AvgPool3dBackward);
ACL_NN_COMMAND(BatchNormReduceBackward);
ACL_NN_COMMAND(BatchMatmulQuant);
ACL_NN_COMMAND(GeGluBackward);
ACL_NN_COMMAND(GeGluV3Backward);
ACL_NN_COMMAND(BatchNormElemtBackward);
ACL_NN_COMMAND(ReplicationPad1d);
ACL_NN_COMMAND(PromptFlashAttention);
ACL_NN_COMMAND(XLogYScalarOther);
ACL_NN_COMMAND(InplaceXLogYScalarOther);
ACL_NN_COMMAND(LayerNormBackward);
ACL_NN_COMMAND(InplaceCopy);
ACL_NN_COMMAND(FmodScalar);
ACL_NN_COMMAND(InplaceFmodScalar);
ACL_NN_COMMAND(Lerp);
ACL_NN_COMMAND(InplaceLerp);
ACL_NN_COMMAND(L1Loss);
ACL_NN_COMMAND(LogAddExp2);
ACL_NN_COMMAND(MoeInitRoutingV2);
ACL_NN_COMMAND(Exp2);
ACL_NN_COMMAND(InplaceExp2);
ACL_NN_COMMAND(Unique);
ACL_NN_COMMAND(GroupedMatmulV2);
ACL_NN_COMMAND(ConvertWeightToINT4Pack);
ACL_NN_COMMAND(Cummax);
ACL_NN_COMMAND(ForeachMinimumScalarV2);
ACL_NN_COMMAND(ArgMax);
ACL_NN_COMMAND(PdistForward);
ACL_NN_COMMAND(Rsubs);
ACL_NN_COMMAND(Rsub);
ACL_NN_COMMAND(ReflectionPad1d);
ACL_NN_COMMAND(SwinTransformerLnQkvQuant);
ACL_NN_COMMAND(InplaceMaskedScatter);
ACL_NN_COMMAND(GeScalar);
ACL_NN_COMMAND(InplaceGeScalar);
ACL_NN_COMMAND(InplaceScatterUpdate);
ACL_NN_COMMAND(Median);
ACL_NN_COMMAND(MedianDim);
ACL_NN_COMMAND(NanMedian);
ACL_NN_COMMAND(NanMedianDim);
ACL_NN_COMMAND(BitwiseXorTensor);
ACL_NN_COMMAND(InplaceBitwiseXorTensor);
ACL_NN_COMMAND(SiluBackward);
ACL_NN_COMMAND(ApplyFusedEmaAdam);
ACL_NN_COMMAND(BitwiseOrTensor);
ACL_NN_COMMAND(InplaceBitwiseOrTensor);
ACL_NN_COMMAND(MaxPool2dWithMaskBackward);
ACL_NN_COMMAND(MaxPool2dWithIndicesBackward);
ACL_NN_COMMAND(ForeachZeroInplace);
ACL_NN_COMMAND(ForeachMulScalar);
ACL_NN_COMMAND(ApplyAdamWV2);
ACL_NN_COMMAND(SliceV2);
ACL_NN_COMMAND(RoiAlign);
ACL_NN_COMMAND(QuantMatmulAllReduceV3);
ACL_NN_COMMAND(GridSampler2DBackward);
ACL_NN_COMMAND(AlltoAllAllGatherBatchMatMul);
ACL_NN_COMMAND(EqTensor);
ACL_NN_COMMAND(InplaceEqTensor);
ACL_NN_COMMAND(Trunc);
ACL_NN_COMMAND(InplaceTrunc);
ACL_NN_COMMAND(LeScalar);
ACL_NN_COMMAND(InplaceLeScalar);
ACL_NN_COMMAND(DeepNormGrad);
ACL_NN_COMMAND(Stack);
ACL_NN_COMMAND(Bincount);
ACL_NN_COMMAND(ForeachAddcmulScalarList);
ACL_NN_COMMAND(UpsampleBicubic2dBackward);
ACL_NN_COMMAND(BitwiseAndScalar);
ACL_NN_COMMAND(InplaceBitwiseAndScalar);
ACL_NN_COMMAND(BatchNormElemt);
ACL_NN_COMMAND(GroupedMatMulAllReduce);
ACL_NN_COMMAND(LinalgQr);
ACL_NN_COMMAND(ChamferDistanceBackward);
ACL_NN_COMMAND(Cumsum);
ACL_NN_COMMAND(CumsumV2);
ACL_NN_COMMAND(ForeachExp);
ACL_NN_COMMAND(ForeachCopy);
ACL_NN_COMMAND(ForeachNeg);
ACL_NN_COMMAND(Embedding);
ACL_NN_COMMAND(ForeachCos);
ACL_NN_COMMAND(UpsampleNearest2dBackward);
ACL_NN_COMMAND(AddLayerNormGrad);
ACL_NN_COMMAND(NeTensor);
ACL_NN_COMMAND(InplaceNeTensor);
ACL_NN_COMMAND(LogicalOr);
ACL_NN_COMMAND(InplaceLogicalOr);
ACL_NN_COMMAND(SplitTensor);
ACL_NN_COMMAND(Log10);
ACL_NN_COMMAND(InplaceLog10);
ACL_NN_COMMAND(GlobalMaxPool);
ACL_NN_COMMAND(MaxPool);
ACL_NN_COMMAND(UpsampleLinear1d);
ACL_NN_COMMAND(Sinh);
ACL_NN_COMMAND(InplaceSinh);
ACL_NN_COMMAND(ForeachSubList);
ACL_NN_COMMAND(ForeachSin);
ACL_NN_COMMAND(UpsampleNearest3d);
ACL_NN_COMMAND(HardsigmoidBackward);
ACL_NN_COMMAND(MoeGatingTopKSoftmaxV2);
ACL_NN_COMMAND(MseLossOut);
ACL_NN_COMMAND(MoeGatingTopKSoftmax);
ACL_NN_COMMAND(Addcdiv);
ACL_NN_COMMAND(InplaceAddcdiv);
ACL_NN_COMMAND(ForeachSign);
ACL_NN_COMMAND(AllGatherMatmul);
ACL_NN_COMMAND(ForeachMinimumScalar);
ACL_NN_COMMAND(Sort);
ACL_NN_COMMAND(ForeachMulList);
ACL_NN_COMMAND(ForeachMaximumScalar);
ACL_NN_COMMAND(AdaptiveAvgPool2dBackward);
ACL_NN_COMMAND(ForeachAddcmulScalar);
ACL_NN_COMMAND(InplaceMaskedFillTensor);
ACL_NN_COMMAND(ReduceLogSum);
ACL_NN_COMMAND(InplaceZero);
ACL_NN_COMMAND(Resize);
ACL_NN_COMMAND(Sinkhorn);
ACL_NN_COMMAND(PromptFlashAttentionV2);
ACL_NN_COMMAND(MaxN);
ACL_NN_COMMAND(Mean);
ACL_NN_COMMAND(MeanV2);
ACL_NN_COMMAND(MoeFinalizeRoutingV2Grad);
ACL_NN_COMMAND(MoeTokenPermute);
ACL_NN_COMMAND(InplaceMatmulAllReduceAddRmsNorm);
ACL_NN_COMMAND(ForeachMaximumList);
ACL_NN_COMMAND(Var);
ACL_NN_COMMAND(VarCorrection);
ACL_NN_COMMAND(LtTensor);
ACL_NN_COMMAND(InplaceLtTensor);
ACL_NN_COMMAND(Tril);
ACL_NN_COMMAND(InplaceTril);
ACL_NN_COMMAND(FFNV2);
ACL_NN_COMMAND(Atanh);
ACL_NN_COMMAND(InplaceAtanh);
ACL_NN_COMMAND(Addcmul);
ACL_NN_COMMAND(InplaceAddcmul);
ACL_NN_COMMAND(IsInScalarTensor);
ACL_NN_COMMAND(AdaptiveAvgPool2d);
ACL_NN_COMMAND(Acosh);
ACL_NN_COMMAND(InplaceAcosh);
ACL_NN_COMMAND(BatchNorm);
ACL_NN_COMMAND(Cast);
ACL_NN_COMMAND(ReplicationPad1dBackward);
ACL_NN_COMMAND(ForeachAtan);
ACL_NN_COMMAND(ForeachSubListV2);
ACL_NN_COMMAND(InplaceQuantMatmulAllReduceAddRmsNorm);
ACL_NN_COMMAND(QuantMatmulAllReduceAddRmsNorm);
ACL_NN_COMMAND(FloorDivide);
ACL_NN_COMMAND(FloorDivides);
ACL_NN_COMMAND(InplaceFloorDivide);
ACL_NN_COMMAND(InplaceFloorDivides);
ACL_NN_COMMAND(LogSigmoid);
ACL_NN_COMMAND(LogSigmoidForward);
ACL_NN_COMMAND(All);
ACL_NN_COMMAND(SoftshrinkBackward);
ACL_NN_COMMAND(UpsampleNearest1dBackward);
ACL_NN_COMMAND(SeluBackward);
ACL_NN_COMMAND(AdaptiveMaxPool2d);
ACL_NN_COMMAND(Addr);
ACL_NN_COMMAND(InplaceAddr);
ACL_NN_COMMAND(Celu);
ACL_NN_COMMAND(InplaceCelu);
ACL_NN_COMMAND(WeightQuantBatchMatmulV2);
ACL_NN_COMMAND(ForeachMulScalarList);
ACL_NN_COMMAND(Diag);
ACL_NN_COMMAND(AminmaxDim);
ACL_NN_COMMAND(SigmoidBackward);
ACL_NN_COMMAND(MatmulCompressDequant);
ACL_NN_COMMAND(ReduceNansum);
ACL_NN_COMMAND(Add);
ACL_NN_COMMAND(Adds);
ACL_NN_COMMAND(InplaceAdd);
ACL_NN_COMMAND(InplaceAdds);
ACL_NN_COMMAND(Norm);
ACL_NN_COMMAND(OneHot);
ACL_NN_COMMAND(UpsampleNearestExact2d);
ACL_NN_COMMAND(ForeachErfc);
ACL_NN_COMMAND(RingAttentionUpdate);
ACL_NN_COMMAND(PrecisionCompare);
ACL_NN_COMMAND(ForeachLerpList);
ACL_NN_COMMAND(MoeInitRoutingQuant);
ACL_NN_COMMAND(ChannelShuffle);
ACL_NN_COMMAND(RReluWithNoise);
ACL_NN_COMMAND(InplaceRReluWithNoise);
ACL_NN_COMMAND(Sub);
ACL_NN_COMMAND(Subs);
ACL_NN_COMMAND(InplaceSub);
ACL_NN_COMMAND(InplaceSubs);
ACL_NN_COMMAND(BinaryCrossEntropy);
ACL_NN_COMMAND(XLogYTensor);
ACL_NN_COMMAND(InplaceXLogYTensor);
ACL_NN_COMMAND(Convolution);
ACL_NN_COMMAND(ConvTbc);
ACL_NN_COMMAND(ConvDepthwise2d);
ACL_NN_COMMAND(SplitWithSize);
ACL_NN_COMMAND(ForeachSigmoid);
ACL_NN_COMMAND(ForeachErf);
ACL_NN_COMMAND(LayerNorm);
ACL_NN_COMMAND(LayerNormWithImplMode);
ACL_NN_COMMAND(MaxUnpool2dBackward);
ACL_NN_COMMAND(BitwiseNot);
ACL_NN_COMMAND(Sigmoid);
ACL_NN_COMMAND(InplaceSigmoid);
ACL_NN_COMMAND(Lgamma);
ACL_NN_COMMAND(ForeachLog10);
ACL_NN_COMMAND(Addbmm);
ACL_NN_COMMAND(InplaceAddbmm);
ACL_NN_COMMAND(Argsort);
ACL_NN_COMMAND(IsPosInf);
ACL_NN_COMMAND(MaxDim);
ACL_NN_COMMAND(GroupedBiasAddGrad);
ACL_NN_COMMAND(GroupedBiasAddGradV2);
ACL_NN_COMMAND(IndexCopy);
ACL_NN_COMMAND(InplaceIndexCopy);
ACL_NN_COMMAND(ForeachAddcmulScalarV2);
ACL_NN_COMMAND(MoeTokenUnpermuteGrad);
ACL_NN_COMMAND(RepeatInterleave);
ACL_NN_COMMAND(RepeatInterleaveWithDim);
ACL_NN_COMMAND(RepeatInterleaveInt);
ACL_NN_COMMAND(RepeatInterleaveIntWithDim);
ACL_NN_COMMAND(RepeatInterleaveTensor);
ACL_NN_COMMAND(LogicalXor);
ACL_NN_COMMAND(Abs);
ACL_NN_COMMAND(UpsampleBilinear2d);
ACL_NN_COMMAND(UpsampleNearestExact3d);
ACL_NN_COMMAND(ForeachDivScalarV2);
ACL_NN_COMMAND(TriangularSolve);
ACL_NN_COMMAND(SoftplusBackward);
ACL_NN_COMMAND(CircularPad3dBackward);
ACL_NN_COMMAND(LeakyRelu);
ACL_NN_COMMAND(InplaceLeakyRelu);
ACL_NN_COMMAND(ForeachPowScalarAndTensor);
ACL_NN_COMMAND(ScatterNd);
ACL_NN_COMMAND(ReduceSum);
ACL_NN_COMMAND(MoeInitRoutingV2Grad);
ACL_NN_COMMAND(IndexSelect);
ACL_NN_COMMAND(SearchSorted);
ACL_NN_COMMAND(SearchSorteds);
ACL_NN_COMMAND(ForeachNorm);
ACL_NN_COMMAND(LeakyReluBackward);
ACL_NN_COMMAND(Mish);
ACL_NN_COMMAND(InplaceMish);
ACL_NN_COMMAND(Minimum);
ACL_NN_COMMAND(Arange);
ACL_NN_COMMAND(MoeComputeExpertTokens);
ACL_NN_COMMAND(Floor);
ACL_NN_COMMAND(InplaceFloor);
ACL_NN_COMMAND(Tan);
ACL_NN_COMMAND(InplaceTan);
ACL_NN_COMMAND(GroupQuant);
ACL_NN_COMMAND(HardswishBackward);
ACL_NN_COMMAND(IsFinite);
ACL_NN_COMMAND(Frac);
ACL_NN_COMMAND(InplaceFrac);
ACL_NN_COMMAND(ForeachAddScalarV2);
ACL_NN_COMMAND(Erfc);
ACL_NN_COMMAND(InplaceErfc);
ACL_NN_COMMAND(InplaceQuantScatter);
ACL_NN_COMMAND(InplaceFillTensor);
ACL_NN_COMMAND(Atan);
ACL_NN_COMMAND(InplaceAtan);
ACL_NN_COMMAND(ReflectionPad1dBackward);
ACL_NN_COMMAND(Polar);
ACL_NN_COMMAND(KlDivBackward);
ACL_NN_COMMAND(ForeachLog);
ACL_NN_COMMAND(ForeachDivScalarList);
ACL_NN_COMMAND(NLLLoss2dBackward);
ACL_NN_COMMAND(AdaptiveAvgPool3d);
ACL_NN_COMMAND(Ger);
ACL_NN_COMMAND(EmbeddingDenseBackward);
ACL_NN_COMMAND(SmoothL1LossBackward);
ACL_NN_COMMAND(ReflectionPad2dBackward);
ACL_NN_COMMAND(ReplicationPad2dBackward);
ACL_NN_COMMAND(IndexFillTensor);
ACL_NN_COMMAND(InplaceIndexFillTensor);
ACL_NN_COMMAND(Histc);
ACL_NN_COMMAND(QuantMatmul);
ACL_NN_COMMAND(QuantMatmulV2);
ACL_NN_COMMAND(Atan2);
ACL_NN_COMMAND(InplaceAtan2);
ACL_NN_COMMAND(Scatter);
ACL_NN_COMMAND(ScatterValue);
ACL_NN_COMMAND(InplaceScatter);
ACL_NN_COMMAND(InplaceScatterValue);
ACL_NN_COMMAND(MishBackward);
ACL_NN_COMMAND(GlobalAveragePool);
ACL_NN_COMMAND(InplacePut);
ACL_NN_COMMAND(ThresholdBackward);
ACL_NN_COMMAND(ForeachMinimumScalarList);
ACL_NN_COMMAND(Hardswish);
ACL_NN_COMMAND(InplaceHardswish);
ACL_NN_COMMAND(GtTensor);
ACL_NN_COMMAND(InplaceGtTensor);
ACL_NN_COMMAND(ForeachMaximumScalarV2);
ACL_NN_COMMAND(UpsampleBilinear2dBackward);
ACL_NN_COMMAND(EmbeddingBag);
ACL_NN_COMMAND(BitwiseOrScalar);
ACL_NN_COMMAND(InplaceBitwiseOrScalar);
ACL_NN_COMMAND(BitwiseXorScalar);
ACL_NN_COMMAND(InplaceBitwiseXorScalar);
ACL_NN_COMMAND(Muls);
ACL_NN_COMMAND(Mul);
ACL_NN_COMMAND(InplaceMuls);
ACL_NN_COMMAND(InplaceMul);
ACL_NN_COMMAND(NanToNum);
ACL_NN_COMMAND(InplaceNanToNum);
ACL_NN_COMMAND(Gcd);
ACL_NN_COMMAND(Real);
ACL_NN_COMMAND(GeTensor);
ACL_NN_COMMAND(InplaceGeTensor);
ACL_NN_COMMAND(FFNV3);
ACL_NN_COMMAND(Hardshrink);
ACL_NN_COMMAND(Renorm);
ACL_NN_COMMAND(InplaceRenorm);
ACL_NN_COMMAND(InstanceNorm);
ACL_NN_COMMAND(BinaryCrossEntropyWithLogitsBackward);
ACL_NN_COMMAND(Eye);
ACL_NN_COMMAND(CircularPad2d);
ACL_NN_COMMAND(AminmaxAll);
ACL_NN_COMMAND(LinalgCross);
ACL_NN_COMMAND(SoftMarginLossBackward);
ACL_NN_COMMAND(QuantMatmulV3);
ACL_NN_COMMAND(Sum);
ACL_NN_COMMAND(ForeachAddcmulList);
ACL_NN_COMMAND(AdaptiveAvgPool3dBackward);
ACL_NN_COMMAND(ForeachAbs);
ACL_NN_COMMAND(Glu);
ACL_NN_COMMAND(Ceil);
ACL_NN_COMMAND(InplaceCeil);
ACL_NN_COMMAND(Addmv);
ACL_NN_COMMAND(PromptFlashAttentionV3);
ACL_NN_COMMAND(WeightQuantBatchMatmulV3);
ACL_NN_COMMAND(Lerps);
ACL_NN_COMMAND(InplaceLerps);
ACL_NN_COMMAND(FmodTensor);
ACL_NN_COMMAND(InplaceFmodTensor);
ACL_NN_COMMAND(PowTensorTensor);
ACL_NN_COMMAND(InplacePowTensorTensor);
ACL_NN_COMMAND(DynamicQuantV2);
ACL_NN_COMMAND(GroupedMatmulV4);
ACL_NN_COMMAND(LogicalAnd);
ACL_NN_COMMAND(InplaceLogicalAnd);
ACL_NN_COMMAND(NLLLoss);
ACL_NN_COMMAND(UpsampleNearest3dBackward);
ACL_NN_COMMAND(Nonzero);
ACL_NN_COMMAND(PreluBackward);
ACL_NN_COMMAND(MseLoss);
ACL_NN_COMMAND(Index);
ACL_NN_COMMAND(ReplicationPad3dBackward);
// ACL_NN_COMMAND(CalculateConvolutionWeightSize); // not a command
ACL_NN_COMMAND(TransConvolutionWeight);
ACL_NN_COMMAND(CtcLoss);
ACL_NN_COMMAND(ForeachTan);
ACL_NN_COMMAND(SoftmaxBackward);
ACL_NN_COMMAND(MoeFinalizeRouting);
ACL_NN_COMMAND(Log);
ACL_NN_COMMAND(InplaceLog);
ACL_NN_COMMAND(Im2col);
ACL_NN_COMMAND(Quantize);
ACL_NN_COMMAND(Prod);
ACL_NN_COMMAND(ProdDim);
ACL_NN_COMMAND(IncreFlashAttentionV4);
ACL_NN_COMMAND(HardshrinkBackward);
ACL_NN_COMMAND(Tanh);
ACL_NN_COMMAND(InplaceTanh);
ACL_NN_COMMAND(Scale);
ACL_NN_COMMAND(MaxUnpool2d);
ACL_NN_COMMAND(ConstantPadNd);
ACL_NN_COMMAND(Cosh);
ACL_NN_COMMAND(InplaceCosh);
ACL_NN_COMMAND(BatchNormGatherStatsWithCounts);
ACL_NN_COMMAND(Mm);
ACL_NN_COMMAND(Range);
ACL_NN_COMMAND(StdMeanCorrection);
ACL_NN_COMMAND(Elu);
ACL_NN_COMMAND(InplaceElu);
ACL_NN_COMMAND(GridSampler3DBackward);
ACL_NN_COMMAND(UpsampleBicubic2d);
ACL_NN_COMMAND(WeightQuantMatmulAllReduceAddRmsNorm);
ACL_NN_COMMAND(GridSampler3D);
ACL_NN_COMMAND(IsInTensorScalar);
ACL_NN_COMMAND(BinaryCrossEntropyBackward);
ACL_NN_COMMAND(TanhBackward);
ACL_NN_COMMAND(ForeachAddScalar);
ACL_NN_COMMAND(SwishBackward);
ACL_NN_COMMAND(StridedSliceAssignV2);
ACL_NN_COMMAND(TransQuantParamV2);
ACL_NN_COMMAND(BlendImagesCustom);
ACL_NN_COMMAND(SwinAttentionScoreQuant);
ACL_NN_COMMAND(Baddbmm);
ACL_NN_COMMAND(InplaceBaddbmm);
ACL_NN_COMMAND(InplaceMaskedFillScalar);
ACL_NN_COMMAND(WeightQuantBatchMatmul);
// ACL_NN_COMMAND(TransQuantParam); // no stream
ACL_NN_COMMAND(MinDim);
ACL_NN_COMMAND(Shrink);
ACL_NN_COMMAND(MaxUnpool3d);
ACL_NN_COMMAND(ApplyRotaryPosEmb);
ACL_NN_COMMAND(ForeachSubScalarV2);
ACL_NN_COMMAND(DeepNorm);
ACL_NN_COMMAND(ForeachAddListV2);
ACL_NN_COMMAND(KlDiv);
ACL_NN_COMMAND(ForeachLerpScalar);
ACL_NN_COMMAND(Amax);
ACL_NN_COMMAND(Im2colBackward);
ACL_NN_COMMAND(Kthvalue);
ACL_NN_COMMAND(AvgPool2dBackward);
ACL_NN_COMMAND(FFN);
ACL_NN_COMMAND(GridSampler2D);
ACL_NN_COMMAND(MatmulAllReduceAddRmsNorm);
ACL_NN_COMMAND(ForeachPowList);
ACL_NN_COMMAND(Maximum);
ACL_NN_COMMAND(GroupNormSwish);
ACL_NN_COMMAND(ForeachCosh);
ACL_NN_COMMAND(Log1p);
ACL_NN_COMMAND(InplaceLog1p);
ACL_NN_COMMAND(Triu);
ACL_NN_COMMAND(InplaceTriu);
ACL_NN_COMMAND(FakeQuantPerTensorAffineCachemask);
ACL_NN_COMMAND(ForeachTanh);
ACL_NN_COMMAND(NeScalar);
ACL_NN_COMMAND(InplaceNeScalar);
ACL_NN_COMMAND(GatherNd);
ACL_NN_COMMAND(IncreFlashAttention);
ACL_NN_COMMAND(Dot);
ACL_NN_COMMAND(ForeachDivList);
ACL_NN_COMMAND(Max);
ACL_NN_COMMAND(UpsampleBilinear2dAABackward);
ACL_NN_COMMAND(Matmul);
ACL_NN_COMMAND(Repeat);
ACL_NN_COMMAND(UniqueConsecutive);
ACL_NN_COMMAND(DiagFlat);
ACL_NN_COMMAND(Equal);
ACL_NN_COMMAND(ReflectionPad3dBackward);
ACL_NN_COMMAND(ScatterAdd);
ACL_COMMAND(AclNNRfft1DCommand, AclCommand, aclnnStatus, NnOp::Rfft1D, void *, workspace, uint64_t, workspaceSize, aclOpExecutor *, executor);
ACL_NN_COMMAND(FlashAttentionScoreGrad);
ACL_NN_COMMAND(FlashAttentionUnpaddingScoreGrad);
ACL_NN_COMMAND(FlashAttentionScoreGradV2);
ACL_NN_COMMAND(FlashAttentionUnpaddingScoreGradV2);
ACL_NN_COMMAND(BitwiseAndTensor);
ACL_NN_COMMAND(InplaceBitwiseAndTensor);
ACL_NN_COMMAND(Softshrink);
ACL_NN_COMMAND(UpsampleNearestExact2dBackward);
ACL_NN_COMMAND(InplaceOne);
ACL_NN_COMMAND(BackgroundReplace);
ACL_NN_COMMAND(Clamp);
ACL_NN_COMMAND(ClampMin);
ACL_NN_COMMAND(ClampMinTensor);
ACL_NN_COMMAND(InplaceClampMinTensor);
ACL_NN_COMMAND(ClampTensor);
ACL_NN_COMMAND(ClampMax);
ACL_NN_COMMAND(InplaceClampMax);
ACL_NN_COMMAND(ClampMaxTensor);
ACL_NN_COMMAND(InplaceClampMaxTensor);
ACL_NN_COMMAND(ForeachExpm1);
ACL_NN_COMMAND(MaskedSoftmaxWithRelPosBias);
ACL_NN_COMMAND(ForeachPowScalar);
ACL_NN_COMMAND(AffineGrid);
ACL_NN_COMMAND(BatchMatMul);
ACL_NN_COMMAND(ForeachSubScalarList);
ACL_NN_COMMAND(Erf);
ACL_NN_COMMAND(InplaceErf);
ACL_NN_COMMAND(LeTensor);
ACL_NN_COMMAND(InplaceLeTensor);
ACL_NN_COMMAND(GeluBackward);
ACL_NN_COMMAND(MatmulAllReduce);
ACL_NN_COMMAND(EqScalar);
ACL_NN_COMMAND(InplaceEqScalar);
ACL_NN_COMMAND(AscendAntiQuant);
ACL_NN_COMMAND(NonzeroV2);
ACL_NN_COMMAND(Cummin);
ACL_NN_COMMAND(MaxPool3dWithArgmaxBackward);
ACL_NN_COMMAND(MoeInitRouting);
ACL_NN_COMMAND(ArgMin);
ACL_NN_COMMAND(Unique2);
ACL_NN_COMMAND(Inverse);
ACL_NN_COMMAND(UniqueDim);
ACL_NN_COMMAND(GroupNormSilu);
ACL_NN_COMMAND(GroupNormSiluV2);
ACL_NN_COMMAND(RmsNorm);
ACL_NN_COMMAND(AddRmsNorm);
ACL_NN_COMMAND(MoeTokenUnpermute);
ACL_NN_COMMAND(UpsampleTrilinear3dBackward);
ACL_NN_COMMAND(RemainderTensorTensor);
ACL_NN_COMMAND(RemainderTensorScalar);
ACL_NN_COMMAND(RemainderScalarTensor);
ACL_NN_COMMAND(InplaceRemainderTensorTensor);
ACL_NN_COMMAND(InplaceRemainderTensorScalar);
ACL_NN_COMMAND(ForeachSqrt);
ACL_NN_COMMAND(Log2);
ACL_NN_COMMAND(InplaceLog2);
ACL_NN_COMMAND(XLogYScalarSelf);
ACL_COMMAND(AclNNStftCommand, AclCommand, aclnnStatus, NnOp::Stft, void *, workspace, uint64_t, workspaceSize, aclOpExecutor *, executor);
ACL_NN_COMMAND(ForeachLog1p);
ACL_NN_COMMAND(IsClose);
ACL_NN_COMMAND(ForeachMinimumList);
ACL_NN_COMMAND(MoeTokenPermuteGrad);
ACL_NN_COMMAND(ReplicationPad3d);
ACL_NN_COMMAND(Swish);
ACL_NN_COMMAND(IndexPutImpl);
ACL_NN_COMMAND(ForeachRoundOffNumberV2);
ACL_NN_COMMAND(AddLayerNorm);
ACL_NN_COMMAND(IncreFlashAttentionV3);
ACL_NN_COMMAND(BatchMatMulReduceScatterAlltoAll);
ACL_NN_COMMAND(ForeachMulScalarV2);
ACL_NN_COMMAND(UpsampleNearestExact1dBackward);
ACL_NN_COMMAND(AvgPool3d);
ACL_NN_COMMAND(Cat);
ACL_NN_COMMAND(Asin);
ACL_NN_COMMAND(InplaceAsin);
ACL_NN_COMMAND(Exp);
ACL_NN_COMMAND(InplaceExp);
ACL_NN_COMMAND(MultiScaleDeformableAttentionGrad);
ACL_NN_COMMAND(UpsampleTrilinear3d);
ACL_NN_COMMAND(MaskedSelect);
ACL_NN_COMMAND(InplaceWeightQuantMatmulAllReduceAddRmsNorm);
ACL_NN_COMMAND(ForeachAddScalarList);
ACL_NN_COMMAND(Neg);
ACL_NN_COMMAND(InplaceNeg);
ACL_NN_COMMAND(SWhere);
ACL_NN_COMMAND(Cos);
ACL_NN_COMMAND(InplaceCos);
ACL_NN_COMMAND(ForeachAddcdivScalarV2);
ACL_NN_COMMAND(Gemm);
ACL_NN_COMMAND(ReflectionPad3d);
ACL_NN_COMMAND(MatmulAllReduceV2);
ACL_NN_COMMAND(MseLossBackward);
ACL_NN_COMMAND(Expand);
ACL_NN_COMMAND(BidirectionLSTMV2);
ACL_NN_COMMAND(ForeachRoundOffNumber);
ACL_NN_COMMAND(EmbeddingRenorm);
ACL_NN_COMMAND(Slogdet);
ACL_NN_COMMAND(Gelu);
ACL_NN_COMMAND(QuantMatmulV4);
ACL_NN_COMMAND(Sin);
ACL_NN_COMMAND(InplaceSin);
ACL_NN_COMMAND(Acos);
ACL_NN_COMMAND(InplaceAcos);
ACL_NN_COMMAND(Prelu);
ACL_NN_COMMAND(AscendQuantV3);
ACL_NN_COMMAND(Round);
ACL_NN_COMMAND(InplaceRound);
ACL_NN_COMMAND(RoundDecimals);
ACL_NN_COMMAND(InplaceRoundDecimals);
ACL_NN_COMMAND(MinN);
ACL_NN_COMMAND(MultiScaleDeformableAttnFunction);
ACL_NN_COMMAND(Flip);
ACL_NN_COMMAND(Expm1);
ACL_NN_COMMAND(InplaceExpm1);
ACL_NN_COMMAND(LinalgVectorNorm);
ACL_NN_COMMAND(LogSumExp);
ACL_NN_COMMAND(FakeQuantPerChannelAffineCachemask);
ACL_NN_COMMAND(NLLLoss2d);
ACL_NN_COMMAND(ForeachPowScalarList);
ACL_NN_COMMAND(LogSoftmaxBackward);
ACL_NN_COMMAND(CtcLossBackward);
ACL_NN_COMMAND(LtScalar);
ACL_NN_COMMAND(InplaceLtScalar);
ACL_NN_COMMAND(RmsNormGrad);
ACL_NN_COMMAND(Relu);
ACL_NN_COMMAND(InplaceRelu);
ACL_NN_COMMAND(GroupedMatmulV3);
ACL_NN_COMMAND(ConvolutionBackward);
ACL_NN_COMMAND(ConvTbcBackward);
ACL_NN_COMMAND(GroupNormBackward);
ACL_NN_COMMAND(Slice);
ACL_NN_COMMAND(CircularPad2dBackward);
ACL_NN_COMMAND(MultilabelMarginLoss);
ACL_NN_COMMAND(UpsampleBilinear2dAA);
ACL_NN_COMMAND(WeightQuantMatmulAllReduce);
ACL_NN_COMMAND(IsNegInf);
ACL_NN_COMMAND(LogAddExp);
ACL_NN_COMMAND(FusedInferAttentionScoreV2);
ACL_NN_COMMAND(GtScalar);
ACL_NN_COMMAND(InplaceGtScalar);
ACL_NN_COMMAND(Logdet);
ACL_NN_COMMAND(Any);
ACL_NN_COMMAND(Addmm);
ACL_NN_COMMAND(InplaceAddmm);
ACL_NN_COMMAND(Selu);
ACL_NN_COMMAND(InplaceSelu);
ACL_NN_COMMAND(SoftMarginLoss);
ACL_NN_COMMAND(UpsampleNearest1d);
ACL_NN_COMMAND(Asinh);
ACL_NN_COMMAND(InplaceAsinh);
ACL_NN_COMMAND(Flatten);
ACL_NN_COMMAND(BinaryCrossEntropyWithLogits);
ACL_NN_COMMAND(QuantMatmulAllReduceV2);
ACL_NN_COMMAND(LogSoftmax);
ACL_NN_COMMAND(MaxUnpool3dBackward);
ACL_NN_COMMAND(GroupNorm);
ACL_NN_COMMAND(Take);
ACL_NN_COMMAND(GeluV2);
ACL_NN_COMMAND(Roll);
ACL_NN_COMMAND(ForeachAcos);
ACL_NN_COMMAND(ForeachAddcdivScalar);
ACL_NN_COMMAND(ForeachMaximumScalarList);
ACL_NN_COMMAND(ForeachSubScalar);
ACL_NN_COMMAND(PowTensorScalar);
ACL_NN_COMMAND(InplacePowTensorScalar);
ACL_NN_COMMAND(PowScalarTensor);
ACL_NN_COMMAND(Linspace);
ACL_NN_COMMAND(EluBackward);
ACL_NN_COMMAND(SilentCheck);
ACL_NN_COMMAND(BidirectionLSTM);
ACL_NN_COMMAND(QuantMatmulAllReduce);
ACL_NN_COMMAND(ForeachAddList);
ACL_NN_COMMAND(InplaceFillScalar);
ACL_NN_COMMAND(MrgbaCustom);
ACL_NN_COMMAND(IndexAdd);
ACL_NN_COMMAND(Rsqrt);
ACL_NN_COMMAND(InplaceRsqrt);
ACL_NN_COMMAND(SwiGlu);
ACL_NN_COMMAND(FlashAttentionScore);
ACL_NN_COMMAND(FlashAttentionVarLenScore);
ACL_NN_COMMAND(FlashAttentionScoreV2);
ACL_NN_COMMAND(FlashAttentionVarLenScoreV2);
ACL_NN_COMMAND(ForeachAsin);
ACL_NN_COMMAND(GeluBackwardV2);
ACL_NN_COMMAND(Softmax);
ACL_NN_COMMAND(Einsum);
ACL_NN_COMMAND(Softplus);
ACL_NN_COMMAND(ReplicationPad2d);
ACL_NN_COMMAND(ForeachDivScalar);
ACL_NN_COMMAND(Signbit);
ACL_NN_COMMAND(Sinc);
ACL_NN_COMMAND(InplaceSinc);
ACL_NN_COMMAND(BatchNormStats);
ACL_NN_COMMAND(GeGlu);
ACL_NN_COMMAND(GeGluV3);
ACL_NN_COMMAND(Digamma);
ACL_NN_COMMAND(ForeachLog2);
ACL_NN_COMMAND(Gather);
ACL_NN_COMMAND(AvgPool2d);
ACL_NN_COMMAND(AscendQuant);
ACL_NN_COMMAND(Std);
ACL_NN_COMMAND(SwiGluGrad);
ACL_NN_COMMAND(Hardtanh);
ACL_NN_COMMAND(InplaceHardtanh);
ACL_NN_COMMAND(UpsampleBicubic2dAAGrad);
ACL_NN_COMMAND(Sqrt);
ACL_NN_COMMAND(InplaceSqrt);
ACL_NN_COMMAND(Permute);
ACL_NN_COMMAND(ReflectionPad2d);
ACL_NN_COMMAND(LogicalNot);
ACL_NN_COMMAND(InplaceLogicalNot);
ACL_NN_COMMAND(NonMaxSuppression);
ACL_NN_COMMAND(Topk);
ACL_NN_COMMAND(Div);
ACL_NN_COMMAND(Divs);
ACL_NN_COMMAND(DivMod);
ACL_NN_COMMAND(DivMods);
ACL_NN_COMMAND(InplaceDiv);
ACL_NN_COMMAND(InplaceDivs);
ACL_NN_COMMAND(InplaceDivMod);
ACL_NN_COMMAND(InplaceDivMods);
ACL_NN_COMMAND(Mv);
ACL_NN_COMMAND(Qr);
ACL_NN_COMMAND(ScatterNdUpdate);
ACL_NN_COMMAND(NLLLossBackward);
ACL_NN_COMMAND(UpsampleBicubic2dAA);
ACL_NN_COMMAND(Threshold);
ACL_NN_COMMAND(InplaceThreshold);
ACL_NN_COMMAND(ForeachReciprocal);

} // namespace xsched::ascend
