//
//  GLM4MOELite.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm4_moe_lite.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class GLM4MoELiteAttention: Module {
    let config: GLM4MoELiteConfiguration
    let hiddenSize: Int
    let numHeads: Int
    let maxPositionEmbeddings: Int
    let ropeTheta: Float
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    var scale: Float

    let rope: OffsetLayer

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    init(_ config: GLM4MoELiteConfiguration) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.ropeTheta = config.ropeTheta
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

        self.scale = pow(Float(qHeadDim), -0.5)

        if let qLoraRank = qLoraRank {
            self._qAProj.wrappedValue = Linear(
                hiddenSize, qLoraRank, bias: config.attentionBias
            )
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: config.rmsNormEps)
            self._qBProj.wrappedValue = Linear(
                qLoraRank, numHeads * qHeadDim, bias: false
            )
        } else {
            self._qProj.wrappedValue = Linear(hiddenSize, numHeads * qHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize,
            kvLoraRank + qkRopeHeadDim,
            bias: config.attentionBias
        )
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        self._kvBProj.wrappedValue = Linear(
            kvLoraRank,
            numHeads * (qHeadDim - qkRopeHeadDim + vHeadDim),
            bias: false
        )
        self._oProj.wrappedValue = Linear(
            numHeads * vHeadDim, hiddenSize, bias: config.attentionBias
        )

        if let ropeScaling = config.ropeScaling {
            let mscaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0
            if mscaleAllDim != 0 {
                guard let scalingFactor = ropeScaling["factor"]?.asFloat() else {
                    fatalError("rope_scaling.factor is required when mscale_all_dim is set")
                }
                if scalingFactor > 1 {
                    let s = 0.1 * mscaleAllDim * log(scalingFactor) + 1.0
                    self.scale = self.scale * s * s
                }
            }
        }

        self.rope = initializeRope(
            dims: qkRopeHeadDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q: MLXArray
        if let qProj {
            q = qProj(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }

        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = splitQ[0]
        var qPe = splitQ[1]

        var compressedKv = kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let splitKv = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = splitKv[0]
        let values = splitKv[1]

        let offset = cache?.offset ?? 0
        qPe = rope(qPe, offset: offset)
        kPe = rope(kPe, offset: offset)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        let keys = concatenated([kNope, kPe], axis: -1)
        let queries = concatenated([qNope, qPe], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

class GLM4MoELiteMLP: Module, UnaryLayer {
    let hiddenSize: Int
    let intermediateSize: Int

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: GLM4MoELiteConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        self.hiddenSize = hiddenSize ?? config.hiddenSize
        self.intermediateSize = intermediateSize ?? config.intermediateSize

        _gateProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class GLM4MoELiteGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int
    let selectExperts: @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray)

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: GLM4MoELiteConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoELiteGate requires nRoutedExperts")
        }

        precondition(config.topkMethod == "noaux_tc", "Unsupported topk method.")

        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        _weight.wrappedValue = zeros([nRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([nRoutedExperts])

        let topK = config.numExpertsPerTok
        let normTopkProb = config.normTopkProb
        let nGroup = config.nGroup
        let topkGroup = config.topkGroup
        let routedScalingFactor = config.routedScalingFactor

        let compiledSelect = compile { inputs in
            let gates = inputs[0]
            let correctionBias = inputs[1]

            var scores = sigmoid(gates.asType(.float32))
            let originalScores = scores
            scores = scores + correctionBias
            if nGroup > 1 {
                scores = unflatten(scores, axis: -1, shape: [nGroup, -1])
                let groupScores = top(scores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
                let k = nGroup - topkGroup
                let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
                scores = putAlong(
                    scores, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
                scores = flattened(scores, start: -2, end: -1)
            }

            let k = topK
            let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
            var selectedScores = takeAlong(originalScores, inds, axis: -1)

            if topK > 1, normTopkProb {
                let denominator = selectedScores.sum(axis: -1, keepDims: true)
                selectedScores = selectedScores / denominator
            }
            selectedScores = selectedScores * routedScalingFactor

            return [inds, selectedScores]
        }

        self.selectExperts = { gates, correctionBias in
            let outputs = compiledSelect([gates, correctionBias])
            return (outputs[0], outputs[1])
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        selectExperts(x.matmul(weight.T), eScoreCorrectionBias)
    }
}

class GLM4MoELiteMoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    let gate: GLM4MoELiteGate

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM4MoELiteMLP?

    init(_ config: GLM4MoELiteConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoELiteMoE requires nRoutedExperts")
        }

        self.numExpertsPerTok = config.numExpertsPerTok
        self.gate = GLM4MoELiteGate(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: nRoutedExperts
        )

        if let shared = config.nSharedExperts, shared > 0 {
            let intermediateSize = config.moeIntermediateSize * shared
            _sharedExperts.wrappedValue = GLM4MoELiteMLP(
                config, intermediateSize: intermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class GLM4MoELiteDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: GLM4MoELiteAttention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: GLM4MoELiteConfiguration, layerIdx: Int) {
        _selfAttn.wrappedValue = GLM4MoELiteAttention(config)

        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = GLM4MoELiteMoE(config)
        } else {
            self.mlp = GLM4MoELiteMLP(config)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class GLM4MoELiteModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [GLM4MoELiteDecoderLayer]
    let norm: RMSNorm

    init(_ config: GLM4MoELiteConfiguration) {
        precondition(config.vocabSize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.numHiddenLayers)
            .map { idx in
                GLM4MoELiteDecoderLayer(config, layerIdx: idx)
            }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class GLM4MoELiteModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: GLM4MoELiteModelInner
    let configuration: GLM4MoELiteConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: GLM4MoELiteConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self.model = GLM4MoELiteModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        for l in 0 ..< configuration.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for n in ["gate_proj", "down_proj", "up_proj"] {
                for k in ["weight", "scales", "biases"] {
                    let key = "\(prefix).mlp.experts.0.\(n).\(k)"
                    if sanitized[key] != nil, let nRoutedExperts = configuration.nRoutedExperts {
                        let toJoin = (0 ..< nRoutedExperts).map { e in
                            sanitized.removeValue(
                                forKey: "\(prefix).mlp.experts.\(e).\(n).\(k)")!
                        }
                        sanitized["\(prefix).mlp.switch_mlp.\(n).\(k)"] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        let numNextnPredictLayers = configuration.numNextnPredictLayers
        if numNextnPredictLayers > 0 {
            sanitized = sanitized.filter { key, _ in
                for idx in 0 ..< numNextnPredictLayers {
                    if key.hasPrefix("model.layers.\(configuration.numHiddenLayers + idx)") {
                        return false
                    }
                }
                return true
            }
        }

        return sanitized
    }
}

public struct GLM4MoELiteConfiguration: Codable, Sendable {
    var modelType: String = "glm4_moe_lite"
    var vocabSize: Int = 154_880
    var hiddenSize: Int = 2_048
    var intermediateSize: Int = 10_240
    var moeIntermediateSize: Int = 1_536
    var numHiddenLayers: Int = 47
    var numAttentionHeads: Int = 20
    var numKeyValueHeads: Int = 20
    var nSharedExperts: Int? = 1
    var nRoutedExperts: Int? = 64
    var routedScalingFactor: Float = 1.8
    var kvLoraRank: Int = 512
    var qLoraRank: Int? = 768
    var qkRopeHeadDim: Int = 64
    var qkNopeHeadDim: Int = 192
    var vHeadDim: Int = 256
    var topkMethod: String = "noaux_tc"
    var scoringFunc: String = "sigmoid"
    var normTopkProb: Bool = true
    var nGroup: Int = 1
    var topkGroup: Int = 1
    var numExpertsPerTok: Int = 4
    var moeLayerFreq: Int = 1
    var firstKDenseReplace: Int = 1
    var maxPositionEmbeddings: Int = 202_752
    var rmsNormEps: Float = 1e-5
    var ropeTheta: Float = 1_000_000.0
    var ropeScaling: [String: StringOrNumber]? = nil
    var ropeTraditional: Bool = true
    var attentionBias: Bool = false
    var attentionDropout: Float = 0.0
    var partialRotaryFactor: Float = 1.0
    var tieWordEmbeddings: Bool = false
    var numNextnPredictLayers: Int = 1

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeTraditional = "rope_traditional"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? vocabSize
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
            ?? moeIntermediateSize
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? numHiddenLayers
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
            ?? numAttentionHeads
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? numKeyValueHeads
        if container.contains(.nSharedExperts) {
            self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        } else {
            self.nSharedExperts = 1
        }
        if container.contains(.nRoutedExperts) {
            self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        } else {
            self.nRoutedExperts = 64
        }
        self.routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor)
            ?? routedScalingFactor
        self.kvLoraRank = try container.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? kvLoraRank
        if container.contains(.qLoraRank) {
            self.qLoraRank = try container.decodeIfPresent(Int.self, forKey: .qLoraRank)
        } else {
            self.qLoraRank = 768
        }
        self.qkRopeHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? qkRopeHeadDim
        self.qkNopeHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? qkNopeHeadDim
        self.vHeadDim = try container.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? vHeadDim
        self.topkMethod = try container.decodeIfPresent(String.self, forKey: .topkMethod)
            ?? topkMethod
        self.scoringFunc = try container.decodeIfPresent(String.self, forKey: .scoringFunc)
            ?? scoringFunc
        self.normTopkProb =
            try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? normTopkProb
        self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? nGroup
        self.topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? topkGroup
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
            ?? numExpertsPerTok
        self.moeLayerFreq =
            try container.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? moeLayerFreq
        self.firstKDenseReplace =
            try container.decodeIfPresent(Int.self, forKey: .firstKDenseReplace)
            ?? firstKDenseReplace
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? maxPositionEmbeddings
        self.rmsNormEps =
            try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? ropeTraditional
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? attentionBias
        self.attentionDropout =
            try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? attentionDropout
        self.partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
            ?? partialRotaryFactor
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? tieWordEmbeddings
        self.numNextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers)
            ?? numNextnPredictLayers
    }
}

// MARK: - LoRA

extension GLM4MoELiteModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
