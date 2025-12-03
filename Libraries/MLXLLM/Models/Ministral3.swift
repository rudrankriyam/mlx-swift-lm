// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/ministral3.py

private func getLlama4AttnScale(
    start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int
) -> MLXArray {
    let positions = MLXArray(stride(from: start, to: stop, by: 1))
    let scaling = 1.0 + beta * MLX.log(1.0 + MLX.floor(positions / Float(maxPositionEmbeddings)))
    return scaling.expandedDimensions(axis: -1)
}

/// Custom message generator for Ministral3 that formats messages in Llama-3 style.
/// Since the model doesn't include a chat template, this formats the prompt manually.
/// Format: <s>[INST] user_msg [/INST] assistant_response
private struct Ministral3MessageGenerator: MessageGenerator {
    init() {}

    func generate(message: Chat.Message) -> Message {
        // Extract text content
        let text: String
        switch message.role {
        case .system:
            // System messages are prepended to the first user message
            text = "<<SYS>>\n\(message.content)\n<</SYS>>\n\n"
        case .user:
            text = "[INST] \(message.content) [/INST]"
        case .assistant:
            text = " \(message.content)"
        case .tool:
            text = "\n\(message.content)"
        }
        return ["role": message.role.rawValue, "content": text]
    }

    func generate(messages: [Chat.Message]) -> [Message] {
        var formattedMessages: [Message] = []
        var systemMessage: String = ""

        // Extract system message if present
        for message in messages {
            if message.role == .system {
                systemMessage = message.content
                break
            }
        }

        // Format messages
        for (index, message) in messages.enumerated() {
            if message.role == .system {
                continue // Already handled
            }

            var formattedMessage = generate(message: message)

            // If this is the first user message and we have a system message, prepend it
            if message.role == .user && !systemMessage.isEmpty && index == 0 {
                let systemText = "<<SYS>>\n\(systemMessage)\n<</SYS>>\n\n"
                if var content = formattedMessage["content"] as? String {
                    content = content.replacingOccurrences(of: "[INST]", with: "\(systemText)[INST]")
                    formattedMessage["content"] = content
                }
            }

            formattedMessages.append(formattedMessage)
        }

        return formattedMessages
    }
}

private class Attention: Module {

    let args: Ministral3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(_ args: Ministral3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        var offset = 0
        if let cache {
            offset = cache.offset
        }
        queries = MLXFast.RoPE(queries, dimensions: args.resolvedHeadDimensions, traditional: args.ropeTraditional, base: args.ropeTheta, scale: 1.0, offset: offset)
        keys = MLXFast.RoPE(keys, dimensions: args.resolvedHeadDimensions, traditional: args.ropeTraditional, base: args.ropeTheta, scale: 1.0, offset: offset)

        queries = queries * attnScale

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

        return wo(output)
    }
}

private class MLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: Ministral3Configuration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = silu(gate(x))
        return down(activation * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let useSliding: Bool

    init(_ args: Ministral3Configuration, useSliding: Bool = false) {
        self.useSliding = useSliding
        self._attention.wrappedValue = Attention(args)
        self._mlp.wrappedValue = MLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), attnScale: attnScale, mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

private class Ministral3ModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [TransformerBlock]
    let norm: RMSNorm
    let layerTypes: [String]
    let slidingWindow: Int?
    let faIndex: Int
    let swaIndex: Int?
    let args: Ministral3Configuration

    init(_ args: Ministral3Configuration) {
        precondition(args.vocabularySize > 0)

        self.args = args
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layerTypes = args.layerTypes ?? Array(repeating: "full_attention", count: args.hiddenLayers)
        self.slidingWindow = args.slidingWindow

        self.layers = self.layerTypes.map { layerType in
            TransformerBlock(args, useSliding: layerType == "sliding_attention")
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        // Find indices for different layer types
        self.faIndex = self.layerTypes.firstIndex(of: "full_attention") ?? 0
        self.swaIndex = self.layers.firstIndex(where: { $0.useSliding })
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let maskFullAttn = createAttentionMask(h: h, cache: cache != nil ? [cache![faIndex]] : nil)

        var maskSliding: MLXFast.ScaledDotProductAttentionMaskMode?
        if let swaIndex = swaIndex {
            maskSliding = createAttentionMask(h: h, cache: cache != nil ? [cache![swaIndex]] : nil)
        }

        let attnScale = getLlama4AttnScale(
            start: cache?[0].offset ?? 0,
            stop: (cache?[0].offset ?? 0) + inputs.dim(1),
            beta: args.ropeParameters?["llama_4_scaling_beta"]?.asFloat() ?? 0.0,
            maxPositionEmbeddings: args.ropeParameters?["original_max_position_embeddings"]?.asInt() ?? 2048
        ).asType(h.dtype)

        for (i, layer) in layers.enumerated() {
            let mask = layer.useSliding ? (maskSliding ?? .none) : maskFullAttn
            h = layer(h, attnScale: attnScale, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

/// Model for Ministral3.
public class Ministral3Model: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: Ministral3ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let args: Ministral3Configuration

    public init(_ args: Ministral3Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Ministral3ModelInner(args)
        self.args = args
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Handle weight key prefixes for multimodal models
        // For models converted with text_model prefix (like Ministral3 on HuggingFace),
        // we need to rename the prefix from "text_model." to "model."

        var processedWeights = weights

        // Check if weights have text_model prefix and handle it
        // Convert "text_model.embed_tokens.weight" -> "model.embed_tokens.weight"
        // Convert "text_model.norm.weight" -> "model.norm.weight"
        for (key, value) in weights {
            if key.hasPrefix("text_model.") {
                let newKey = "model." + key.dropFirst("text_model.".count)
                processedWeights[newKey] = value
                processedWeights.removeValue(forKey: key)
            } else if key.hasPrefix("language_model.") {
                let newKey = "model." + key.dropFirst("language_model.".count)
                processedWeights[newKey] = value
                processedWeights.removeValue(forKey: key)
            }
        }

        // Remove unused precomputed rotary frequencies and scales
        let sanitized = processedWeights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq") &&
            !$0.key.contains("activation_scale")
        }

        // Handle weight scale inversions
        var newWeights: [String: MLXArray] = [:]
        for (key, value) in sanitized {
            if key.contains("_scale_inv") {
                let baseKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = sanitized[baseKey] {
                    newWeights[baseKey] = weight * value
                }
            } else if !key.contains("_scale_inv") && !key.contains("activation_scale") {
                newWeights[key] = value
            }
        }

        // Remove lm_head.weight if tying word embeddings
        if args.tieWordEmbeddings {
            newWeights.removeValue(forKey: "lm_head.weight")
        }

        return newWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let layerTypes = args.layerTypes ?? Array(repeating: "full_attention", count: args.hiddenLayers)
        return layerTypes.map { layerType in
            if layerType == "sliding_attention", let slidingWindow = args.slidingWindow {
                return RotatingKVCache(maxSize: slidingWindow)
            } else {
                return KVCacheSimple()
            }
        }
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        // Check if model supports system messages
        do {
            let probe = [["role": "system", "content": "test"]]
            _ = try tokenizer.applyChatTemplate(messages: probe)
            return DefaultMessageGenerator()
        } catch {
            // No chat template - use custom formatter for Llama-3/Ministral-3 format
            return Ministral3MessageGenerator()
        }
    }
}

public struct Ministral3Configuration: Codable, Sendable {

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]?
    var ropeParameters: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool = true
    var layerTypes: [String]?
    var slidingWindow: Int?
    var attentionBias: Bool = false
    var mlpBias: Bool = false

    public init(
        hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int, attentionHeads: Int,
        headDimensions: Int? = nil, rmsNormEps: Float, vocabularySize: Int, kvHeads: Int,
        maxPositionEmbeddings: Int? = nil, ropeTheta: Float = 10_000, ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil, ropeParameters: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true, layerTypes: [String]? = nil, slidingWindow: Int? = nil,
        attentionBias: Bool = false, mlpBias: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.ropeParameters = ropeParameters
        self.tieWordEmbeddings = tieWordEmbeddings
        self.layerTypes = layerTypes
        self.slidingWindow = slidingWindow
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
    }

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        ropeParameters = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
    }
}

// MARK: - LoRA

extension Ministral3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
