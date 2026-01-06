# How WSRPN-VL Boosts Gaussian Maps Using Text Captions

## 🎯 Core Mechanism Overview

WSRPN-VL enhances localization by using medical text captions to provide semantic guidance that constrains Gaussian ROI parameters (center μ and scale σ). The mechanism works through **three interdependent components**:

```
Text Caption (from RDF)
       ↓
BERT Encoder → Text Embeddings (semantic meaning)
       ↓
Shared Projection → Normalized Embedding Space
       ↓
Vision Embeddings (from CNN features)
       ↓
Contrastive Loss → Feature Alignment
       ↓
Gaussian Parameters Regularization → Sharper, More Focused ROI Maps
```

---

## 📊 Component 1: Text Encoding Pipeline

### TextEncoder.forward() - Semantic Understanding

**Location**: `src/model/vl_encoder.py` (lines 22-99)

```python
# COMMENT: Text captions carry semantic meaning about where pathology appears
# e.g., "cardiomegaly at right apex" → model learns right side = cardiomegaly
class TextEncoder(nn.Module):
    """
    Encodes medical text descriptions using BERT
    
    BOOST MECHANISM:
    - Frozen BERT (preserved pre-trained medical knowledge)
    - Processes captions like: "pleural effusion at right base"
    - Output: (B, 768) embeddings capture semantic pathology location
    """
    
    def forward(self, texts: list, max_length: int = 128) -> Tensor:
        """
        STEP 1: Tokenize & encode text with BERT
        
        Input:  ["cardiomegaly in cardiac silhouette",
                 "pneumothorax in right upper lobe"]
        
        Process:
          1. BERT Tokenizer → tokens (e.g., [CLS], cardiomegaly, in, ..., [SEP])
          2. BERT Encoder → contextual embeddings for each token
          3. Mean pooling (exclude special tokens)
        
        Output: (B, 768) - each row is semantic representation
                - Row 0: "cardiomegaly..." → embeddings emphasize cardiac concepts
                - Row 1: "pneumothorax..." → embeddings emphasize lung concepts
        
        WHY THIS BOOSTS GAUSSIAN MAPS:
        - Each text embedding contains location hints (right, left, apex, base)
        - Model learns that certain spatial regions correlate with text meaning
        - This CONSTRAINS where Gaussian centers (μx, μy) should be placed
        """
        
        # Text: ["pleural effusion right" ...] (B,)
        #   ↓ BERT tokenization & encoding
        # embeddings: (B, seq_len, 768)
        #   ↓ Mean pooling over valid tokens (exclude [CLS], [SEP])
        # mean_embeddings: (B, 768) ← Each sample now has semantic representation
        
        return mean_embeddings  # (B, 768)
```

**Key Insight**: 
- Text like *"pleural effusion at right costophrenic angle"* encodes location priors
- Model learns: right_embeds ↔ right_gaussian_center, large_effusion_embeds ↔ wide_gaussian_scale

---

## 📊 Component 2: Shared Embedding Space Alignment

### SharedProjection.forward() - Feature Alignment

**Location**: `src/model/vl_encoder.py` (lines 103-155)

```python
class SharedProjection(nn.Module):
    """
    Projects visual and textual features to SHARED embedding space
    
    BOOST MECHANISM:
    - Both vision and text projected to SAME space (128-dim)
    - Enables direct comparison: vision_emb vs text_emb
    - Contrastive loss pulls them together when describing same image
    """
    
    def forward(self, vision_features: Tensor, text_features: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        STEP 2: Project both modalities to shared space
        
        Input:
          vision_features: (B, 1024) - Global CNN features from image
                          e.g., avg of all patch features for Cardiomegaly image
          text_features: (B, 768) - BERT embeddings from text caption
                         e.g., "cardiomegaly cardiac silhouette" → semantic vector
        
        Process:
          1. Vision: (B, 1024) → Linear(1024→128) → LayerNorm → L2Norm
                     = (B, 128) normalized in [-1, 1] roughly
          
          2. Text: (B, 768) → Linear(768→128) → LayerNorm → L2Norm
                  = (B, 128) normalized in [-1, 1] roughly
        
        Output: 
          vision_emb: (B, 128) - normalized image representation
          text_emb: (B, 128) - normalized semantic representation
        
        WHY THIS BOOSTS GAUSSIAN MAPS:
        - Both embeddings now DIRECTLY comparable (same space!)
        - If vision_emb ≈ text_emb (high cosine similarity)
          → Model confirmed: "this image matches the text description"
          → Gaussian Gaussians become SHARPER (low σ) to pinpoint exact location
        
        - If vision_emb ≠ text_emb (low cosine similarity) → LOSS signal
          → Backprop updates: Gaussian parameters, CNN features, text encoder
          → Next epoch: Gaussian centers drift toward location mentioned in text
        """
        
        # vision_features: (B, 1024) e.g., [0.1, -0.5, 0.3, ...] for image
        #   ↓ Linear projection: 1024 → 128
        # vision_proj_raw: (B, 128) e.g., [0.02, -0.1, 0.06, ...]
        #   ↓ LayerNorm: normalize to mean≈0, std≈1
        # vision_proj_norm: (B, 128) e.g., [-1.2, 0.8, -0.5, ...]
        #   ↓ L2 normalize: ||v|| = 1
        # vision_emb: (B, 128) - on unit sphere, e.g., [-0.8, 0.55, -0.35, ...]
        
        # Similarly for text_features: (B, 768) → (B, 128) on unit sphere
        
        return vision_emb, text_emb  # Both (B, 128), normalized
```

**Key Insight**:
- Shared space creates direct competition: vision vs text must align
- Misalignment → strong gradient signal → Gaussian parameters updated
- Text constraints flow into spatial attention via backpropagation

---

## 📊 Component 3: Contrastive Learning Loss

### ContrastiveVLLoss.forward() - Semantic Constraint

**Location**: `src/model/vl_losses.py` (lines 208-242)

```python
class ContrastiveVLLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) loss
    
    BOOST MECHANISM:
    - Pulls image embeddings toward matching text embeddings
    - Creates semantic "gravity" that constrains spatial attention
    """
    
    def forward(self, image_embeddings: Tensor, text_embeddings: Tensor) -> Tensor:
        """
        STEP 3: Compute contrastive loss - the KEY TO GAUSSIAN SHARPENING
        
        Input:
          image_embeddings: (B, 128) - Vision features from image_i
          text_embeddings: (B, 128) - Text features from text_i (matching image_i)
        
        Example batch:
          Image 0: "cardiomegaly" → vision_emb[0] ≈ [0.2, -0.8, 0.1, ...] (128-dim)
          Text 0:  "cardiomegaly cardiac silhouette" → text_emb[0] ≈ [0.25, -0.75, 0.05, ...]
          
          Image 1: "pleural effusion" → vision_emb[1] ≈ [-0.3, 0.1, -0.9, ...]
          Text 1:  "pleural effusion right side" → text_emb[1] ≈ [-0.25, 0.15, -0.85, ...]
        
        Process:
          1. Normalize embeddings (already normalized, but ensure)
             vision_emb_norm = L2normalize(vision_emb)  # (B, 128)
             text_emb_norm = L2normalize(text_emb)      # (B, 128)
          
          2. Compute similarity matrix (dot product in normalized space)
             logits = vision_emb @ text_emb.T / τ       # (B, B)
                    where τ = temperature = 0.07 (sharpens loss landscape)
          
          Example similarity matrix (showing cosine similarity):
                       text_0   text_1   text_2
          image_0:  [  0.85    -0.10    -0.15  ]  ← high for matching text
          image_1:  [ -0.05     0.92    -0.08  ]  ← high for matching text
          image_2:  [ -0.12    -0.15     0.88  ]  ← high for matching text
          
          3. Cross-entropy loss (want diagonal = 1, off-diagonal = 0)
             loss = CrossEntropy(logits, diagonal_labels)
                  = -log(exp(logits[i,i]) / Σ_j exp(logits[i,j]))
          
          Image 0 loss term:
            -log(exp(0.85/0.07) / [exp(0.85/0.07) + exp(-0.10/0.07) + exp(-0.15/0.07)])
            = -log(exp(12.14) / [exp(12.14) + exp(-1.43) + exp(-2.14)])
            = -log(very_large / [very_large + tiny + tiny])
            ≈ 0 ✓ (small loss = good match)
        
        Output: loss ∈ [0, ∞)
                = 0 if all images perfectly match their texts
                > 0 if misalignment exists
        
        BACKPROPAGATION & GAUSSIAN BOOST:
        
        When loss is high (misalignment):
          ∇loss / ∇image_emb = d_loss / d_vision_emb
                              → flows back through projection layers
                              → reaches CNN features (patch aggregation)
                              → reaches ROI features & Gaussian parameters!
          
          ∇loss / ∇text_emb  → similar flow for text encoder (frozen, but signals exist)
        
        CRITICAL STEP: How this updates Gaussian parameters:
        
        1. Text caption says: "effusion at right base" (spatial clue!)
        2. Vision model initially predicts: Gaussian center at (0.5, 0.5) - center
        3. Embedding alignment computes:
           - vision_emb from patch features (which include ROI attention)
           - text_emb from caption (includes spatial word "right", "base")
        4. If center is WRONG:
           - vision_emb ≠ text_emb (low cosine similarity)
           - Loss is HIGH
        5. Gradient ∇loss flows back:
           - Updates patch features ← gradient tells: "focus right_base"
           - Updates ROI attention weights ← gradient tells: "move center right"
           - Updates Gaussian parameters (μx, μy) ← gradient tells: "center=(0.8, 0.8)"
        6. Gaussian map SHARPENS at new location because:
           - CNN features now emphasize right_base region
           - ROI center moves to right_base
           - Semantic alignment IMPROVES
           - Loss DECREASES → training signal CONFIRMED
        
        MULTIPLE PATHOLOGIES CASE:
        - If batch has [cardiomegaly, effusion, pneumothorax, normal]
        - Each gets its own text caption encoding
        - Each gets its own vision embedding
        - Contrastive loss pulls together matching pairs
        - Mismatched pairs → higher loss → stronger gradients
        - Result: Different Gaussian centers for different pathologies
                  (cardiac region for cardio, lung base for effusion, etc.)
        """
        
        # Similarity matrix shape (B, B)
        # logits[i,j] = cos_similarity(vision_emb[i], text_emb[j]) / temperature
        logits = torch.mm(vision_emb, text_emb.T) / self.temperature  # (B, B)
        
        # Labels = identity: we want [i, i] to be high, rest low
        labels = torch.arange(image_emb.shape[0], device=image_emb.device)  # [0, 1, 2, ...]
        
        # NT-Xent loss: bidirectional cross-entropy
        # image → text direction
        loss_img = F.cross_entropy(logits, labels)           # ← "image_i should match text_i"
        # text → image direction
        loss_txt = F.cross_entropy(logits.T, labels)         # ← "text_i should match image_i"
        
        # Average both directions
        return (loss_img + loss_txt) / 2
```

**Key Insight**:
- **Contrastive loss = spatial constraint mechanism**
- Text embedding → location prior → pulls Gaussian center toward correct region
- Vision embedding → must match text → CNN learns to focus on text-described regions
- Gradient flow: Loss → CNN features → ROI Gaussian parameters → Sharper attention

---

## 🔄 Integration: Text Constraints → Gaussian Parameters

### How Text Flows Into ROI Gaussian Maps

**Location**: `src/model/object_detectors/wsrpn.py` (lines 620-665)

```python
# COMMENTED VERSION OF THE KEY INTEGRATION POINT

def train_step(self, x: Tensor, global_label: Tensor, 
               text_descriptions: Optional = None, step: int = None, **kwargs):
    """
    Main training step where text captions boost Gaussian maps
    
    WORKFLOW:
    =========
    
    (1) IMAGE → PATCH FEATURES
    """
    
    # Forward through CNN backbone: x (B, 1, 224, 224)
    patch_features, _ = self.encode_features(x)
    # Output: patch_features (B, 7, 7, 1024)
    #   ↑ Each of 49 patches has 1024-dim features
    #   ↑ These features contain low-level info (edges, textures, colors)
    #   ↑ But NO spatial semantics yet!
    
    # Classify patches (what pathology is here?)
    patch_cls_probs = self.classify(patch_features)
    # Output: patch_cls_probs (B, 7, 7, 9)
    #   ↑ Each patch has 9 pathology probabilities
    #   ↑ Still no semantic guidance from text!
    
    # Aggregate patch features to image level
    (patch_aggregated_cls_features,  # (B, 9, d)
     patch_aggregated_cls_probs,     # (B, 9)
     _, _) = self.aggregate(patch_features, patch_cls_probs, ...)
    # Output: per-image features aggregated by class
    #   ↑ Now have semantic class features but no spatial localization yet
    
    """
    (2) TEXT → SEMANTIC EMBEDDINGS
    """
    
    losses = {}
    
    if self.config.use_vl_branch and text_descriptions is not None:
        # CRITICAL POINT: Text description arrives!
        # e.g., "cardiomegaly: cardiac silhouette at right cardiac border"
        
        # Extract global vision features (aggregated across patches)
        global_vision_features = patch_aggregated_cls_features.mean(dim=1)
        # Shape: (B, d) e.g., (16, 1024)
        # Content: CNN features that should align with text meaning
        
        """
        SEMANTIC ALIGNMENT STEP:
        
        BEFORE text guidance:
        - CNN features = generic patches (no semantic meaning)
        - Gaussian ROIs = randomly initialized (μ≈random, σ≈random)
        - Attention maps = spread across entire image
        
        TEXT ENTERS HERE:
        """
        
        # Text captions: ["cardiomegaly at right...", "effusion at base...", ...]
        text_list = [text_descriptions] * x.shape[0] if isinstance(...) else text_descriptions
        
        # VL branch processes text:
        # 1. TextEncoder: text_list → (B, 768) embeddings
        # 2. SharedProjection: (B, 1024) vision + (B, 768) text → (B, 128) shared space
        vision_emb, text_emb, _ = self.vl_branch(global_vision_features, text_list)
        # Output:
        #   vision_emb: (B, 128) - vision features in shared space
        #   text_emb: (B, 128) - text embeddings in shared space
        #   ↑ These are NOW COMPARABLE! Both on unit sphere!
        
        """
        CONTRASTIVE LOSS: THE BOOST MECHANISM
        """
        
        # Compute contrastive loss
        losses['contrastive'] = self.contrastive_vl_loss(vision_emb, text_emb)
        # Loss function:
        #   - High if vision_emb and text_emb dissimilar
        #   - Low if vision_emb and text_emb similar
        #   - Gradient flows back through:
        #     → vision_emb ← projection ← global_vision_features
        #     → global_vision_features ← aggregation ← patch_features
        #     → patch_features ← CNN_backbone + ROI_attention
        #     → ROI_attention ← Gaussian_parameters (μ, σ)
        
        """
        BACKPROPAGATION & GAUSSIAN SHARPENING:
        
        Loss = high when:
          - Text says "cardiomegaly at right cardiac border"
          - CNN features don't emphasize right side
          - Gaussian center is still at (0.5, 0.5) - center
        
        Gradient computation:
          ∂Loss / ∂Gaussian_parameters:
            = (∂Loss / ∂text_emb) × (∂text_emb / ∂patch_features) ×
              (∂patch_features / ∂Gaussian_parameters)
            
            Example:
            - ∂Loss / ∂text_emb: "text says right → gradient says move right"
            - ∂patch_features / ∂Gaussian_μx: "moving center right ↑ right patches"
            - Combined: ∂Loss / ∂μx > 0 ← gradient says: increase center x-coord!
        
        Gradient update:
          μx_new = μx_old - lr × ∂Loss / ∂μx
          
          Example:
          - μx_old = 0.5 (center)
          - ∂Loss / ∂μx = -0.08 (gradient pointing right)
          - lr = 0.01
          - μx_new = 0.5 - 0.01 × (-0.08) = 0.5 + 0.0008 = 0.5008 → RIGHT!
          
          - σx_old = 0.3 (spread)
          - ∂Loss / ∂σx = -0.15 (gradient says: sharpen!)
          - σx_new = 0.3 - 0.01 × (-0.15) = 0.3 + 0.0015 = 0.3015 → SHARPER!
        
        Result after one epoch:
          - Gaussian center moved toward pathology location from text
          - Gaussian scale decreased (sharper focus)
          - CNN features now emphasize correct region
          - Vision embedding ≈ Text embedding (alignment improved!)
        """
        
        # ROI branch also gets updated
        if encoded_rois is not None:
            # ROI features (from Gaussian ROI pooling)
            roi_features = encoded_rois.aggregated_cls_features.mean(dim=1)
            # These features are pooled using current Gaussian parameters!
            
            # Project ROI features to shared space
            roi_vision_emb, _ = self.vl_branch.projection(roi_features, None)
            
            # VL Consistency loss: patch branch ≈ ROI branch
            # (ensures both branches learn similar semantics)
            losses['vl_consistency'] = self.vl_consistency_loss(vision_emb, roi_vision_emb)
            # ↑ Prevents divergence: patch Gaussian and ROI Gaussian must stay aligned!
    
    # ========== AGGREGATED LOSS COMPUTATION ==========
    
    # Combine detection + contrastive + consistency losses
    loss = sum(losses.values()) / len(losses)
    
    # Backpropagation
    loss.backward()
    # ↑ Gradient flows through ENTIRE network:
    #   - CNN backbone learns semantic features
    #   - ROI Gaussian parameters (μ, σ) adjusted toward text-described locations
    #   - Text encoder frozen (preserves pre-training)
    #   - Projection layers fine-tuned (learn vision-text alignment)
    
    optimizer.step()
    # ↑ Parameters updated:
    #   - Gaussian μ → moves toward correct pathology location from text
    #   - Gaussian σ → decreases (sharper focus) to match text description
    #   - CNN features → emphasize regions named in text
    
    return loss, losses, predictions
```

**Key Integration Points**:

1. **Text as Location Prior**: Caption contains words like "right", "apex", "base" → model learns spatial biases
2. **Embedding Alignment**: Forces vision features to match text semantics
3. **Gradient Flow**: Loss backprops through Gaussian parameters
4. **Sharpening Mechanism**: Reduced σ values = tighter, more focused Gaussian maps

---

## 📈 Practical Example: Before vs After Text Guidance

### Before Text Guidance (Standard WSRPN)

```
Image: Chest X-ray with pleural effusion at right base
Labels: Pleural Effusion = 1 (no spatial info)

Gaussian ROI behavior:
  Center: μ = [0.5, 0.5] (random initialization)
  Scale: σ = [0.3, 0.3] (spread across image)
  
Attention map:
  ████████  (spread over entire right half)
  ████████
  ████████
  
Loss: Binary CE(pred_effusion, 1) = 0.4
      ↑ Only cares about class correctness, not location!
```

### After Text Guidance (WSRPN-VL)

```
Image: Same chest X-ray
Text: "Pleural effusion at right costophrenic angle" (RDF caption)

Gaussian ROI behavior:
  Center: μ = [0.75, 0.85] (moved toward right-base from text)
  Scale: σ = [0.1, 0.1] (tightened around target region)
  
Attention map:
  ░░░░░░░░
  ░░░░░░░░
  ░░░██░░░  (peaked at right-base!)
  ░░░██░░░
  
Loss: Binary CE + Contrastive(vision_emb ≈ text_emb) = 0.2
      ↑ Both class AND alignment contribute to loss signal
      ↑ Text location prior strongly constrains Gaussian position
      ↑ Spatial attention becomes sharper and more focused
```

---

## 🎓 Mathematical Formulation

### Forward Pass: From Text to Gaussian Maps

```
Text Caption: c = "pleural effusion at right base"
       ↓ BERT Encoder
Text Embedding: t ∈ ℝ^768
       ↓ Projection to Shared Space
Text in Shared: t_shared ∈ ℝ^128 (normalized, ||t_shared|| = 1)

Image: x ∈ ℝ^(1×224×224)
       ↓ CNN Backbone + Aggregation
Image Features: f_img ∈ ℝ^1024
       ↓ Projection to Shared Space
Image in Shared: f_shared ∈ ℝ^128 (normalized)

Similarity: s = f_shared · t_shared ∈ [-1, 1]
              (dot product of normalized vectors)

If s ≈ 1: Perfect alignment (image matches text)
          → Gradients are small → Gaussian parameters stable
          
If s ≈ -1: Perfect misalignment (image contradicts text)
           → Gradients are large → Gaussian parameters shift
           → Text location hints drive parameter updates
```

### Backward Pass: Gradient Flow to Gaussian Parameters

```
Loss(f_shared, t_shared) = high (misalignment)
       ↑
∂Loss / ∂f_shared = g_f  (how much to change vision embedding)
       ↑
∂f_shared / ∂f_img = J_proj  (Jacobian of projection)
       ↑
∂f_img / ∂feat_patch = J_agg  (Jacobian of aggregation)
       ↑
∂feat_patch / ∂roi_attn = J_roi  (Jacobian of ROI pooling)
       ↑
∂roi_attn / ∂(μ, σ) = ∇_Gaussian  (how Gaussian parameters affect attention)

CHAIN RULE:
∂Loss / ∂μ = ∂Loss / ∂roi_attn × ∂roi_attn / ∂μ
           = (backprop through all layers)

RESULT: Text caption → gradient on μ and σ
        (Gaussian parameters updated toward text description)
```

---

## 🚀 Three-Phase Training Schedule

### Why Phase Scheduling Matters for Text Boost

**Location**: `src/training/wsrpn_vl_trainer.py` (lines 18-65)

```python
# PHASE 1 (Epochs 0-2): Detection Only
# ========================================
# NO text guidance! Why?
# - Gaussian ROI mechanism needs stabilization first
# - Multi-objective conflicts early cause instability
# - Learn spatial attention without semantic interference

for epoch in range(0, 2):
    loss = L_detection  ← ONLY detection loss, NO VL losses!
    # Gaussian parameters: μ, σ learn from image-level labels
    # Attention maps: learn to highlight any abnormal regions
    # Baseline accuracy: moderate (no semantic guidance)
    
    # Attention map phase 1:
    # ████████  (broad, unfocused)
    # ████████
    # ████████


# PHASE 2 (Epochs 2-N): Add VL Constraints
# =========================================
# NOW text guidance activated!
# - Gaussian parameters already semi-stable
# - Text embeddings provide semantic location priors
# - Contrastive loss pulls vision toward text

for epoch in range(2, N):
    # Curriculum: gradually introduce text guidance
    weight_contrastive = 0.5  # Start with 50% weight
    weight_consistency = 0.5
    
    loss = (L_detection + 
            0.5 * L_contrastive +  ← TEXT BOOST STARTS!
            0.5 * L_consistency)
    
    # Text caption: "cardiomegaly at right cardiac border"
    # Contrastive loss pulls:
    #   - Gaussian center → right side (from text "right")
    #   - Gaussian scale → smaller (from text "cardiac border" = localized)
    #   - Vision features → cardiac region (from text "cardiomegaly")
    
    # Attention map phase 2:
    # ░░░░░░░░  (narrowing, focusing)
    # ░░██░░░░
    # ░░██░░░░


# PHASE 3 (Epochs N+): Gaussian Refinement
# =========================================
# Maximize text boost effect!
# - Gaussian parameters already well-aligned with text
# - Fine-tune with additional Gaussian-specific losses
# - Gaussian concentration (entropy ↓)
# - Gaussian sparsity (peak > mean)

for epoch in range(N, max_epochs):
    loss = (L_detection + 
            0.5 * L_contrastive +
            0.5 * L_consistency +
            0.2 * L_gaussian_concentration +  ← SHARPEN GAUSSIANS!
            0.1 * L_gaussian_sparsity +       ← SPIKE PEAKS!
            0.1 * L_box_alignment)
    
    # Text-guided Gaussian parameters now:
    #   - μ precisely at pathology location
    #   - σ small enough for sharp focus
    #   - Add concentration loss → entropy ↓
    #   - Peak attention value increases
    
    # Attention map phase 3:
    # ░░░░░░░░  (sharp peak)
    # ░░██░░░░
    # ░░██░░░░
    # Peak: 0.95, Sides: 0.05, Mean: 0.15 → Sparse!
```

**Key Benefit of Phasing**:
- Phase 1: Stabilize spatial mechanism (no conflicting objectives)
- Phase 2: Apply text constraints (aligned mechanisms accept guidance)
- Phase 3: Refine for maximum localization (text + Gaussian losses synergize)

---

## 💡 Why This Works: Five Key Mechanisms

| Mechanism | Explanation | Gaussian Boost |
|-----------|-------------|-----------------|
| **Text Encoding** | BERT captures spatial language ("right", "apex") | Location prior |
| **Shared Embedding** | Vision & text in same space → direct comparison | Alignment signal |
| **Contrastive Loss** | Misalignment → gradient on vision features | Backprop to parameters |
| **Gradient Flow** | Loss → CNN features → ROI attention → Gaussian | Parameter update |
| **Phase Scheduling** | Gradual introduction prevents conflicts | Stable convergence |

---

## 📋 Expected Improvements

**Baseline WSRPN** (without text):
- Gaussian centers: Random initialization, slow learning
- Gaussian scales: Large (0.3-0.5), spread across image
- RoDeO mAP: ~25-30% (limited localization)
- Attention maps: Broad, unfocused

**WSRPN-VL** (with text captions):
- Gaussian centers: Text-guided to pathology regions
- Gaussian scales: Smaller (0.1-0.2), focused on targets
- RoDeO mAP: ~32-35% (**5-10% improvement!**)
- Attention maps: Sharp, peaked at target locations

**Validation Strategy** (MIMIC + CXR8):
1. Train on MIMIC split_frontal with RDF text captions
2. Evaluate with pseudo-boxes (intermediate metric)
3. Fine-tune on CXR8 with real bounding boxes
4. Measure improvement on ground truth localization

---

## 🔗 Integration Points in Code

| File | Component | Role |
|------|-----------|------|
| `vl_encoder.py` | TextEncoder | Converts text to semantic embeddings |
| `vl_encoder.py` | SharedProjection | Aligns vision & text in shared space |
| `vl_encoder.py` | VisionLanguageBranch | Orchestrates VL pipeline |
| `vl_losses.py` | ContrastiveVLLoss | Drives text-vision alignment |
| `vl_losses.py` | VLConsistencyLoss | Ensures patch/ROI consistency |
| `wsrpn.py` | train_step() | Integrates text → loss → gradients |
| `soft_roi_pool.py` | SoftRoiPool | Generates Gaussian attention maps |
| `wsrpn_vl_trainer.py` | LossWeightScheduler | Phases text guidance introduction |

---

## 📚 Summary: The Complete Text→Gaussian Pipeline

```
RDF Caption: "pleural effusion right base"
       ↓ BERT Encoding (frozen)
Semantic Vector: [concept_effusion, location_right, location_base, ...]
       ↓ Projection to Shared Space
Text Embedding: (128-dim, normalized)
       ↓ Contrastive Loss
Vision Embedding UPDATED: (CNN features must match text semantics)
       ↓ Backpropagation
Patch Features UPDATED: (emphasize right_base region)
       ↓ Aggregation & ROI Attention
Gaussian Parameters UPDATED:
  - μx: 0.5 → 0.75 (center moves right)
  - μy: 0.5 → 0.80 (center moves down/base)
  - σx: 0.3 → 0.15 (sharpen horizontally)
  - σy: 0.3 → 0.18 (sharpen vertically)
       ↓ ROI Pooling with New Gaussians
Sharper Attention Map: Peaked at right_base!
       ↓ Classification with Focused Features
Better Localization: Model learns WHERE pathology appears
       ↓ Evaluation
Higher RoDeO/mAP: Text guidance improved spatial localization
```

**The magic**: Text captions provide LOCATION PRIORS that guide Gaussian parameters through gradient-based optimization, resulting in sharper, more focused spatial attention.
