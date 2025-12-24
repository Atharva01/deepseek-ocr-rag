# WSRPN-VL: Clean Code Implementation Strategy

**Document Date**: December 24, 2025  
**Status**: Strategy Document (Before Implementation)  
**Objective**: Define clean architecture, code organization, and implementation phases

---

## 1. Current State Analysis

### 1.1 Existing WSRPN Codebase Structure

```
src/
├── model/
│   ├── backbone/              # Feature extractors (DenseNet, ViT, DINO)
│   ├── object_detectors/
│   │   └── wsrpn.py          # Main WSRPN model
│   ├── losses.py             # Existing loss functions (BCE, SupConPerClass)
│   ├── model_components.py   # Utility components (attention, pooling)
│   ├── model_interface.py    # Abstract base classes
│   ├── model_loader.py       # Model instantiation
│   ├── soft_roi_pool.py      # Gaussian soft ROI pooling
│   └── positional_embedding.py
│
├── data/
│   ├── __init__.py
│   ├── cxr8.py              # CXR8 dataset
│   ├── vindr.py             # VinDR dataset
│   └── datasets.py          # Generic dataset loaders
│
├── conf/                     # Hydra configuration
│   └── (config yamls)
│
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation entry point
└── utils/                   # Utility functions
```

### 1.2 Current Dependencies
- PyTorch (model, training)
- Transformers (backbone loaders)
- Hydra (configuration management)
- OpenCV/PIL (image processing)
- NumPy/SciPy (numerical operations)

### 1.3 Key Patterns Observed
```
✓ Model registry pattern (decorator-based instantiation)
✓ Configuration via Hydra dataclasses
✓ Modular loss functions (easy to add new ones)
✓ Abstract interfaces (ObjectDetectorModelInterface)
✗ Limited vision-language integration
✗ No explicit phase-based training scheduler
✗ BERT/text encoder not yet integrated
```

---

## 2. Clean Architecture Design

### 2.1 Design Principles

```
PRINCIPLE 1: Single Responsibility
├─ Each module has one clear purpose
├─ Vision model ≠ Text model ≠ Training loop
└─ Easy to test and maintain

PRINCIPLE 2: Dependency Injection
├─ Components receive dependencies as arguments
├─ No global state or hard-coded imports
└─ Testable without side effects

PRINCIPLE 3: Separation of Concerns
├─ Model architecture ≠ Loss computation ≠ Data loading
├─ Phase scheduling ≠ Training mechanics
└─ Inference ≠ Training

PRINCIPLE 4: Interface Contracts
├─ Define clear input/output specifications
├─ Type hints everywhere
├─ Exceptions for contract violations

PRINCIPLE 5: Gradual Adoption
├─ VL components orthogonal to existing WSRPN
├─ Minimal changes to core model
├─ Feature flags for optional components
```

### 2.2 Proposed New Component Structure

**Legend**: 
- 🟢 GREEN = Existing (unchanged or minimal changes)
- 🔴 RED = New (to be created)
- 🟡 YELLOW = Modified (enhanced with feature flags)

```
src/
├── model/
│   ├── backbone/                    🟢 EXISTING
│   │   └── (DenseNet, ViT, DINO loaders - unchanged)
│   │
│   ├── vision_language/             🔴 NEW DIRECTORY ← VL INTEGRATION
│   │   ├── __init__.py
│   │   ├── text_encoder.py          # BERT wrapper (frozen)
│   │   ├── vision_projector.py      # Vision → shared space (global, patch, ROI)
│   │   ├── text_projector.py        # Text → shared space
│   │   ├── vl_branch.py             # Orchestrator (combines text + vision)
│   │   └── vl_utils.py              # Helper functions (normalization, etc)
│   │
│   ├── training/                    🔴 NEW DIRECTORY ← PHASE MANAGEMENT
│   │   ├── __init__.py
│   │   ├── loss_scheduler.py        # Phase-based loss weight scheduling
│   │   ├── metrics_tracker.py       # Multi-phase metrics aggregation
│   │   └── training_utils.py        # Helper functions (logging, etc)
│   │
│   ├── gaussian_losses/             🔴 NEW DIRECTORY ← GAUSSIAN OPTIMIZATION
│   │   ├── __init__.py
│   │   ├── concentration.py         # Gaussian concentration loss
│   │   ├── sparsity.py              # Gaussian sparsity loss
│   │   ├── alignment.py             # Box-Gaussian alignment loss
│   │   └── suppression.py           # Negative region suppression loss
│   │
│   ├── object_detectors/            🟡 EXISTING (minor changes)
│   │   ├── wsrpn.py                 # EXISTING: minimal modifications
│   │   │                             # - Add use_vl_branch flag
│   │   │                             # - Add use_gaussian_losses flag
│   │   │                             # - Optional VL initialization
│   │   │                             # - Backward compatible (no breaking changes)
│   │   └── wsrpn_vl.py              # 🔴 NEW: subclass WSRPN with full VL support
│   │
│   ├── losses.py                    🟡 MODIFIED
│   │   # - Add ContrastiveVLLoss
│   │   # - Add other VL-specific losses
│   │   # - Existing BCE, SupConPerClass unchanged
│   │
│   ├── model_components.py          🟢 EXISTING (unchanged)
│   ├── model_interface.py           🟢 EXISTING (unchanged)
│   ├── model_loader.py              🟢 EXISTING (unchanged)
│   ├── soft_roi_pool.py             🟢 EXISTING (unchanged)
│   ├── positional_embedding.py      🟢 EXISTING (unchanged)
│   └── backbone/                    🟢 EXISTING (unchanged)
│
├── data/
│   ├── __init__.py                  🟢 EXISTING (unchanged)
│   ├── cxr8.py                      🟢 EXISTING (unchanged)
│   ├── vindr.py                     🟢 EXISTING (unchanged)
│   └── datasets.py                  🟡 MODIFIED (optional)
│                                    # - Add text caption support (optional)
│                                    # - Handle tokenization in collate_fn
│
├── training/                        🔴 NEW DIRECTORY ← TRAINING ORCHESTRATION
│   ├── __init__.py
│   ├── train_pipeline.py            # Main training loop (enhanced)
│   ├── validators.py                # Input/output validation
│   └── phase_manager.py             # Phase lifecycle management
│
├── conf/                            🟢 EXISTING (add new yaml files)
│   ├── model/
│   │   ├── wsrpn.yaml               🟢 EXISTING
│   │   ├── wsrpn_vl.yaml            🔴 NEW: VL-specific config
│   │   └── (other configs - unchanged)
│   │
│   ├── training/                    🔴 NEW DIRECTORY
│   │   ├── phases.yaml              # Phase definitions (warmup, gaussian, vl)
│   │   ├── loss_scheduler.yaml      # Loss weight schedules
│   │   └── gaussian.yaml            # Gaussian loss config
│   │
│   └── (other configs - unchanged)
│
├── train.py                         🟢 EXISTING (unchanged for now)
│                                    # Can add train_wsrpn_vl.py later
│
├── evaluate.py                      🟢 EXISTING (unchanged)
│
├── utils/                           🟢 EXISTING
│   └── (unchanged)
│
├── metrics/                         🟢 EXISTING (unchanged)
│
├── unittests/                       🟢 EXISTING (unchanged)
│
└── plot/                            🟢 EXISTING (unchanged)
```

### 2.2.1 Structure Comparison: Old vs New

```
BEFORE (Current WSRPN):
─────────────────────────────────────
src/
├── model/
│   ├── object_detectors/wsrpn.py   ← Single detection model
│   ├── losses.py                   ← BCE, SupConPerClass only
│   └── (supporting modules)
│
├── train.py                        ← Single training script
│
└── data/
    └── datasets.py                 ← Image + Labels only

Issues:
  ✗ No VL integration
  ✗ No phase-based scheduling
  ✗ No Gaussian optimization losses
  ✗ No structured phase management
  ✗ Hard to add multi-modal learning


AFTER (WSRPN-VL with Clean Architecture):
──────────────────────────────────────────
src/
├── model/
│   ├── vision_language/            ← NEW: VL branch (orthogonal to WSRPN)
│   │   ├── text_encoder.py        # BERT integration
│   │   ├── vision_projector.py    # Shared embedding space
│   │   ├── text_projector.py      # Shared embedding space
│   │   └── vl_branch.py           # Unified VL module
│   │
│   ├── training/                   ← NEW: Training infrastructure
│   │   ├── loss_scheduler.py      # Phase-based scheduling
│   │   └── metrics_tracker.py     # Multi-phase metrics
│   │
│   ├── gaussian_losses/            ← NEW: Gaussian optimization
│   │   ├── concentration.py       # Sharp Gaussians
│   │   ├── sparsity.py           # Sparse attention
│   │   ├── alignment.py          # Box consistency
│   │   └── suppression.py        # False positive reduction
│   │
│   ├── object_detectors/
│   │   ├── wsrpn.py              ← MINIMAL CHANGES (backward compatible)
│   │   └── wsrpn_vl.py           ← NEW: Full VL support
│   │
│   ├── losses.py                 ← ADD: ContrastiveVLLoss
│   └── (existing modules)
│
├── training/                        ← NEW: Training orchestration
│   ├── train_pipeline.py          # Enhanced training loop
│   └── validators.py              # Input validation
│
├── conf/
│   ├── model/wsrpn_vl.yaml       ← NEW: VL configuration
│   └── training/                  ← NEW: Phase config
│
└── data/
    └── datasets.py               ← OPTIONAL: text support


Benefits:
  ✓ Clean separation: VL orthogonal to detection
  ✓ Modular: Each loss independently testable
  ✓ Backward compatible: Original WSRPN still works
  ✓ Extensible: Easy to add new phases/losses
  ✓ Type-safe: Full type hints everywhere
  ✓ Reproducible: Centralized scheduling


KEY CHANGES SUMMARY:
──────────────────
1. NEW directories: vision_language/, training/, gaussian_losses/
2. NEW classes: ~12 classes (text encoder, projectors, losses, scheduler)
3. NEW files: ~12 new files (no deletions)
4. MODIFIED files: 3 files (wsrpn.py, losses.py, conf yamls)
5. UNCHANGED files: All existing model/data/utils files

Impact Analysis:
  ├─ Lines of code added: ~2500 new lines
  ├─ Lines of code modified: ~100 lines in existing files
  ├─ Lines of code deleted: 0 (backward compatible)
  ├─ Breaking changes: 0 (feature flags used)
  └─ Existing functionality: 100% preserved
```

### 2.3 Module Dependency Graph

```
         [Training Pipeline]
              ↓ uses
    ┌─────────┴──────────┐
    ↓                    ↓
[Phase Manager]    [Loss Scheduler]
    ↓                    ↓
    └─────────┬──────────┘
              ↓ coordinates
         [WSRPN-VL Model]
              ↓
    ┌─────────┼──────────────────┐
    ↓         ↓                  ↓
 [Backbone][VL Branch]    [Gaussian Losses]
   (Frozen)   ├─ Vision Encoder     ├─ Concentration
              ├─ Text Encoder      ├─ Sparsity
              ├─ Projectors        └─ Alignment
              └─ Contrastive Loss

Data flows: Image + Text → Model → Losses → Gradients → Optimizer
```

---

## 3. Clean Code Implementation Strategy

### 3.1 File Creation Plan (Minimal Changes)

**PHASE A: Core VL Components (No WSRPN modification)**

```
FILE 1: src/model/vision_language/text_encoder.py
Purpose: BERT wrapper with frozen parameters
Content:
  - class TextEncoderBERT
  - Input: tokenized text (input_ids, attention_mask)
  - Output: embeddings (B, d_hidden)
  - Signature: forward(input_ids, attention_mask) → Tensor
  - Type hints: All inputs/outputs typed
  - Config: Model name, hidden_size, freeze flag

FILE 2: src/model/vision_language/vision_projector.py
Purpose: Project vision features to shared space
Content:
  - class VisionProjector (for whole image)
  - class PatchVisionProjector (for patch features)
  - class ROIVisionProjector (for ROI features)
  - Input: feature tensor (B, d) or (B, K, d)
  - Output: normalized embeddings (B, d_shared) or (B, K, d_shared)
  - Configuration: input_dim, output_dim, hidden_dims

FILE 3: src/model/vision_language/text_projector.py
Purpose: Project text embeddings to shared space
Content:
  - class TextProjector
  - Input: text embeddings (B, 768)
  - Output: normalized embeddings (B, d_shared)
  - Configuration: input_dim, output_dim, hidden_dims

FILE 4: src/model/vision_language/vl_branch.py
Purpose: Unified VL branch combining text + vision
Content:
  - class VisionLanguageBranch
  - Input: image features, image labels, text + tokenizer
  - Forward: Orchestrate encoders + projectors
  - Output: vision_embeddings, text_embeddings (normalized, on shared space)
  - Initialization: Load pre-trained BERT, initialize projectors

FILE 5: src/model/training/loss_scheduler.py
Purpose: Phase-based loss weight scheduling
Content:
  - class LossWeightScheduler
  - Methods:
    * get_phase(step: int) → (phase_name: str, step_in_phase: int)
    * get_weights(step: int) → Dict[loss_name, weight]
    * get_phase_info(step: int) → DetailedPhaseInfo
  - Configuration: Phase definitions (name, step_range, weight_profiles)
  - Phases: Warmup (0-N), Gaussian (N-M), VL (M-T)
  - State tracking: Current phase, transitions, upcoming changes

FILE 6: src/model/gaussian_losses/concentration.py
Purpose: Gaussian concentration loss (peak sharpness)
Content:
  - class GaussianConcentrationLoss
  - Input: Gaussian parameters (center, scale), target locations
  - Output: scalar loss value
  - Metric: Entropy of Gaussian map
  - Interpretability: Lower → sharper attention

FILE 7: src/model/gaussian_losses/sparsity.py
Purpose: Gaussian sparsity loss (region-focused)
Content:
  - class GaussianSparsityLoss
  - Input: Gaussian parameters, image features
  - Output: scalar loss value
  - Metric: Attention mass outside 3-sigma region
  - Interpretability: Lower → more sparse

FILE 8: src/model/gaussian_losses/alignment.py
Purpose: Box-Gaussian parameter alignment
Content:
  - class BoxGaussianAlignmentLoss
  - Input: Predicted boxes, Gaussian parameters
  - Output: scalar loss value
  - Ensures: Consistency between representations

FILE 9: src/model/gaussian_losses/suppression.py
Purpose: Suppress false positives in normal regions
Content:
  - class NegativeRegionSuppressionLoss
  - Input: Predictions in known normal regions, soft labels
  - Output: scalar loss value
  - Use case: Reduce activations in "definitely normal" areas
```

**PHASE B: Training & Scheduling (Minimal WSRPN changes)**

```
FILE 10: src/training/phase_manager.py
Purpose: Lifecycle management for multi-phase training
Content:
  - class PhaseManager
  - State machine: Warmup → Gaussian → VL
  - Methods:
    * start_phase(phase_name)
    * is_phase_transition(step)
    * get_current_phase()
  - Logging: Print phase transitions to console + log file

FILE 11: src/training/train_pipeline.py
Purpose: Enhanced training loop with phase support
Content:
  - function: train_wsrpn_vl(config, model, dataloaders, device)
  - Loop structure:
    * FOR each step:
      - Get loss weights from scheduler
      - Forward pass (image + text)
      - Compute all losses
      - Weighted sum
      - Backward pass
      - Optimizer step
  - Logging: Separate tracking per loss component
  - Checkpointing: Save best + latest

FILE 12: src/training/validators.py
Purpose: Input validation and contract enforcement
Content:
  - def validate_batch(batch) → bool
  - def validate_config(config) → bool
  - def validate_model_output(output) → bool
  - Error messages: Clear, actionable feedback
```

**PHASE C: WSRPN Augmentation (Minimal modifications)**

```
MODIFIED FILE: src/model/object_detectors/wsrpn.py
Changes: Backward compatible, feature-flag based
  ├─ Add use_vl_branch: bool parameter
  ├─ Add use_gaussian_losses: bool parameter
  ├─ If use_vl_branch=true:
  │  ├─ Initialize VisionLanguageBranch
  │  ├─ Extend forward() to handle text input
  │  └─ Return VL embeddings alongside predictions
  ├─ If use_gaussian_losses=true:
  │  ├─ Initialize Gaussian loss functions
  │  └─ Include in loss computation
  └─ Else: Behave like original WSRPN (no breaking changes)

STRATEGY: Inheritance with optional mixins
  class WSRPNBase (original)
  class WSRPN_VL(WSRPNBase):  ← VL components added here
```

### 3.2 Clean Code Principles Applied

**PRINCIPLE: Type Hints Everywhere**

```python
# ✓ GOOD: Clear contracts
def forward(
    self,
    images: Tensor,                      # (B, 1, 224, 224)
    labels: Tensor,                      # (B, 14)
    text_tokens: Optional[Dict] = None,  # Optional for backward compat
    step: int = 0
) -> Dict[str, Any]:
    """
    Forward pass with optional VL.
    
    Args:
        images: Batch of X-ray images
        labels: Ground truth binary labels
        text_tokens: Tokenized text descriptions (optional)
        step: Current training step (for phase scheduling)
    
    Returns:
        {
            'predictions': Tensor,        # Predictions
            'losses': Dict[str, Tensor],  # Loss components
            'vl_embeddings': Optional[Dict],  # If VL enabled
            'metrics': Dict[str, float]   # Evaluation metrics
        }
    """
    ...

# ✗ BAD: Ambiguous
def forward(self, x, y, z=None, s=0):
    ...
```

**PRINCIPLE: Single Responsibility**

```python
# ✓ GOOD: Each class does one thing
class VisionProjector(nn.Module):
    """Project vision features to shared embedding space."""
    def forward(self, features: Tensor) -> Tensor:
        return normalized_embeddings

class TextProjector(nn.Module):
    """Project text embeddings to shared embedding space."""
    def forward(self, embeddings: Tensor) -> Tensor:
        return normalized_embeddings

# ✗ BAD: Mixed responsibilities
class VLModule(nn.Module):
    def forward(self, images, text):
        # Projects vision, text, computes loss, updates metrics, logs...
        # Too much!
```

**PRINCIPLE: Dependency Injection**

```python
# ✓ GOOD: Dependencies passed in
class LossScheduler:
    def __init__(self, phase_config: PhaseConfig):
        self.phases = phase_config.phases
    
    def get_weights(self, step: int) -> Dict[str, float]:
        ...

scheduler = LossScheduler(config.training.phases)
weights = scheduler.get_weights(current_step)

# ✗ BAD: Hard-coded dependencies
class LossScheduler:
    def get_weights(self, step: int) -> Dict[str, float]:
        # What phases? Hard-coded to specific training strategy
        if step < 1000:
            return {'detection': 1.0, 'vl': 0.0}
        ...
```

**PRINCIPLE: Clear Contracts (Interfaces)**

```python
# ✓ GOOD: Protocol-based (Python 3.8+)
from typing import Protocol

class TextEncoder(Protocol):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """Returns (B, hidden_size) embeddings."""
        ...

# Any class implementing this protocol works
class BERTEncoder:
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.bert(input_ids, attention_mask)[1]  # [CLS] pooling

# ✗ BAD: Loose coupling
def compute_vl_loss(text_model):
    # What does text_model have? No idea!
    result = text_model.something()
```

**PRINCIPLE: Configuration Objects**

```python
# ✓ GOOD: Dataclass-based configuration
from dataclasses import dataclass

@dataclass
class VLConfig:
    use_vl_branch: bool = True
    text_model: str = "bert-base-uncased"
    shared_dim: int = 128
    freeze_text_encoder: bool = True
    temperature: float = 0.15

@dataclass
class GaussianConfig:
    use_concentration_loss: bool = True
    use_sparsity_loss: bool = True
    concentration_weight: float = 0.3
    sparsity_weight: float = 0.3

# Usage: config.vl.use_vl_branch, config.gaussian.concentration_weight
# Type-safe, IDE-friendly, validation-friendly

# ✗ BAD: String-based configuration
config = {
    'vl_branch': True,
    'text_model': 'bert',
    'shared_dim': '128',  # Oops, string instead of int
    'freeze_encoder': 'true'  # Hard to parse
}
```

---

## 4. Implementation Phases

### 4.1 Phase ZERO: Setup & Preparation (1-2 days)

**Objective**: Foundation for clean implementation

Tasks:
- [ ] Create VL-specific directories (vision_language/, training/, gaussian_losses/)
- [ ] Setup type checking (mypy configuration)
- [ ] Create test infrastructure
- [ ] Document module interfaces
- [ ] Setup logging (structured logging)

Success Criteria:
- ✓ Directory structure matches design
- ✓ Type checking configured
- ✓ Test runners operational
- ✓ Logging structured

### 4.2 Phase ONE: Core VL Components (2-3 days)

**Objective**: Implement VL branch independently

Tasks:
- [ ] Implement TextEncoderBERT (src/model/vision_language/text_encoder.py)
- [ ] Implement VisionProjector variants (src/model/vision_language/vision_projector.py)
- [ ] Implement TextProjector (src/model/vision_language/text_projector.py)
- [ ] Create VisionLanguageBranch orchestrator
- [ ] Write unit tests for each component
- [ ] Integration test: VL branch standalone

Success Criteria:
- ✓ All unit tests pass
- ✓ Forward pass works with sample inputs
- ✓ Output dimensions correct
- ✓ Type hints complete
- ✓ Docstrings comprehensive

### 4.3 Phase TWO: Loss Scheduling & Gaussian Losses (2-3 days)

**Objective**: Implement training infrastructure

Tasks:
- [ ] Implement LossWeightScheduler
- [ ] Implement PhaseManager
- [ ] Implement GaussianConcentrationLoss
- [ ] Implement GaussianSparsityLoss
- [ ] Implement BoxGaussianAlignmentLoss
- [ ] Write tests for scheduler (phase transitions)
- [ ] Write tests for loss functions

Success Criteria:
- ✓ Phase transitions occur at correct steps
- ✓ Loss weights match specifications
- ✓ All Gaussian losses return scalar tensors
- ✓ Gradient flow verified

### 4.4 Phase THREE: WSRPN Integration (2-3 days)

**Objective**: Connect VL to existing WSRPN

Tasks:
- [ ] Create WSRPN_VL variant (inheritance-based)
- [ ] Add feature flags (use_vl_branch, use_gaussian_losses)
- [ ] Modify forward() for optional text input
- [ ] Extend loss computation
- [ ] Backward compatibility tests (original WSRPN still works)
- [ ] Integration tests (full model with VL)

Success Criteria:
- ✓ Original WSRPN unchanged (backward compatible)
- ✓ WSRPN_VL adds VL without side effects
- ✓ Feature flags work correctly
- ✓ All tests pass

### 4.5 Phase FOUR: Training Pipeline (2-3 days)

**Objective**: Implement train_wsrpn_vl() training loop

Tasks:
- [ ] Implement train_pipeline.py
- [ ] Add phase detection + logging
- [ ] Add metrics tracking (per-phase)
- [ ] Add checkpointing (best + latest)
- [ ] Add validation loop
- [ ] Implement validators.py
- [ ] Write integration tests

Success Criteria:
- ✓ Training loop runs for 10 steps without errors
- ✓ Metrics tracked correctly
- ✓ Checkpoints saved with correct format
- ✓ Phase transitions logged clearly

---

## 5. Code Quality Standards

### 5.1 Type Hints

**MANDATORY**: All functions, methods, class attributes

```python
# ✓ Required style
def compute_loss(
    predictions: Tensor,           # (B, C)
    targets: Tensor,               # (B, C)
    weights: Optional[Tensor] = None  # (B,)
) -> Tensor:  # scalar
    ...

class VLBranch(nn.Module):
    vision_projector: nn.Module
    text_encoder: TextEncoder
    
    def forward(
        self,
        images: Tensor,
        text_tokens: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:  # (vision_emb, text_emb)
        ...
```

### 5.2 Docstrings

**MANDATORY**: Google-style for all public methods

```python
def compute_contrastive_loss(
    vision_embeddings: Tensor,
    text_embeddings: Tensor,
    temperature: float = 0.15
) -> Tensor:
    """Compute NT-Xent contrastive loss between vision and text.
    
    Aligns vision features with text embeddings using normalized
    temperature-scaled cross-entropy loss (SimCLR style).
    
    Args:
        vision_embeddings: Vision feature embeddings (B, D). Should be
            L2-normalized.
        text_embeddings: Text feature embeddings (B, D). Should be
            L2-normalized.
        temperature: Scaling temperature for softmax. Default 0.15.
            Lower → sharper contrast.
    
    Returns:
        Scalar loss value (averaged over batch).
    
    Raises:
        ValueError: If embeddings not normalized or shapes don't match.
    
    Example:
        >>> v_emb = torch.randn(32, 128)
        >>> t_emb = torch.randn(32, 128)
        >>> v_emb = F.normalize(v_emb, p=2, dim=1)
        >>> t_emb = F.normalize(t_emb, p=2, dim=1)
        >>> loss = compute_contrastive_loss(v_emb, t_emb)
    """
    ...
```

### 5.3 Testing

**MANDATORY**: Unit test for every class/function

```
tests/
├── test_text_encoder.py
├── test_vision_projector.py
├── test_vl_branch.py
├── test_loss_scheduler.py
├── test_gaussian_losses.py
├── test_wsrpn_vl.py
└── test_train_pipeline.py

Each test file:
  ├── Fixture for common test data
  ├── Test: initialization with valid config
  ├─ Test: forward pass with shape validation
  ├─ Test: backward pass (gradient flow)
  ├─ Test: edge cases (empty batch, None input)
  └─ Test: error cases (invalid input)
```

### 5.4 Logging

**MANDATORY**: Structured logging for all key operations

```python
import logging

logger = logging.getLogger(__name__)

# ✓ GOOD: Structured, informative
logger.info(
    "Phase transition",
    extra={
        'from_phase': 'warmup',
        'to_phase': 'gaussian',
        'step': 1000,
        'remaining_steps': 3000
    }
)

logger.warning(
    "Loss diverging",
    extra={
        'loss_value': 2.5,
        'threshold': 1.5,
        'step': 500,
        'recommendation': 'Reduce learning rate'
    }
)

# ✗ BAD: Unstructured, hard to parse
print("Switching to gaussian phase at step 1000")
print("WARNING: loss is high!")
```

### 5.5 Error Handling

**MANDATORY**: Validate inputs, raise clear exceptions

```python
# ✓ GOOD: Clear validation + actionable errors
def forward(self, images: Tensor, text_tokens: Dict[str, Tensor]):
    if images.shape[1] != 1:
        raise ValueError(
            f"Expected grayscale images (B, 1, H, W), "
            f"got shape {images.shape}. "
            f"Hint: Convert to single channel with torch.mean()"
        )
    
    if images.shape[-1] != 224 or images.shape[-2] != 224:
        raise ValueError(
            f"Expected 224x224 images, got {images.shape[-2:]}. "
            f"Hint: Resize with transforms.Resize((224, 224))"
        )
    
    if 'input_ids' not in text_tokens:
        raise ValueError(
            f"text_tokens missing 'input_ids'. "
            f"Got keys: {list(text_tokens.keys())}. "
            f"Hint: Use tokenizer() to prepare input"
        )
    
    # ... computation ...

# ✗ BAD: Silent failures
def forward(self, images, text_tokens):
    if images.ndim != 4:
        images = images.unsqueeze(0)  # Implicit assumption!
    
    text_emb = text_tokens.get('input_ids', None)
    # What if 'input_ids' is missing? Silent None?
```

---

## 6. Key Design Decisions

### Decision 1: VL Components are Orthogonal

**Choice**: VL branch separate from WSRPN core

**Rationale**:
- ✓ WSRPN can train without VL (backward compatible)
- ✓ VL can be tested independently
- ✓ Easy to toggle on/off with feature flags
- ✓ Minimal changes to existing code

**Alternative Rejected**: Tightly integrate VL into WSRPN
- ✗ Breaking changes to existing code
- ✗ Harder to debug issues
- ✗ Can't disable VL if needed

### Decision 2: Phase Scheduling is Centralized

**Choice**: LossWeightScheduler controls all phase logic

**Rationale**:
- ✓ Single source of truth
- ✓ Easy to experiment with phase boundaries
- ✓ Reproducible phase transitions
- ✓ Logging/debugging simpler

**Alternative Rejected**: Phase logic distributed in training loop
- ✗ Hard-coded if/else statements
- ✗ Difficult to modify phase schedule
- ✗ Validation scattered

### Decision 3: Gaussian Losses are Separate Classes

**Choice**: One loss class per Gaussian optimization component

**Rationale**:
- ✓ Each loss independently testable
- ✓ Easy to enable/disable specific losses
- ✓ Clear responsibility
- ✓ Reusable in other models

**Alternative Rejected**: Monolithic GaussianLoss class
- ✗ Hard to debug individual components
- ✗ Can't enable sparsity without concentration
- ✗ Tightly coupled

### Decision 4: Inheritance for WSRPN_VL

**Choice**: Create WSRPN_VL(WSRPN) subclass

**Rationale**:
- ✓ Original WSRPN unchanged
- ✓ Code reuse (inherits all base functionality)
- ✓ Easy to compare baseline vs VL
- ✓ Polymorphic (both treated as detectors)

**Alternative Rejected**: Monkeypatch existing WSRPN
- ✗ Pollutes WSRPN namespace
- ✗ Hard to understand what's modified
- ✗ Fragile to base class changes

---

## 7. Validation & Testing Strategy

### 7.1 Unit Testing

```
Each module tested independently:

✓ TextEncoderBERT
  ├─ Load pre-trained BERT
  ├─ Forward pass shape check
  ├─ Gradient flow (if unfrozen)
  ├─ Token sequence handling
  └─ Batch vs single sample

✓ VisionProjector
  ├─ Shape transformation
  ├─ Normalization (L2)
  ├─ Gradient flow
  └─ Edge case: batch_size=1

✓ LossScheduler
  ├─ Phase transitions at correct steps
  ├─ Weight values match spec
  ├─ Boundary conditions (step=0, step=max)
  └─ Error handling (invalid step)

✓ GaussianLosses
  ├─ Loss returns scalar
  ├─ Loss ≥ 0
  ├─ Gradient flow
  ├─ Edge cases (all-zero input, saturated input)
  └─ Dimensionality checks
```

### 7.2 Integration Testing

```
Component combinations:

✓ VL Branch End-to-End
  ├─ Image + text → embeddings
  ├─ Embedding shapes correct
  ├─ Embeddings L2-normalized
  ├─ Batch processing
  └─ Backward pass works

✓ WSRPN + VL
  ├─ Original WSRPN works (use_vl_branch=False)
  ├─ WSRPN_VL works (use_vl_branch=True)
  ├─ Predictions unchanged (deterministic)
  ├─ VL outputs correct format
  └─ Loss computation includes VL

✓ Training Loop
  ├─ 10-step training run without errors
  ├─ Loss decreasing (or at least not NaN)
  ├─ Checkpoints saved correctly
  ├─ Phase transitions logged
  └─ Metrics tracked per phase
```

### 7.3 Smoke Tests (Quick Validation)

```bash
# 1. Import all modules
python -c "from src.model.vision_language import *; print('✓')"

# 2. Create model instance
python -c "
from src.model.object_detectors.wsrpn_vl import WSRPN_VL
model = WSRPN_VL(config)
print(f'✓ Model created: {model}')"

# 3. Forward pass
python -c "
import torch
from src.model.object_detectors.wsrpn_vl import WSRPN_VL
model = WSRPN_VL(config)
x = torch.randn(2, 1, 224, 224)
out = model(x)
print(f'✓ Forward pass output keys: {out.keys()}')"

# 4. Loss computation
python -c "
import torch
from src.model.losses import ContrastiveVLLoss
loss_fn = ContrastiveVLLoss()
v_emb = torch.randn(4, 128)
t_emb = torch.randn(4, 128)
loss = loss_fn(v_emb, t_emb)
print(f'✓ Loss: {loss.item():.4f}')"
```

---

## 8. Checklist for Implementation

### Pre-Implementation
- [ ] Review current WSRPN code (wsrpn.py, train.py)
- [ ] Understand model registry pattern
- [ ] Understand Hydra configuration
- [ ] Setup type checking (mypy)
- [ ] Create test infrastructure
- [ ] Document this strategy with team

### Phase ZERO: Setup
- [ ] Create directory structure
- [ ] Create __init__.py files
- [ ] Create test directories
- [ ] Setup logging configuration
- [ ] Document module interfaces

### Phase ONE: VL Components
- [ ] Implement TextEncoderBERT
  - [ ] Load BERT model
  - [ ] Handle tokenization
  - [ ] Freeze parameters
  - [ ] Test with sample inputs
  
- [ ] Implement VisionProjector
  - [ ] Project global features
  - [ ] L2 normalization
  - [ ] Test shapes
  
- [ ] Implement TextProjector
  - [ ] Project text embeddings
  - [ ] L2 normalization
  - [ ] Test shapes
  
- [ ] Create VisionLanguageBranch
  - [ ] Orchestrate encoders
  - [ ] Test end-to-end
  
- [ ] Write unit tests for all components

### Phase TWO: Training Infrastructure
- [ ] Implement LossWeightScheduler
  - [ ] Define phase transitions
  - [ ] Test phase boundaries
  
- [ ] Implement Gaussian losses
  - [ ] Concentration loss
  - [ ] Sparsity loss
  - [ ] Alignment loss
  - [ ] Test each independently
  
- [ ] Implement PhaseManager
  - [ ] State machine logic
  - [ ] Transition logging

### Phase THREE: WSRPN Integration
- [ ] Create WSRPN_VL variant
  - [ ] Inherit from WSRPN
  - [ ] Add VL initialization
  - [ ] Extend forward()
  - [ ] Extend loss computation
  
- [ ] Backward compatibility testing
  - [ ] Original WSRPN unchanged
  - [ ] Feature flags work
  
- [ ] Integration tests

### Phase FOUR: Training Pipeline
- [ ] Implement train_wsrpn_vl()
  - [ ] Phase detection
  - [ ] Loss computation (weighted)
  - [ ] Gradient updates
  - [ ] Metrics tracking
  
- [ ] Add validation loop
  - [ ] Validation metrics
  - [ ] Best checkpoint saving
  
- [ ] Input validators
  - [ ] Batch validation
  - [ ] Config validation
  
- [ ] Full integration test

### Post-Implementation
- [ ] Run full test suite
- [ ] Type checking (mypy)
- [ ] Code review
- [ ] Documentation update
- [ ] Performance profiling
- [ ] Smoke tests on sample data

---

## 9. Summary

### Clean Architecture Benefits

```
✓ Maintainability: Clear separation of concerns
✓ Testability: Each component independently testable
✓ Reusability: Gaussian losses work with other detectors
✓ Extensibility: Easy to add new phases/losses
✓ Debuggability: Type hints + logging make bugs obvious
✓ Reproducibility: Deterministic, well-documented
✓ Collaboration: Clear interfaces, minimal merge conflicts
```

### Key Principles

```
1. Orthogonal Components: VL ⊥ WSRPN core
2. Centralized Scheduling: One source of truth for phases
3. Dependency Injection: No global state
4. Type Safety: All functions type-hinted
5. Comprehensive Testing: Every class has tests
6. Clear Contracts: Interfaces define expectations
7. Structured Logging: Debug-friendly diagnostics
```

### Timeline Estimate

```
Phase ZERO: Setup                   1-2 days
Phase ONE: VL Components            2-3 days
Phase TWO: Loss & Scheduling        2-3 days
Phase THREE: WSRPN Integration      2-3 days
Phase FOUR: Training Pipeline       2-3 days
─────────────────────────────────────────────
Total:                              11-17 days

With expert developer: 9-12 days
With debugging/iteration: 15-20 days
```

---

**Status**: Ready for Implementation  
**Next Step**: Begin Phase ZERO (Setup & Preparation)  
**Validation**: All design decisions documented with rationale
