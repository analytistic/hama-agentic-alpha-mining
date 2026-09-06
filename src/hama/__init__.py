"""HAMA: Harness-Augmented Agentic Alpha Mining."""

from .agent import Agent
from .attribution import (
    ComponentAttribution,
    ComponentAttributor,
    InvocationKind,
    InvocationRef,
    render_prefill,
)
from .environment import AlphaMiningEnvironment, FactorEvaluator
from .evaluation import PolicyEvaluation, evaluate_factor_pool_splits, evaluate_policy
from .advantage import (
    AdvantageBatch,
    AdvantageRecord,
    EMAReturnBaseline,
    estimate_ema_advantages,
)
from .harness import Harness, MemoryBank, SkillLibrary
from .model import Model
from .optimization import (
    AggregatedSemanticGradient,
    AtomicGradientProposal,
    HarnessOptimizationResult,
    ConflictAwareSemanticGradientEngine,
    HarnessComponent,
    HarnessEvolutionOptimizer,
    ParameterEdit,
    ParameterRef,
    optimize_harness_with_attribution,
)
from .prefill import PrefillScore, PrefillScorer, QwenPrefillScorer
from .qlib_evaluator import (
    EqualWeightZScoreCombiner,
    FactorCombiner,
    FactorEvaluationError,
    QlibEvaluatorConfig,
    QlibFactorEvaluator,
)
from .rollout import rollout
from .training import (
    HamaTrainer,
    TrainingResult,
    TrainingRound,
    TrainingConfig,
)
from .types import (
    AgentRun,
    Factor,
    FactorEdit,
    FactorEditOperation,
    HarnessTrace,
    MarketState,
    MemoryEntry,
    Message,
    Skill,
    ToolCall,
    ToolRunResult,
    Transition,
)

__all__ = [
    "AdvantageBatch",
    "AdvantageRecord",
    "EMAReturnBaseline",
    "Agent",
    "AgentRun",
    "AggregatedSemanticGradient",
    "AlphaMiningEnvironment",
    "AtomicGradientProposal",
    "ComponentAttribution",
    "ComponentAttributor",
    "ConflictAwareSemanticGradientEngine",
    "EqualWeightZScoreCombiner",
    "Factor",
    "FactorCombiner",
    "FactorEdit",
    "FactorEditOperation",
    "FactorEvaluationError",
    "FactorEvaluator",
    "HamaTrainer",
    "Harness",
    "HarnessComponent",
    "HarnessEvolutionOptimizer",
    "HarnessOptimizationResult",
    "HarnessTrace",
    "InvocationKind",
    "InvocationRef",
    "MarketState",
    "MemoryBank",
    "MemoryEntry",
    "Message",
    "Model",
    "ParameterEdit",
    "ParameterRef",
    "PolicyEvaluation",
    "PrefillScore",
    "PrefillScorer",
    "QlibEvaluatorConfig",
    "QlibFactorEvaluator",
    "QwenPrefillScorer",
    "Skill",
    "SkillLibrary",
    "ToolCall",
    "ToolRunResult",
    "TrainingConfig",
    "TrainingResult",
    "TrainingRound",
    "Transition",
    "estimate_ema_advantages",
    "evaluate_factor_pool_splits",
    "evaluate_policy",
    "optimize_harness_with_attribution",
    "render_prefill",
    "rollout",
]
