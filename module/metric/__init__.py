from .audiometrics import (
    ScaleInvariantSignalDistortionRatio,
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    MeanSquaredError,
)

METRIC_MAP = {
    "si_sdr": ScaleInvariantSignalDistortionRatio,
    "ScaleInvariantSignalDistortionRatio": ScaleInvariantSignalDistortionRatio,
    "stoi": ShortTimeObjectiveIntelligibility,
    "ShortTimeObjectiveIntelligibility": ShortTimeObjectiveIntelligibility,
    "pesq": PerceptualEvaluationSpeechQuality,
    "PerceptualEvaluationSpeechQuality": PerceptualEvaluationSpeechQuality,
    "mse": MeanSquaredError,
    "MeanSquaredError": MeanSquaredError,
}
