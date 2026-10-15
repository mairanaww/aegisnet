"""
AegisNet API — Data models and schemas.
Pydantic schemas for request validation and response serialization.
"""

from typing import Any

from pydantic import BaseModel, Field


class FlowFeatures(BaseModel):
    """
    Network flow feature representation for AegisNet analysis.
    Maps common network flow properties to CICIDS-2017 feature format.
    """

    flow_duration: float | None = Field(default=0.0, description="Flow duration in microseconds")
    total_fwd_packets: float | None = Field(
        default=0.0, description="Total packets in the forward direction"
    )
    total_backward_packets: float | None = Field(
        default=0.0, description="Total packets in the backward direction"
    )
    total_length_fwd_packets: float | None = Field(
        default=0.0, description="Total size of packet in forward direction"
    )
    total_length_bwd_packets: float | None = Field(
        default=0.0, description="Total size of packet in backward direction"
    )
    fwd_packet_length_mean: float | None = Field(
        default=0.0, description="Mean size of packet in forward direction"
    )
    bwd_packet_length_mean: float | None = Field(
        default=0.0, description="Mean size of packet in backward direction"
    )
    flow_bytes_per_sec: float | None = Field(default=0.0, description="Flow byte rate (bytes/s)")
    flow_packets_per_sec: float | None = Field(
        default=0.0, description="Flow packet rate (packets/s)"
    )
    flow_iat_mean: float | None = Field(
        default=0.0, description="Mean time between two packets sent in the flow"
    )
    syn_flag_count: float | None = Field(default=0.0, description="Number of packets with SYN flag")
    ack_flag_count: float | None = Field(default=0.0, description="Number of packets with ACK flag")
    psh_flag_count: float | None = Field(default=0.0, description="Number of packets with PSH flag")
    fin_flag_count: float | None = Field(default=0.0, description="Number of packets with FIN flag")
    rst_flag_count: float | None = Field(default=0.0, description="Number of packets with RST flag")
    packet_length_variance: float | None = Field(
        default=0.0, description="Variance of packet length"
    )
    fwd_packet_length_max: float | None = Field(
        default=0.0, description="Maximum size of packet in forward direction"
    )
    fwd_packet_length_min: float | None = Field(
        default=0.0, description="Minimum size of packet in forward direction"
    )
    bwd_packet_length_max: float | None = Field(
        default=0.0, description="Maximum size of packet in backward direction"
    )
    bwd_packet_length_min: float | None = Field(
        default=0.0, description="Minimum size of packet in backward direction"
    )
    flow_iat_std: float | None = Field(
        default=0.0, description="Standard deviation time between two packets sent in flow"
    )
    flow_iat_max: float | None = Field(
        default=0.0, description="Maximum time between two packets sent in the flow"
    )
    flow_iat_min: float | None = Field(
        default=0.0, description="Minimum time between two packets sent in the flow"
    )
    down_up_ratio: float | None = Field(default=0.0, description="Download and upload ratio")
    average_packet_size: float | None = Field(default=0.0, description="Average size of packets")
    source_ip: str | None = Field(default=None, description="Source IP address of the flow session")

    def to_cicids_dict(self) -> dict[str, Any]:
        """
        Convert the Pydantic model back to the CICIDS-2017 column dictionary format
        with required leading and trailing spaces.
        """
        mapping = {
            "flow_duration": " Flow Duration",
            "total_fwd_packets": " Total Fwd Packets",
            "total_backward_packets": " Total Backward Packets",
            "total_length_fwd_packets": "Total Length of Fwd Packets",
            "total_length_bwd_packets": " Total Length of Bwd Packets",
            "fwd_packet_length_mean": " Fwd Packet Length Mean",
            "bwd_packet_length_mean": " Bwd Packet Length Mean",
            "flow_bytes_per_sec": " Flow Bytes/s",
            "flow_packets_per_sec": " Flow Packets/s",
            "flow_iat_mean": " Flow IAT Mean",
            "syn_flag_count": " SYN Flag Count",
            "ack_flag_count": " ACK Flag Count",
            "psh_flag_count": " PSH Flag Count",
            "fin_flag_count": " FIN Flag Count",
            "rst_flag_count": " RST Flag Count",
            "packet_length_variance": " Packet Length Variance",
            "fwd_packet_length_max": " Fwd Packet Length Max",
            "fwd_packet_length_min": " Fwd Packet Length Min",
            "bwd_packet_length_max": " Bwd Packet Length Max",
            "bwd_packet_length_min": " Bwd Packet Length Min",
            "flow_iat_std": " Flow IAT Std",
            "flow_iat_max": " Flow IAT Max",
            "flow_iat_min": " Flow IAT Min",
            "down_up_ratio": " Down/Up Ratio",
            "average_packet_size": " Average Packet Size",
        }

        cicids_dict: dict[str, Any] = {}
        for attr, cicids_col in mapping.items():
            val = getattr(self, attr)
            cicids_dict[cicids_col] = 0.0 if val is None else float(val)

        # Include space-padded alias for total length of fwd packets
        cicids_dict[" Total Length of Fwd Packets"] = cicids_dict["Total Length of Fwd Packets"]

        if self.source_ip is not None:
            cicids_dict[" Source IP"] = self.source_ip
            cicids_dict["Source IP"] = self.source_ip
        else:
            cicids_dict[" Source IP"] = "UNKNOWN"
            cicids_dict["Source IP"] = "UNKNOWN"

        return cicids_dict


class AnalysisResponse(BaseModel):
    """Result of flow session analysis."""

    anomaly_score: float = Field(..., description="Anomaly score in [0, 1] from Isolation Forest")
    attacker_profile: str = Field(..., description="Attacker profile classification from K-Means")
    risk_score: float = Field(..., description="Composite risk score in [0, 1]")
    risk_level: str = Field(..., description="Risk classification: LOW | MEDIUM | HIGH")
    risk_band: str = Field(..., description="Legacy risk band derived from anomaly score")
    cluster_id: int = Field(..., description="Cluster identifier assigned by K-Means")
    timestamp: str = Field(..., description="ISO 8601 evaluation timestamp")
    risk_breakdown: dict[str, Any] = Field(
        default_factory=dict, description="Detailed component breakdown of risk score"
    )


class HealthResponse(BaseModel):
    """API health and readiness status."""

    status: str = Field(..., description="Health status (e.g., 'ok' or 'degraded')")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(
        ..., description="Whether all required ML model artifacts are loaded"
    )
    uptime_seconds: float = Field(
        ..., description="Number of seconds the API server has been running"
    )


class ErrorResponse(BaseModel):
    """Standard error response structure."""

    error: str = Field(..., description="Error message or error category")
    detail: str | None = Field(default=None, description="Detailed diagnostic or error description")


class RetrainRequest(BaseModel):
    """Model retrain request payload."""

    dry_run: bool = Field(
        default=True,
        description="When true, models are validated without overwriting active models",
    )
    min_samples: int = Field(
        default=500, description="Minimum new attack samples required to perform retrain"
    )


class RetrainResponse(BaseModel):
    """Model retrain execution result."""

    success: bool = Field(
        ..., description="Indicates whether the retrain cycle completed successfully"
    )
    message: str = Field(..., description="Status message or summary of retrain operation")
    metrics: dict[str, Any] | None = Field(
        default=None, description="Training and evaluation metrics if available"
    )


class BatchAnalysisRequest(BaseModel):
    """Batch flow analysis request."""

    flows: list[FlowFeatures] = Field(..., description="List of network flow features to analyze")


class BatchAnalysisResponse(BaseModel):
    """Batch flow analysis response."""

    results: list[AnalysisResponse] = Field(
        ..., description="List of analysis results for each flow"
    )
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
