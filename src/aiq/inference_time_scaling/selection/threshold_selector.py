import logging
from typing import List

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_its_strategy
from aiq.inference_time_scaling.models.its_item import ITSItem
from aiq.inference_time_scaling.models.selection_config import ThresholdSelectionConfig
from aiq.inference_time_scaling.models.stage_enums import PipelineTypeEnum
from aiq.inference_time_scaling.models.stage_enums import StageTypeEnum
from aiq.inference_time_scaling.models.strategy_base import StrategyBase

logger = logging.getLogger(__name__)


class ThresholdSelector(StrategyBase):
    """
    Downselects only those ITSItems whose 'score' >= config.threshold.
    """

    async def build_components(self, builder: Builder) -> None:
        # No special components needed
        pass

    def supported_pipeline_types(self) -> List[PipelineTypeEnum]:
        return [PipelineTypeEnum.TOOL_USE]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.SELECTION

    async def ainvoke(self,
                      items: List[ITSItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> List[ITSItem]:
        threshold = self.config.threshold
        selected = [itm for itm in items if (itm.score is not None and itm.score >= threshold)]
        logger.info("ThresholdSelector: %d items => %d items (threshold=%.1f)", len(items), len(selected), threshold)
        return selected


@register_its_strategy(config_type=ThresholdSelectionConfig)
async def register_threshold_selector(config: ThresholdSelectionConfig, builder: Builder):
    selector = ThresholdSelector(config)
    yield selector
