"""Harbor's Pi agent, with Claude Opus 5.5 registered for adaptive thinking.

Pi 1.1.0 does not list claude-opus-5-5, so Harbor registers it as a custom model. Without
compat.forceAdaptiveThinking Pi sends budget-based thinking, which Opus 5.5 rejects with a 400;
with it Pi sends adaptive thinking and maps its thinking level to Anthropic's effort. Pi itself
is unchanged.
"""

from harbor.agents.installed.pi import Pi


class PiAdaptive(Pi):
    def _build_custom_models_json(self, access, model_id):
        models = super()._build_custom_models_json(access, model_id)
        for provider in (models or {}).get("providers", {}).values():
            for model in provider["models"]:
                model["compat"] = {"forceAdaptiveThinking": True}
        return models
