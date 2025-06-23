from gymnasium.envs.registration import register

register(
    id="dfc_gymnasium/UtilityScalePVBESS-v0",
    entry_point="dfc_gymnasium.envs:UtilityScalePVBESS",
)

register(
	id="dfc_gymnasium/UtilityScalePVBESS-v0-nocurtailment",
	entry_point="dfc_gymnasium.envs:UtilityScalePVBESSnC",
)