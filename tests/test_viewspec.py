from palette_gen.solvers import ViewingSpec

NIGHT_VIEW_LIGHT = ViewingSpec(
    "night_light", T=6500, Lsw=1, Lmax=15, Lb=8, bg_hex="#e8e8e8"
)
DAY_VIEW_LIGHT = ViewingSpec(
    "day_light", T=6500, Lsw=100, Lmax=60, Lb=40, bg_hex="#e8e8e8"
)
